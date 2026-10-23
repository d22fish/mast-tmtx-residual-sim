"""
Training of the learned residual on the training windows, gradient through TORAX with TGLF-NN fixed

AdamW on the batch mean gradient, validation loss after every epoch, determine best weights by validation loss
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from multiprocessing import get_context

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from torax import experimental

projectDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, projectDir)
from learned_residual import objective
from learned_residual import tglfnn_residual_model as model
from learned_residual import training_windows as tw

windowsDir = os.path.join(projectDir, 'pulse_database', 'training')
constantsPath = os.path.join(projectDir, 'pulse_database', 'calibration', 'chosen_constants.json')
defaultOut = os.path.join(projectDir, 'pulse_database', 'training', 'runs')

def memoryGb():
    """
    Memory of current process
    """
    return int(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(os.getpid())])) / 1e6

def penalty(weights, features, maxFactor, hiddenSizes):
    """
    Mean squared log multiplier over a window's start state features
    """
    return jnp.mean(jnp.log(model.multipliers(weights, features, maxFactor, hiddenSizes)) ** 2)

def windowLoss(weights, stepFn, targets, times, rho, features, penaltyWeight, maxSteps, maxFactor, hiddenSizes):
    """
    Objective of one window with given weights and penalty on the multipliers
    """
    provider = stepFn.runtime_params_provider.update_provider_from_mapping({'transport_model.weights': weights})
    states, _, finalStep = experimental.run_loop_jit(stepFn, maxSteps, provider)
    # Unused steps repeat the last state at times 1s apart
    steps = jnp.arange(maxSteps + 1)
    reached = jnp.minimum(steps, finalStep)
    simTimes = jnp.where(steps <= finalStep, states.t, states.t[finalStep] + (steps - finalStep))
    profiles = {'Te': states.core_profiles.T_e.value[reached], 'ne': states.core_profiles.n_e.value[reached], 'Ti': states.core_profiles.T_i.value[reached], 'time': simTimes, 'rho': rho}
    return objective.shotLoss(profiles, targets) + penaltyWeight * penalty(weights, features, maxFactor, hiddenSizes), times[-1] - states.t[finalStep]

def worker(connection, windows, runs, constants, settings, options):
    """
    Window owner in its own process, builds window once, returns loss and gradient requests over the pipe
    """
    hiddenSizes = tuple(options['hiddenSizes'])
    dt = options['dt']
    nSteps = int(round(settings['length'] / dt))
    built, template, rho = {}, None, None

    def build(index):
        """
        Step function, padded targets, step times, scaled start features of one window
        """
        nonlocal template, rho
        window = windows.loc[index]
        config = tw.windowConfig(runs.loc[window['runId']], float(window['t0']), constants, settings['length'], dt, hiddenSizes)
        stepFn = tw.stepFunction(config)
        if template is None:
            template = stepFn
            rho = jnp.asarray(stepFn.geometry_provider(float(window['t0'])).rho_norm)
        stepFn = tw.shareCompiled(template, stepFn)
        state, _ = experimental.get_initial_state_and_post_processed_outputs(stepFn)
        features = model.standardize(jnp.asarray(model.stateFeatures(stepFn, state)), jnp.asarray(settings['featureMean']), jnp.asarray(settings['featureSpread']))
        targets = {name: {key: jnp.asarray(value) for key, value in points.items()} for name, points in tw.windowTargets(int(window['shotId']), float(window['t0']), float(window['t1']), settings['targetSizes']).items()}
        times = jnp.asarray(float(window['t0']) + dt * np.arange(nSteps + 1))
        return stepFn, targets, times, features

    lossFn = lambda w, fn, tg, tm, rh, ft: windowLoss(w, fn, tg, tm, rh, ft, options['penalty'], nSteps + 4, options['maxFactor'], hiddenSizes)
    valueAndGrad = jax.jit(jax.value_and_grad(lossFn, has_aux=True))
    value = jax.jit(lossFn)
    while True:
        kind, weights, indices = connection.recv()
        if kind == 'stop':
            break
        weights = jnp.asarray(weights)
        lossSum, gradSum, used, skipped = 0.0, np.zeros(len(weights)), 0, []
        for index in indices:
            if index in built and built[index] is None:
                skipped.append((int(index), 'window failed to build'))
                continue
            try:
                if index not in built:
                    built[index] = None
                    built[index] = build(index)
                stepFn, targets, times, features = built[index]
                if kind == 'grad':
                    (loss, shortfall), grad = valueAndGrad(weights, stepFn, targets, times, rho, features)
                    grad = np.asarray(grad)
                else:
                    (loss, shortfall), grad = value(weights, stepFn, targets, times, rho, features), 0.0
                loss = float(loss)
                if float(shortfall) > 1e-9:
                    raise RuntimeError(f'TORAX stopped {float(shortfall) * 1e3:.2f} ms before the window end')
                if not np.isfinite(loss) or not np.all(np.isfinite(grad)):
                    raise FloatingPointError('non-finite loss or gradient')
                lossSum, gradSum, used = lossSum + loss, gradSum + grad, used + 1
            except Exception as e:
                skipped.append((int(index), f'{type(e).__name__}: {e}'[:200]))
        connection.send((lossSum, gradSum, used, skipped, memoryGb()))

class WorkerPool:
    """
    Worker processes that each own the windows 
    """
    def __init__(self, count, windows, runs, constants, settings, options, memoryLimit):
        """
        Starts count workers
        """
        self.count, self.memoryLimit = count, memoryLimit
        self.arguments = (windows, runs, constants, settings, options)
        self.context = get_context('spawn')
        self.workers = [self.start() for _ in range(count)]

    def start(self):
        """
        One worker process and parent end of its pipe
        """
        parentEnd, childEnd = self.context.Pipe()
        process = self.context.Process(target=worker, args=(childEnd, *self.arguments), daemon=True)
        process.start()
        return process, parentEnd

    def request(self, kind, weights, indices):
        """
        Loss sum, gradient sum, window count over the given windows, every worker runs its portion once
        """
        shares = [[index for index in indices if index % self.count == k] for k in range(self.count)]
        for (_, pipe), share in zip(self.workers, shares):
            pipe.send((kind, np.asarray(weights), share))
        lossSum, gradSum, used, skipped = 0.0, 0.0, 0, []
        for k, (process, pipe) in enumerate(self.workers):
            workerLoss, workerGrad, workerUsed, workerSkipped, memory = pipe.recv()
            lossSum, gradSum, used, skipped = lossSum + workerLoss, gradSum + workerGrad, used + workerUsed, skipped + workerSkipped
            if memory > self.memoryLimit:
                print(f'worker {k} at {memory:.1f} GB, replaced', flush=True)
                self.stopOne(k)
                self.workers[k] = self.start()
        return lossSum, gradSum, used, skipped

    def stopOne(self, k):
        """
        Stops one worker
        """
        process, pipe = self.workers[k]
        pipe.send(('stop', None, None))
        process.join(timeout=60)

    def stop(self):
        """
        Stops every worker
        """
        for k in range(self.count):
            self.stopOne(k)

def fileHash(path):
    """
    First 12 characters of the sha256 of a file
    """
    with open(path, 'rb') as hashedFile:
        return hashlib.sha256(hashedFile.read()).hexdigest()[:12]

def train(name, constants, settings, runs, windows, workers=3, epochs=30, batchSize=8, learningRate=1e-3, warmupSteps=50, weightDecay=1e-4,
          penaltyWeight=1e-2, patience=5, seed=0, hiddenSizes=(32, 32), maxFactor=5.0, memoryLimit=7.0, overfit=None, outDir=defaultOut, provenance=None):
    """
    AdamW on the batch mean gradient over shuffled train windows, validation loss after every epoch, early stop after
    patience epochs without improvement
    """
    runDir = os.path.join(outDir, name)
    os.makedirs(runDir, exist_ok=False)
    rng = np.random.default_rng(seed)
    trainIndex = list(windows.index[windows['split'] == 'train'])
    validationIndex = list(windows.index[windows['split'] == 'validation'])
    if overfit is not None:
        trainIndex = list(rng.permutation(trainIndex)[:overfit])
        validationIndex = trainIndex
    stepsPerEpoch = int(np.ceil(len(trainIndex) / batchSize))
    schedule = optax.warmup_cosine_decay_schedule(0.0, learningRate, min(warmupSteps, stepsPerEpoch * epochs - 1), stepsPerEpoch * epochs)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=weightDecay))
    weights = jnp.asarray(model.initWeights(seed, hiddenSizes))
    state = optimizer.init(weights)
    options = {'hiddenSizes': list(hiddenSizes), 'dt': 0.005, 'penalty': penaltyWeight, 'maxFactor': maxFactor}
    manifest = {'name': name, 'started': datetime.now().isoformat(timespec='seconds'), 'trainWindows': len(trainIndex), 'validationWindows': len(validationIndex),
                'workers': workers, 'epochs': epochs, 'batchSize': batchSize, 'learningRate': learningRate, 'warmupSteps': warmupSteps, 'weightDecay': weightDecay,
                'penaltyWeight': penaltyWeight, 'patience': patience, 'seed': seed, 'hiddenSizes': list(hiddenSizes), 'maxFactor': maxFactor, 'overfit': overfit,
                'featureMean': settings['featureMean'], 'featureSpread': settings['featureSpread'], **(provenance or {})}
    with open(os.path.join(runDir, 'manifest.json'), 'w') as manifestFile:
        json.dump(manifest, manifestFile, indent=1)
    pool = WorkerPool(workers, windows, runs, constants, settings, options, memoryLimit)
    history, best, sinceBest, skippedLog = [], np.inf, 0, []
    try:
        for epoch in range(epochs):
            start = time.time()
            order = rng.permutation(trainIndex)
            trainLoss, trainCount = 0.0, 0
            for step in range(stepsPerEpoch):
                batch = [int(i) for i in order[step * batchSize:(step + 1) * batchSize]]
                lossSum, gradSum, used, skipped = pool.request('grad', weights, batch)
                skippedLog += [(epoch, *item) for item in skipped]
                if used == 0:
                    continue
                updates, state = optimizer.update(jnp.asarray(gradSum / used), state, weights)
                weights = optax.apply_updates(weights, updates)
                trainLoss, trainCount = trainLoss + lossSum, trainCount + used
            validationSum, _, validationCount, skipped = pool.request('loss', weights, [int(i) for i in validationIndex])
            skippedLog += [(epoch, *item) for item in skipped]
            validationLoss = validationSum / max(validationCount, 1)
            row = {'epoch': epoch, 'trainLoss': trainLoss / max(trainCount, 1), 'validationLoss': validationLoss, 'trainWindows': trainCount,
                   'validationWindows': validationCount, 'skipped': len(skippedLog), 'seconds': round(time.time() - start, 1)}
            history.append(row)
            pd.DataFrame(history).to_csv(os.path.join(runDir, 'history.csv'), index=False)
            if validationLoss < best:
                best, sinceBest = validationLoss, 0
                np.save(os.path.join(runDir, 'best_weights.npy'), np.asarray(weights))
            else:
                sinceBest += 1
            print(f'epoch {epoch}: train {row["trainLoss"]:.4f}, validation {validationLoss:.4f}, best {best:.4f}, {row["seconds"]:.0f} s, {len(skippedLog)} windows skipped so far', flush=True)
            if sinceBest >= patience:
                print(f'no validation improvement in {patience} epochs, stopping', flush=True)
                break
    finally:
        pool.stop()
        pd.DataFrame(skippedLog, columns=['epoch', 'window', 'error']).to_csv(os.path.join(runDir, 'skipped_windows.csv'), index=False)
    return history

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the learned residual on the training windows')
    parser.add_argument('--name', required=True, help='name of this training run, a folder under --out')
    parser.add_argument('--batch', default='dbv5', help='batch the runs come from, by full name or by label')
    parser.add_argument('--constants', default=constantsPath, help='constants file the baseline and windows used')
    parser.add_argument('--windows', default=windowsDir, help='folder with windows.csv and windows.json from training_windows')
    parser.add_argument('--source', default='baseline', choices=['baseline', 'profiles'], help='where the window start Ti and psi come from')
    parser.add_argument('--workers', type=int, default=3, help='worker processes')
    parser.add_argument('--epochs', type=int, default=30, help='largest number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='windows per gradient step')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='peak learning rate')
    parser.add_argument('--penalty', type=float, default=1e-2, help='weight of the mean squared log multiplier penalty')
    parser.add_argument('--patience', type=int, default=5, help='epochs without validation improvement before stopping')
    parser.add_argument('--seed', type=int, default=0, help='seed for the weights and the window order')
    parser.add_argument('--memory-limit', type=float, default=7.0, help='worker memory in GB above which the worker is replaced')
    parser.add_argument('--overfit', type=int, help='train and validate on this many train windows only')
    parser.add_argument('--out', default=defaultOut, help='folder for training runs')
    args = parser.parse_args()

    with open(args.constants) as constantsFile:
        constants = json.load(constantsFile)
    with open(os.path.join(args.windows, 'windows.json')) as settingsFile:
        settings = json.load(settingsFile)
    constants.update(featureMean=settings['featureMean'], featureSpread=settings['featureSpread'])
    windows = pd.read_csv(os.path.join(args.windows, 'windows.csv'))
    runs = tw.trainingRuns(args.batch, tuple(settings['splits']), args.source).set_index('runId')
    windows = windows[windows['runId'].isin(runs.index)].reset_index(drop=True)
    provenance = {'constantsFile': args.constants, 'constantsHash': fileHash(args.constants), 'windowsHash': fileHash(os.path.join(args.windows, 'windows.csv')), 'source': args.source}
    train(args.name, constants, settings, runs, windows, args.workers, args.epochs, args.batch_size, args.learning_rate, penaltyWeight=args.penalty,
          patience=args.patience, seed=args.seed, memoryLimit=args.memory_limit, overfit=args.overfit, outDir=args.out, provenance=provenance)
