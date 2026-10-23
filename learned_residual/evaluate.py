"""
Testing of trained residual, full pulse TORAX runs of test shots with best weights scored against same shots tuned baseline

Every shot gets objective loss, median sim/measured ratio, RMSE, reduced chi^2 per channel for baseline and hybrid over 
whole window, first 20%, and remaining. Summary gives skill score, 1 - hybrid / baseline mean loss by split, cluster, heating
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import get_context

import numpy as np
import pandas as pd
import xarray as xr

projectDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, projectDir)
from learned_residual import objective
from learned_residual import tglfnn_residual_model as model
from learned_residual.calibrate_constants import loadConfig, runTorax, setBeamInputs, setGasFueling
from learned_residual.rerun_baseline import baselineRuns, constantsHash, measuredValues, profileNames, scalarNames
from learned_residual.shot_splits import testHash

splitsPath = os.path.join(projectDir, 'MAST_info', 'shot_splits.csv')
manifestPath = os.path.join(projectDir, 'MAST_info', 'shot_splits_manifest.json')
constantsPath = os.path.join(projectDir, 'pulse_database', 'calibration', 'chosen_constants.json')
baselineLog = os.path.join(projectDir, 'pulse_database', 'baseline_runs.csv')
trainingDir = os.path.join(projectDir, 'pulse_database', 'training', 'runs')
defaultOut = os.path.join(projectDir, 'pulse_database', 'evaluation')
channelScales = {'Te': 1.0, 'ne': 1e19, 'Ti': 1.0}

def checkTestHash():
    """
    Test hash of current splits file vs splits doc
    """
    with open(manifestPath) as manifestFile:
        frozen = json.load(manifestFile)['testHash']
    current = testHash(pd.read_csv(splitsPath))
    if current != frozen:
        raise SystemExit(f'test shots changed since the splits were frozen, hash {current} against {frozen}')
    return current

def evaluationRuns(batch, splits, constants, logPath=baselineLog):
    """
    Newest passed run of each shot in splits with baseline file for given constants
    """
    runs = baselineRuns(batch, splits=splits)
    log = pd.read_csv(logPath)
    log = log[(log['status'] == 'ok') & (log['constantsHash'] == constantsHash(constants))]
    passed = log.sort_values('builtAt').groupby('runId').tail(1).set_index('runId')['baselinePath']
    missing = runs[~runs['runId'].isin(passed.index)]
    if len(missing):
        print(f'{len(missing)} runs have no baseline for these constants and are left out, rerun_baseline --splits {" ".join(splits)} builds them', flush=True)
    runs = runs[runs['runId'].isin(passed.index)].copy()
    runs['baselinePath'] = runs['runId'].map(passed)
    return runs.reset_index(drop=True)

def loadTraining(name, directory=trainingDir):
    """
    Best weights and manifest of a training run
    """
    runDir = os.path.join(directory, name)
    with open(os.path.join(runDir, 'manifest.json')) as manifestFile:
        manifest = json.load(manifestFile)
    return np.load(os.path.join(runDir, 'best_weights.npy')), manifest

def buildHybrid(task):
    """
    One full pulse run with tuned constants and trained correction
    """
    start = time.time()
    row = {name: task[name] for name in ('runId', 'shotId', 'cluster', 'split', 'hybridPath')}
    try:
        config = loadConfig(task['runDir'])
        config['plasma_composition']['Z_eff'] = task['zeff']
        config = setGasFueling(config, task['fueling'], task['batchFueling'])
        if task['injectedBeam']:
            config = setBeamInputs(config, task['shotId'], task['beamEnergyKev'], 0.0, 0.25, task['heatingCoefficient'], task['absorptionBounds'])
        config['transport'] = model.residualTransportConfig(config['transport'], weights=task['weights'], hiddenSizes=task['hiddenSizes'], holdBase=False,
                                                            featureMean=task['featureMean'], featureSpread=task['featureSpread'], maxFactor=task['maxFactor'])
        profiles, scalars = runTorax(config)
        dataset = xr.merge([profiles[[name for name in profileNames if name in profiles]], scalars[[name for name in scalarNames if name in scalars]]])
        dataset = xr.merge([dataset, measuredValues(task['profilesPath'])])
        dataset.attrs.update({'runId': task['runId'], 'shotId': task['shotId'], 'training': task['training']})
        dataset.to_netcdf(task['hybridPath'] + '.tmp')
        os.replace(task['hybridPath'] + '.tmp', task['hybridPath'])
        row.update({'status': 'ok', 'error': ''})
    except Exception as e:
        row.update({'status': 'failed', 'error': f'{type(e).__name__}: {e}'[:300]})
    row['seconds'] = round(time.time() - start, 1)
    return row

def hybridTasks(runs, constants, weights, manifest, name, outDir):
    """
    One hybrid task per run with the constants of its cluster and trained weights
    """
    beam = constants['beam']
    tasks = []
    for _, run in runs.iterrows():
        profilesPath = os.path.join(projectDir, run['runProfilesPath'])
        tasks.append({'runId': run['runId'], 'shotId': int(run['shotId']), 'cluster': run['cluster'], 'split': run['split'], 'training': name,
                      'runDir': os.path.join(projectDir, run['runOutputDir']), 'profilesPath': profilesPath, 'hybridPath': os.path.join(outDir, f'{run["runId"]}_hybrid.nc'),
                      'zeff': float(constants['zeff'][run['cluster']]['value']), 'fueling': float(constants['fueling'][run['cluster']]['value']),
                      'batchFueling': float(run['cfgGasFuelingEfficiency']), 'injectedBeam': bool(run['shotHeating'] != 'Ohmic' and run.get('cfgBeamAbsorptionFraction') == 1.0),
                      'beamEnergyKev': beam['energyKev'], 'heatingCoefficient': beam['heatingCoefficient'], 'absorptionBounds': tuple(beam['absorptionBounds']),
                      'weights': np.asarray(weights), 'hiddenSizes': tuple(manifest['hiddenSizes']), 'maxFactor': manifest['maxFactor'],
                      'featureMean': np.asarray(manifest['featureMean']), 'featureSpread': np.asarray(manifest['featureSpread'])})
    return tasks

def runHybrids(tasks, workers, logPath):
    """
    Every task without hybrid file across worker processes
    """
    todo = [task for task in tasks if not os.path.exists(task['hybridPath'])]
    print(f'{len(tasks) - len(todo)} hybrid runs already built, {len(todo)} to run on {workers} workers', flush=True)
    start = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'), max_tasks_per_child=1) as pool:
        futures = [pool.submit(buildHybrid, task) for task in todo]
        for finished, future in enumerate(as_completed(futures), 1):
            row = future.result()
            pd.DataFrame([row]).to_csv(logPath, mode='a', header=not os.path.exists(logPath), index=False)
            remaining = (time.time() - start) / finished * (len(todo) - finished)
            outcome = 'ok' if row['status'] == 'ok' else f'failed: {row["error"][:120]}'
            print(f'[{finished}/{len(todo)}] shot {row["shotId"]} {outcome} in {row["seconds"]:.0f} s | about {remaining / 3600:.1f} h left', flush=True)

def channelMetrics(path, shotId, tStart=None, tEnd=None):
    """
    Objective loss, per channel loss, median sim/measured ratio, RMSE, reduced chi squared, point count of run between tStart and tEnd
    """
    profiles = objective.loadProfiles(path)
    targets = objective.loadTargets(shotId, tStart if tStart is not None else float(profiles['time'][0]), tEnd if tEnd is not None else float(profiles['time'][-1]))
    result = {'loss': float(objective.shotLoss(profiles, targets)) if targets else np.nan}
    for name, points in targets.items():
        simulated = np.asarray(objective.simAtPoints(profiles[name], profiles['time'], profiles['rho'], points['times'], points['rho']))
        mask = np.asarray(points['mask'])
        measured, error = np.asarray(points['measured']), np.asarray(objective.normalizedError(simulated, points['measured'], points['sigma'], points['mask']))
        result.update({f'{name}Loss': float(objective.channelLoss(simulated, points['measured'], points['sigma'], points['mask'])), f'{name}Points': int(mask.sum()),
                       f'{name}Ratio': float(np.median(simulated[mask] / measured[mask])), f'{name}Rmse': float(np.sqrt(np.mean(((simulated - measured)[mask] / channelScales[name]) ** 2))),
                       f'{name}ChiSquared': float(np.mean(error[mask] ** 2))})
    return result

def scoreShots(runs, hybridPaths, earlyFraction=0.2):
    """
    Baseline and hybrid metrics of every shot with hybrid file over whole window, first earlyFraction, remaining
    """
    rows = []
    for _, run in runs.iterrows():
        hybridPath = hybridPaths.get(run['runId'])
        if hybridPath is None or not os.path.exists(hybridPath):
            continue
        with xr.open_dataset(run['baselinePath']) as data:
            t0, t1 = float(data['time'][0]), float(data['time'][-1])
        cut = t0 + earlyFraction * (t1 - t0)
        row = {'runId': run['runId'], 'shotId': int(run['shotId']), 'cluster': run['cluster'], 'split': run['split'], 'heating': run['shotHeating']}
        for label, path in (('baseline', run['baselinePath']), ('hybrid', hybridPath)):
            row.update({f'{label}_{key}': value for key, value in channelMetrics(path, int(run['shotId'])).items()})
            row[f'{label}_earlyLoss'] = channelMetrics(path, int(run['shotId']), t0, cut)['loss']
            row[f'{label}_lateLoss'] = channelMetrics(path, int(run['shotId']), cut, t1)['loss']
        rows.append(row)
    return pd.DataFrame(rows)

def skill(table, loss='loss'):
    """
    1 - hybrid mean loss / baseline mean loss
    """
    return 1.0 - table[f'hybrid_{loss}'].mean() / table[f'baseline_{loss}'].mean()

def familyBootstrap(table, loss='loss', draws=2000, seed=0):
    """
    2.5 and 97.5 percentiles of skill score / resamples of whole shot families with replacement
    """
    rng = np.random.default_rng(seed)
    groups = [group for _, group in table.groupby('family')]
    scores = [skill(pd.concat([groups[i] for i in rng.integers(len(groups), size=len(groups))]), loss) for _ in range(draws)]
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))

def summarize(table):
    """
    Shot count, baseline and hybrid mean loss, skill score with interval, fraction of shots improved for every
    split, cluster, heating group, over whole window and its two parts
    """
    rows = []
    groupings = [('all', None), ('split', 'split'), ('cluster', 'cluster'), ('heating', 'heating')]
    for kind, column in groupings:
        groups = [('all', table)] if column is None else list(table.groupby(column))
        for value, group in groups:
            for loss in ('loss', 'earlyLoss', 'lateLoss'):
                valid = group.dropna(subset=[f'baseline_{loss}', f'hybrid_{loss}'])
                if valid.empty:
                    continue
                low, high = familyBootstrap(valid, loss)
                rows.append({'group': kind, 'value': value, 'part': loss, 'shots': len(valid), 'baselineLoss': valid[f'baseline_{loss}'].mean(),
                             'hybridLoss': valid[f'hybrid_{loss}'].mean(), 'skill': skill(valid, loss), 'skillLow': low, 'skillHigh': high,
                             'improvedFraction': float((valid[f'hybrid_{loss}'] < valid[f'baseline_{loss}']).mean())})
    return pd.DataFrame(rows)

def evaluate(name, batch, constants, splits=('test', 'testCampaign'), workers=3, scoreOnly=False, outDir=defaultOut):
    """
    Hybrid runs of test shots, per shot metrics, summary
    """
    hashValue = checkTestHash()
    resultDir = os.path.join(outDir, name)
    os.makedirs(os.path.join(resultDir, 'runs'), exist_ok=True)
    runs = evaluationRuns(batch, splits, constants)
    weights, manifest = loadTraining(name)
    tasks = hybridTasks(runs, constants, weights, manifest, name, os.path.join(resultDir, 'runs'))
    if not scoreOnly:
        runHybrids(tasks, workers, os.path.join(resultDir, 'hybrid_runs.csv'))
    table = scoreShots(runs, {task['runId']: task['hybridPath'] for task in tasks})
    table = table.merge(pd.read_csv(splitsPath)[['shot_id', 'family']], left_on='shotId', right_on='shot_id').drop(columns='shot_id')
    summary = summarize(table)
    for fileName, frame in (('per_shot.csv', table), ('summary.csv', summary)):
        with open(os.path.join(resultDir, fileName), 'x') as resultFile:
            frame.to_csv(resultFile, index=False)
    with open(os.path.join(resultDir, 'evaluation.json'), 'x') as resultFile:
        json.dump({'training': name, 'testHash': hashValue, 'constantsHash': constantsHash(constants), 'splits': list(splits), 'shots': len(table),
                   'scored': datetime.now().isoformat(timespec='seconds')}, resultFile, indent=1)
    print(summary[summary['group'].isin(['all', 'split'])].to_string(index=False), flush=True)
    return table, summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score a trained residual once on the test shots against the tuned baseline')
    parser.add_argument('--name', required=True, help='training run to score, a folder under pulse_database/training/runs')
    parser.add_argument('--batch', default='dbv5', help='batch the runs come from, by full name or by label')
    parser.add_argument('--constants', default=constantsPath, help='constants file the baseline and training used')
    parser.add_argument('--splits', nargs='+', default=['test', 'testCampaign'], help='splits to score')
    parser.add_argument('--workers', type=int, default=3, help='TORAX runs at once')
    parser.add_argument('--score-only', action='store_true', help='score existing hybrid files without running any')
    parser.add_argument('--out', default=defaultOut, help='folder for evaluation results')
    args = parser.parse_args()

    with open(args.constants) as constantsFile:
        constants = json.load(constantsFile)
    try:
        evaluate(args.name, args.batch, constants, tuple(args.splits), args.workers, args.score_only, args.out)
    except FileExistsError as e:
        raise SystemExit(f'{e.filename} already exists, the test is scored once per training run')
