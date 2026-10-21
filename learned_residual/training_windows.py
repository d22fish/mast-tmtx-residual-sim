"""
Training windows of the learned residual, TORAX runs on frozen seed geometry of each run

Each window starts at a Thomson time with Te and ne from the measured profile and Ti and psi from the source run, uses all
time varying inputs, applies feature scaling
"""

import argparse
import json
import os
import sys

import jax
import numpy as np
import pandas as pd
import torax
import xarray as xr
from torax import experimental

projectDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, projectDir)
from learned_residual import objective
from learned_residual import tglfnn_residual_model as model
from learned_residual.calibrate_constants import loadConfig, setBeamInputs, setGasFueling

runsPath = os.path.join(projectDir, 'pulse_database', 'runs.csv')
splitsPath = os.path.join(projectDir, 'MAST_info', 'shot_splits.csv')
constantsPath = os.path.join(projectDir, 'pulse_database', 'calibration', 'constants.json')
baselineLog = os.path.join(projectDir, 'pulse_database', 'baseline_runs.csv')
defaultOut = os.path.join(projectDir, 'pulse_database', 'training')

def trainingRuns(batch, splits=('train', 'validation'), source='baseline', shots=None):
    """
    Newest passed run of each shot in batch in given splits with cluster, split, file the window
    start Ti and psi come from, run's baseline or profiles file
    """
    runs = pd.read_csv(runsPath, dtype={'configHash': str, 'codeHash': str})
    # Batch names of form YYYY-MM-DD_HHMM_<label>
    runs = runs[(runs['runStatus'] == 'ok') & ((runs['batchName'] == batch) | (runs['batchName'].str[16:] == batch))]
    runs = runs.sort_values('runTimestamp').groupby('shotId').tail(1)
    runs = runs.merge(pd.read_csv(splitsPath)[['shot_id', 'cluster', 'split']], left_on='shotId', right_on='shot_id')
    runs = runs[runs['split'].isin(splits)]
    if shots is not None:
        runs = runs[runs['shotId'].isin(shots)]
    runs['profilesPath'] = [os.path.join(projectDir, path) for path in runs['runProfilesPath']]
    if source == 'baseline':
        log = pd.read_csv(baselineLog)
        passed = log[log['status'] == 'ok'].sort_values('builtAt').groupby('runId').tail(1).set_index('runId')['baselinePath']
        runs = runs[runs['runId'].isin(passed.index)]
        runs['sourcePath'] = runs['runId'].map(passed)
    else:
        runs['sourcePath'] = runs['profilesPath']
    return runs.reset_index(drop=True)

def windowStarts(saved, length=0.08, minPoints=5, teFloor=0.005):
    """
    Start times of non-overlapping windows, at first Thomson time at or after the end of prior window
    with at least minPoints, measured Te and ne values above teFloor, 80ms windows
    """
    tEnd = float(saved['time'].values[-1])
    starts = []
    for i, t in enumerate(saved['thomson_time'].values):
        te, ne = saved['teMeasProfile'].values[i], saved['neMeasProfile'].values[i]
        good = np.isfinite(te) & np.isfinite(ne)
        usable = good.sum() >= minPoints and np.all(te[good] > teFloor)
        if usable and t + length <= tEnd and (not starts or t >= starts[-1] + length):
            starts.append(float(t))
    return starts

def seriesAt(series, times):
    """
    Values of a time series config entry, dict of times to values or (times, values) pair at the given times
    """
    if isinstance(series, dict):
        seriesTimes, values = np.array(list(series), dtype=float), np.array(list(series.values()), dtype=float)
    else:
        seriesTimes, values = (np.asarray(x, dtype=float) for x in series)
    order = np.argsort(seriesTimes)
    return np.interp(times, seriesTimes[order], values[order])

def measuredAt(saved, name, t0):
    """
    Measured Thomson time profile on profiles file rho grid, fill gaps by flat extension of nearest measured values
    """
    rho = saved['rho_norm'].values
    row = saved[name].sel(thomson_time=t0, method='nearest').values
    valid = np.isfinite(row)
    return np.interp(rho, rho[valid], row[valid])

def windowConfig(run, t0, constants, length=0.08, dt=0.005, hiddenSizes=model.defaultHidden):
    """
    TORAX config of one window, every time varying input on the window's step times
    """
    config = loadConfig(os.path.join(projectDir, run['runOutputDir']))
    config['plasma_composition']['Z_eff'] = float(constants['zeff'][run['cluster']]['value'])
    config = setGasFueling(config, float(constants['fueling'][run['cluster']]['value']), float(run['cfgGasFuelingEfficiency']))
    beam = constants['beam']
    if run['shotHeating'] != 'Ohmic' and run.get('cfgBeamAbsorptionFraction') == 1.0:
        config = setBeamInputs(config, int(run['shotId']), beam['energyKev'], 0.0, 0.25, beam['heatingCoefficient'], tuple(beam['absorptionBounds']))
    steps = t0 + dt * np.arange(int(round(length / dt)) + 1)
    conditions, sources = config['profile_conditions'], config['sources']
    for name in ('Ip', 'n_e_right_bc', 'T_e_right_bc', 'T_i_right_bc'):
        conditions[name] = (steps, seriesAt(conditions[name], steps))
    sources['gas_puff']['S_total'] = (steps, seriesAt(sources['gas_puff']['S_total'], steps))
    heat = sources.get('generic_heat', {'P_total': (steps, np.zeros_like(steps)), 'gaussian_location': 0.0, 'gaussian_width': 0.25})
    heat['P_total'] = (steps, seriesAt(heat['P_total'], steps))
    particle = sources.get('generic_particle', {'S_total': (steps, np.zeros_like(steps)), 'deposition_location': 0.0, 'particle_width': 0.25})
    particle['S_total'] = (steps, seriesAt(particle['S_total'], steps))
    sources['generic_heat'], sources['generic_particle'] = heat, particle
    with xr.open_dataset(run['profilesPath']) as saved, xr.open_dataset(run['sourcePath']) as start:
        rho = saved['rho_norm'].values
        # One profile under a fixed time key, constant over window
        key = np.array([0.0])
        conditions['T_e'] = (key, rho, measuredAt(saved, 'teMeasProfile', t0)[None])
        conditions['n_e'] = (key, rho, measuredAt(saved, 'neMeasProfile', t0)[None])
        startTime = float(np.clip(t0, start['time'].values[0], start['time'].values[-1]))
        conditions['T_i'] = (key, rho, start['T_i'].interp(time=startTime).values[None])
        conditions['psi'] = (key, rho, start['psi'].interp(time=startTime).values[None])
    conditions['initial_psi_mode'] = 'profile_conditions'
    config['numerics'].update({'t_initial': t0, 't_final': float(steps[-1]), 'fixed_dt': dt, 'adaptive_dt': False})
    config['solver']['solver_type'] = 'linear'
    config['transport'] = model.residualTransportConfig(config['transport'], hiddenSizes=hiddenSizes, holdBase=True,
                                                        featureMean=constants.get('featureMean'), featureSpread=constants.get('featureSpread'))
    return config

def stepFunction(config):
    """
    TORAX step function of a window config
    """
    return experimental.make_step_fn(torax.ToraxConfig.from_dict(config))

def leafLayout(tree):
    """
    Shape and dtype of every leaf of a pytree with its path
    """
    return [(jax.tree_util.keystr(path), np.shape(leaf), np.asarray(leaf).dtype) for path, leaf in jax.tree_util.tree_leaves_with_path(tree)]

def shareCompiled(template, stepFn):
    """
    Step function with the window's runtime params and geometry and template's solver and time step calculator
    """
    children, aux = stepFn.tree_flatten()
    templateChildren, templateAux = template.tree_flatten()
    if jax.tree_util.tree_structure(children) != jax.tree_util.tree_structure(templateChildren):
        raise ValueError('window runtime params or geometry differ in structure from the template')
    for (path, shape, dtype), (_, templateShape, templateDtype) in zip(leafLayout(children), leafLayout(templateChildren)):
        if shape != templateShape or dtype != templateDtype:
            raise ValueError(f'window leaf {path} is {shape} {dtype}, template has {templateShape} {templateDtype}')
    return type(template).tree_unflatten(templateAux, children)

def windowTargets(shotId, t0, t1, sizes):
    """
    Target points of one window from the objective
    """
    targets = objective.loadTargets(shotId, t0, t1)
    padded = {}
    for name, (nTimes, nChannels) in sizes.items():
        points = {key: np.asarray(value) for key, value in targets.get(name, {}).items()}
        if not points:
            points = {'times': np.full(1, t0), 'rho': np.full((1, 1), 0.5), 'measured': np.ones((1, 1)), 'sigma': np.ones((1, 1)), 'mask': np.zeros((1, 1), dtype=bool)}
        times = np.concatenate([points['times'], np.full(nTimes - len(points['times']), t0)])
        grow = lambda array, fill: np.pad(array, ((0, nTimes - array.shape[0]), (0, nChannels - array.shape[1])), constant_values=fill)
        padded[name] = {'times': times, 'rho': grow(points['rho'], 0.5), 'measured': grow(points['measured'], 1.0), 'sigma': grow(points['sigma'], 1.0), 'mask': grow(points['mask'], False)}
    return padded

def targetSizes(windows):
    """
    Largest time and channel count of each channel over every window
    """
    sizes = {name: [1, 1] for name in objective.channels}
    for window in windows.itertuples():
        for name, points in objective.loadTargets(window.shotId, window.t0, window.t1).items():
            sizes[name] = [max(sizes[name][0], int(points['mask'].shape[0])), max(sizes[name][1], int(points['mask'].shape[1]))]
    return sizes

def fitFeatureScaling(runs, windows, constants, count=200, seed=0):
    """
    Median and IQR over 25-75 percentile of each correction feature
    """
    rng = np.random.default_rng(seed)
    picked = windows.iloc[rng.permutation(len(windows))[:count]]
    byRun = runs.set_index('runId')
    rows = []
    for window in picked.itertuples():
        stepFn = stepFunction(windowConfig(byRun.loc[window.runId], window.t0, constants))
        state, _ = experimental.get_initial_state_and_post_processed_outputs(stepFn)
        rows.append(model.stateFeatures(stepFn, state)[1:-1])
    features = np.concatenate(rows)
    q1, median, q3 = np.percentile(features, [25, 50, 75], axis=0)
    spread = (q3 - q1) / 1.349
    return median, np.where(spread > 0, spread, 1.0), len(rows)

def buildWindows(batch, constants, splits=('train', 'validation'), source='baseline', shots=None, length=0.08, scalingCount=200, outDir=defaultOut):
    """
    Window table, padded target sizes, feature scaling for every run in the splits
    """
    os.makedirs(outDir, exist_ok=True)
    runs = trainingRuns(batch, splits, source, shots)
    rows = []
    for _, run in runs.iterrows():
        with xr.open_dataset(run['profilesPath']) as saved:
            starts = windowStarts(saved, length)
        rows += [{'runId': run['runId'], 'shotId': int(run['shotId']), 'cluster': run['cluster'], 'split': run['split'], 't0': t0, 't1': t0 + length} for t0 in starts]
    windows = pd.DataFrame(rows)
    windows.to_csv(os.path.join(outDir, 'windows.csv'), index=False)
    print(f'{len(windows)} windows of {length * 1e3:.0f} ms from {len(runs)} runs', flush=True)
    sizes = targetSizes(windows)
    median, spread, used = fitFeatureScaling(runs, windows[windows['split'] == 'train'], constants, scalingCount)
    settings = {'batch': batch, 'source': source, 'length': length, 'splits': list(splits), 'targetSizes': sizes, 'featureNames': list(model.featureNames),
                'featureMean': median.tolist(), 'featureSpread': spread.tolist(), 'scalingWindows': used}
    with open(os.path.join(outDir, 'windows.json'), 'w') as settingsFile:
        json.dump(settings, settingsFile, indent=1)
    print(f'target sizes {sizes}, feature scaling from {used} windows', flush=True)
    return windows, settings

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build the training windows, padded target sizes, and feature scaling of the learned residual')
    parser.add_argument('--batch', default='dbv5', help='batch the runs come from, by full name or by label')
    parser.add_argument('--constants', default=constantsPath, help='constants.json from calibrate_constants')
    parser.add_argument('--source', default='baseline', choices=['baseline', 'profiles'], help='where the window start Ti and psi come from')
    parser.add_argument('--splits', nargs='+', default=['train', 'validation'], help='splits to build windows for')
    parser.add_argument('--shots', type=int, nargs='+', help='shots to use, every shot of the splits when left out')
    parser.add_argument('--length', type=float, default=0.08, help='window length in s')
    parser.add_argument('--scaling-windows', type=int, default=200, help='train windows the feature scaling comes from')
    parser.add_argument('--out', default=defaultOut, help='folder for windows.csv and windows.json')
    args = parser.parse_args()

    with open(args.constants) as constantsFile:
        constants = json.load(constantsFile)
    buildWindows(args.batch, constants, args.splits, args.source, args.shots, args.length, args.scaling_windows, args.out)
