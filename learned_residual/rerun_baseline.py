"""
Tuned baseline of every run in a batch, TORAX alone on each run's config and seeds with calibrated constants

Zeff and gas fueling efficiency from the shot's cluster in constants.json, NBI heat and beam particle source at the
absorbed fraction from line averaged density
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import get_context

import pandas as pd
import xarray as xr

projectDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, projectDir)
from learned_residual.calibrate_constants import loadConfig, runTorax, setBeamInputs, setGasFueling

runsPath = os.path.join(projectDir, 'pulse_database', 'runs.csv')
splitsPath = os.path.join(projectDir, 'MAST_info', 'shot_splits.csv')
constantsPath = os.path.join(projectDir, 'pulse_database', 'calibration', 'constants.json')
defaultLog = os.path.join(projectDir, 'pulse_database', 'baseline_runs.csv')
profileNames = ('T_e', 'T_i', 'n_e', 'n_i', 'psi', 'psi_norm', 'q', 'j_total', 'vpr', 'sigma_parallel', 'j_bootstrap', 'j_external')
scalarNames = ('Ip', 'P_ohmic_e', 'P_aux_total', 'P_radiation_e', 'P_SOL_total', 'W_thermal_e', 'W_thermal_total', 'T_e_volume_avg', 'n_e_volume_avg', 'tau_E')

def baselineRuns(batch, shots=None, splits=None):
    """
    Newest passed run of each shot in the batch with its cluster and split
    """
    runs = pd.read_csv(runsPath, dtype={'configHash': str, 'codeHash': str})
    # Batch names of form YYYY-MM-DD_HHMM_<label>
    runs = runs[(runs['runStatus'] == 'ok') & ((runs['batchName'] == batch) | (runs['batchName'].str[16:] == batch))]
    runs = runs.sort_values('runTimestamp').groupby('shotId').tail(1)
    runs = runs.merge(pd.read_csv(splitsPath)[['shot_id', 'cluster', 'split']], left_on='shotId', right_on='shot_id')
    if shots is not None:
        runs = runs[runs['shotId'].isin(shots)]
    if splits is not None:
        runs = runs[runs['split'].isin(splits)]
    return runs.reset_index(drop=True)

def constantsHash(constants):
    """
    First 12 characters of the sha256 of the Zeff, fueling, and beam constants
    """
    used = {name: constants[name] for name in ('zeff', 'fueling', 'beam')}
    return hashlib.sha256(json.dumps(used, sort_keys=True).encode()).hexdigest()[:12]

def measuredValues(profilesPath):
    """
    Measured values the run saved with its profiles
    """
    with xr.open_dataset(profilesPath) as saved:
        names = [name for name in saved.data_vars if any(key in name for key in ('Meas', 'EdgeBc', 'EdgeFit', 'edgePoint'))]
        return saved[names].load()

def buildBaseline(task):
    """
    One baseline run written to its baseline file
    """
    start = time.time()
    row = {name: task[name] for name in ('runId', 'shotId', 'cluster', 'split')}
    row.update({'baselinePath': task['baselinePath'], 'constantsHash': task['constantsHash']})
    try:
        config = loadConfig(task['runDir'])
        config['plasma_composition']['Z_eff'] = task['zeff']
        config = setGasFueling(config, task['fueling'], task['batchFueling'])
        if task['injectedBeam']:
            config = setBeamInputs(config, task['shotId'], task['beamEnergyKev'], task['beamLoc'], task['beamWidth'], task['heatingCoefficient'], task['absorptionBounds'])
        profiles, scalars = runTorax(config)
        dataset = xr.merge([profiles[[name for name in profileNames if name in profiles]], scalars[[name for name in scalarNames if name in scalars]]])
        dataset = xr.merge([dataset, measuredValues(task['profilesPath'])])
        dataset.attrs.update({'runId': task['runId'], 'shotId': task['shotId'], 'cluster': task['cluster'], 'constantsHash': task['constantsHash'],
                              'Zeff': task['zeff'], 'gasFuelingEfficiency': task['fueling'], 'beamAbsorbedFromDensity': int(task['injectedBeam'])})
        dataset.to_netcdf(task['baselinePath'] + '.tmp')
        os.replace(task['baselinePath'] + '.tmp', task['baselinePath'])
        row.update({'status': 'ok', 'error': '', 'zeff': task['zeff'], 'fueling': task['fueling']})
    except Exception as e:
        row.update({'status': 'failed', 'error': f'{type(e).__name__}: {e}'[:300], 'zeff': task['zeff'], 'fueling': task['fueling']})
    row.update({'seconds': round(time.time() - start, 1), 'builtAt': datetime.now().isoformat(timespec='seconds')})
    return row

def baselineTasks(runs, constants, outDir=None):
    """
    One task per run with the constants of its cluster
    """
    hashValue = constantsHash(constants)
    beam = constants['beam']
    tasks = []
    for _, run in runs.iterrows():
        profilesPath = os.path.join(projectDir, run['runProfilesPath'])
        baselineName = os.path.basename(profilesPath).replace('_profiles.nc', '_baseline.nc')
        tasks.append({'runId': run['runId'], 'shotId': int(run['shotId']), 'cluster': run['cluster'], 'split': run['split'],
                      'runDir': os.path.join(projectDir, run['runOutputDir']), 'profilesPath': profilesPath,
                      'baselinePath': os.path.join(outDir or os.path.dirname(profilesPath), baselineName), 'constantsHash': hashValue,
                      'zeff': float(constants['zeff'][run['cluster']]['value']), 'fueling': float(constants['fueling'][run['cluster']]['value']),
                      'batchFueling': float(run['cfgGasFuelingEfficiency']), 'injectedBeam': bool(run['shotHeating'] != 'Ohmic' and run.get('cfgBeamAbsorptionFraction') == 1.0),
                      'beamEnergyKev': beam['energyKev'], 'beamLoc': 0.0, 'beamWidth': 0.25, 'heatingCoefficient': beam['heatingCoefficient'],
                      'absorptionBounds': tuple(beam['absorptionBounds'])})
    return tasks

def buildBaselines(tasks, workers=4, logPath=defaultLog):
    """
    Every task without a passed row for the same constants in the log with no baseline file yet
    """
    try:
        log = pd.read_csv(logPath)
        done = set(log.loc[log['status'] == 'ok', 'runId'] + '_' + log.loc[log['status'] == 'ok', 'constantsHash'].astype(str))
    except FileNotFoundError:
        done = set()
    todo = [task for task in tasks if f'{task["runId"]}_{task["constantsHash"]}' not in done or not os.path.exists(task['baselinePath'])]
    print(f'{len(tasks) - len(todo)} baselines already built, {len(todo)} to run on {workers} workers', flush=True)
    start = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'), max_tasks_per_child=1) as pool:
        futures = [pool.submit(buildBaseline, task) for task in todo]
        for finished, future in enumerate(as_completed(futures), 1):
            row = future.result()
            pd.DataFrame([row]).to_csv(logPath, mode='a', header=not os.path.exists(logPath), index=False)
            remaining = (time.time() - start) / finished * (len(todo) - finished)
            outcome = 'ok' if row['status'] == 'ok' else f'failed: {row["error"][:120]}'
            print(f'[{finished}/{len(todo)}] shot {row["shotId"]} {outcome} in {row["seconds"]:.0f} s | about {remaining / 3600:.1f} h left', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rerun every run of a batch in TORAX alone with the calibrated constants')
    parser.add_argument('--batch', default='dbv5', help='batch to rerun, by full name or by label')
    parser.add_argument('--constants', default=constantsPath, help='constants.json from calibrate_constants')
    parser.add_argument('--workers', type=int, default=4, help='TORAX runs at once')
    parser.add_argument('--shots', type=int, nargs='+', help='shots to rerun, every shot of the batch when left out')
    parser.add_argument('--splits', nargs='+', help='splits to rerun, every split when left out')
    parser.add_argument('--out', help='folder for the baseline files, next to each profiles file when left out')
    parser.add_argument('--log', default=defaultLog, help='log of every baseline run')
    args = parser.parse_args()

    with open(args.constants) as constantsFile:
        constants = json.load(constantsFile)
    runs = baselineRuns(args.batch, args.shots, args.splits)
    try:
        tasks = baselineTasks(runs, constants, args.out)
    except KeyError as e:
        traceback.print_exc()
        raise SystemExit(f'constants file has no {e}, run calibrate_constants for every stage first')
    print(f'{len(runs)} runs from {args.batch}, constants {constantsHash(constants)}', flush=True)
    buildBaselines(tasks, args.workers, args.log)
