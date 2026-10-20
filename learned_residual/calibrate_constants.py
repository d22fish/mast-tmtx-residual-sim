"""
Per cluster input constants from sweeps on calibration shots, TORAX only on each shot's saved config and seeds

Zeff from consumed flux and gas fueling efficiency from volume averaged ne, vs EFIT and Thomson, NBI absorbed fraction
from line averaged density. Each stage writes scan table, constants.json holds the per cluster values
"""

import argparse
import json
import os
import pickle
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import get_context

import numpy as np
import pandas as pd
import torax
import xarray as xr

projectDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, projectDir)
from MAST_pulse_batch import beamAbsorbedFraction, loadBeamFueling, loadShotWindow

runsPath = os.path.join(projectDir, 'pulse_database', 'runs.csv')
splitsPath = os.path.join(projectDir, 'MAST_info', 'shot_splits.csv')
targetsDir = os.path.join(projectDir, 'pulse_database', 'targets')
defaultOut = os.path.join(projectDir, 'pulse_database', 'calibration')

def calibrationRuns(batch, shots=None):
    """
    Newest successful run of every calibration shot in a batch found by full batch name or by label with its cluster
    """
    runs = pd.read_csv(runsPath, dtype={'configHash': str, 'codeHash': str})
    # Batch names start with the date and time, YYYY-MM-DD_HHMM_, then the label
    runs = runs[(runs['runStatus'] == 'ok') & ((runs['batchName'] == batch) | (runs['batchName'].str[16:] == batch))]
    runs = runs.sort_values('runTimestamp').groupby('shotId').tail(1)
    splits = pd.read_csv(splitsPath)
    calibration = splits.loc[splits['split'] == 'calibration', ['shot_id', 'cluster', 'heating']]
    runs = runs.merge(calibration, left_on='shotId', right_on='shot_id')
    if shots is not None:
        runs = runs[runs['shotId'].isin(shots)]
    return runs.reset_index(drop=True)

def loadConfig(runDir):
    """
    Saved TORAX config of a run with geometry pointed at the run folder
    """
    with open(os.path.join(runDir, 'torax_config.pkl'), 'rb') as configFile:
        config = pickle.load(configFile)
    config['geometry']['geometry_directory'] = runDir
    return config

def runTorax(config):
    """
    TORAX alone on a config dict, profiles and scalars as datasets
    """
    dataTree, history = torax.run_simulation(torax.ToraxConfig.from_dict(config), log_timestep_info=False, progress_bar=False)
    if str(history.sim_error) != 'SimError.NO_ERROR':
        raise RuntimeError(f'TORAX stopped with {history.sim_error}')
    return dataTree.profiles.to_dataset(), dataTree.scalars.to_dataset()

def prescribedProfile(saved, name, minPoints=3, floor=0.0):
    """
    Measured profile from the profiles file as (times, rho, values) for TORAX, gaps filled by flat extension of the
    nearest measured values at each Thomson time, times with any value at or below floor left out
    """
    values = saved[name].values
    rho = saved['rho_norm'].values
    times, rows = [], []
    for t, row in zip(saved['thomson_time'].values, values):
        valid = np.isfinite(row)
        if valid.sum() >= minPoints and np.all(row[valid] > floor):
            times.append(float(t))
            rows.append(np.interp(rho, rho[valid], row[valid]))
    return np.array(times), rho, np.array(rows)

def interpretiveConfig(config, saved, teFloor=0.005):
    """
    Config with only the current evolving, Te, Ti, and ne prescribed from the measured Thomson profiles, Te above
    teFloor in keV
    """
    for name in ('evolve_electron_heat', 'evolve_ion_heat', 'evolve_density'):
        config['numerics'][name] = False
    te = prescribedProfile(saved, 'teMeasProfile', floor=teFloor)
    config['profile_conditions']['T_e'] = te
    config['profile_conditions']['T_i'] = te
    config['profile_conditions']['n_e'] = prescribedProfile(saved, 'neMeasProfile')
    return config

def consumedFlux(scalars, targetsFile, tStart, tEnd):
    """
    Simulated and EFIT consumed flux in Wb at every EFIT time in the window, both zero at the first EFIT time
    """
    with xr.open_dataset(targetsFile) as targets:
        efitTimes = targets['efit_time'].values
        psiBoundary = targets['efitPsiBoundary'].values
    inWindow = (efitTimes >= tStart) & (efitTimes <= tEnd) & np.isfinite(psiBoundary)
    efitTimes, psiBoundary = efitTimes[inWindow], psiBoundary[inWindow]
    measured = -(psiBoundary - psiBoundary[0]) * 2 * np.pi
    times = scalars['time'].values
    vLoop = scalars['v_loop_lcfs'].values
    simulated = np.concatenate([[0.0], np.cumsum(0.5 * (vLoop[1:] + vLoop[:-1]) * np.diff(times))])
    simulated = np.interp(efitTimes, times, simulated) - np.interp(efitTimes[0], times, simulated)
    return simulated, measured

def logRatioRms(scalars, simName, saved, measName):
    """
    RMS of log(sim/measured) at the Thomson times, sim read off the TORAX times
    """
    thomsonTimes = saved['thomson_time'].values
    measured = saved[measName].values
    simulated = np.interp(thomsonTimes, scalars['time'].values, scalars[simName].values)
    use = np.isfinite(measured) & (measured > 0) & (simulated > 0)
    if not use.any():
        raise ValueError(f'no Thomson times with {measName} in range')
    return float(np.sqrt(np.mean(np.log(simulated[use] / measured[use]) ** 2)))

def setGasFueling(config, efficiency, batchEfficiency):
    """
    Gas puff rate rescaled from the efficiency the batch ran with to the given efficiency
    """
    gasPuff = config['sources']['gas_puff']
    gasPuff['S_total'] = {t: rate * efficiency / batchEfficiency for t, rate in gasPuff['S_total'].items()}
    return config

def setBeamInputs(config, shotId, beamEnergyKev, beamLoc, beamWidth, heatingCoefficient, absorptionBounds):
    """
    NBI heat and particle source at the absorbed fraction from line averaged density
    """
    efm, _, _, _, _, _, ayc, _, _ = loadShotWindow(shotId)
    heat = config['sources']['generic_heat']
    heatTimes, heatPower = (np.asarray(x, dtype=float) for x in heat['P_total'])
    absorbedTimes, absorbed = beamAbsorbedFraction(dict(zip(heatTimes.tolist(), heatPower.tolist())), efm, ayc, heatingCoefficient, absorptionBounds)
    heat['P_total'] = (tuple(heatTimes), tuple(heatPower * np.interp(heatTimes, absorbedTimes, absorbed)))
    t0, t1 = config['numerics']['t_initial'], config['numerics']['t_final']
    fueling = loadBeamFueling(shotId, t0, t1, beamEnergyKev)
    fuelTimes = np.array(list(fueling))
    fuelRate = np.array(list(fueling.values())) * np.interp(fuelTimes, absorbedTimes, absorbed)
    config['sources']['generic_particle'] = {'deposition_location': beamLoc, 'particle_width': beamWidth, 'S_total': dict(zip(fuelTimes.tolist(), fuelRate.tolist()))}
    return config

def scoreTask(task):
    """
    One TORAX run of one shot at one sweep value, the stage score with the time taken
    """
    start = time.time()
    row = {name: task[name] for name in ('stage', 'shotId', 'cluster', 'value')}
    try:
        config = loadConfig(task['runDir'])
        targetsFile = os.path.join(targetsDir, f'{task["shotId"]}_targets.nc')
        with xr.open_dataset(task['profilesPath']) as saved:
            saved = saved.load()
        tStart, tEnd = config['numerics']['t_initial'], config['numerics']['t_final']
        config['plasma_composition']['Z_eff'] = task['zeff']
        if task['stage'] == 'zeff':
            _, scalars = runTorax(interpretiveConfig(config, saved))
            simulated, measured = consumedFlux(scalars, targetsFile, tStart, tEnd)
            row['score'] = float(np.sqrt(np.mean((simulated - measured) ** 2)))
        else:
            config = setGasFueling(config, task['fueling'], task['batchFueling'])
            if task['injectedBeam']:
                config = setBeamInputs(config, task['shotId'], task['beamEnergyKev'], task['beamLoc'], task['beamWidth'], task['heatingCoefficient'], task['absorptionBounds'])
            _, scalars = runTorax(config)
            row['score'] = logRatioRms(scalars, 'n_e_volume_avg', saved, 'neVolAvgMeas')
        row['error'] = ''
    except Exception as e:
        row['score'] = np.nan
        row['error'] = f'{type(e).__name__}: {e}'[:300]
    row['seconds'] = round(time.time() - start, 1)
    return row

def runTasks(tasks, workers, scanPath):
    """
    Every task without a successful row in the scan table across worker processes, failed rows rerun, each finished
    row appended to the scan table with progress and time left
    """
    try:
        saved = pd.read_csv(scanPath)
        saved = saved[saved['error'].isna() | (saved['error'] == '')]
    except FileNotFoundError:
        saved = pd.DataFrame(columns=['stage', 'shotId', 'cluster', 'value', 'score', 'error', 'seconds'])
    saved.to_csv(scanPath, index=False)
    finished = set(zip(saved['shotId'].astype(int), saved['value'].astype(float).round(6)))
    rows = saved.to_dict('records')
    tasks = [task for task in tasks if (task['shotId'], round(task['value'], 6)) not in finished]
    print(f'{len(rows)} runs already in {os.path.basename(scanPath)}, {len(tasks)} to run', flush=True)
    start = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        futures = [pool.submit(scoreTask, task) for task in tasks]
        for finished, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            pd.DataFrame([row]).to_csv(scanPath, mode='a', header=not os.path.exists(scanPath), index=False)
            remaining = (time.time() - start) / finished * (len(tasks) - finished)
            outcome = f'score {row["score"]:.4g}' if not row['error'] else f'failed: {row["error"][:120]}'
            print(f'[{finished}/{len(tasks)}] {row["stage"]} shot {row["shotId"]} value {row["value"]:g} {outcome} in {row["seconds"]:.0f} s | about {remaining / 3600:.1f} h left', flush=True)
    return pd.DataFrame(rows)

def bestValue(values, scores):
    """
    Sweep value at the minimum score, refined by a parabola through the minimum and its two neighbors, and whether
    the minimum sits at an end of the sweep
    """
    values, scores = np.asarray(values, dtype=float), np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    values, scores = values[finite], scores[finite]
    order = np.argsort(values)
    values, scores = values[order], scores[order]
    if np.ptp(scores) <= 1e-6 * np.abs(scores).max():
        raise ValueError('score does not change over the sweep')
    i = int(np.argmin(scores))
    if i == 0 or i == len(values) - 1:
        return float(values[i]), True
    a, b, c = np.polyfit(values[i - 1:i + 2], scores[i - 1:i + 2], 2)
    vertex = -b / (2 * a) if a > 0 else values[i]
    return float(np.clip(vertex, values[i - 1], values[i + 1])), False

def shotBest(scan):
    """
    Best value of every shot in a scan table, NaN for shots with fewer than three finished values or a flat score
    """
    rows = []
    for (shotId, cluster), group in scan.groupby(['shotId', 'cluster']):
        try:
            value, atEdge = bestValue(group['value'], group['score'])
            if group['score'].notna().sum() < 3:
                raise ValueError('too few finished values')
        except ValueError:
            value, atEdge = np.nan, False
        rows.append({'shotId': shotId, 'cluster': cluster, 'best': value, 'atEdge': atEdge})
    return pd.DataFrame(rows)

def clusterValues(best):
    """
    Median, interquartile spread, shot count, and count at a sweep end of the best values in each cluster
    """
    summary = {}
    for cluster, group in best.dropna(subset=['best']).groupby('cluster'):
        q1, median, q3 = np.percentile(group['best'], [25, 50, 75])
        summary[cluster] = {'value': float(median), 'spread': float(q3 - q1), 'shots': int(len(group)), 'atEdge': int(group['atEdge'].sum())}
    return summary

def zeffStage(runs, values, workers, outDir, borrowFrom='Ohmic_above550kA'):
    """
    Zeff of each Ohmic cluster from the consumed flux scan, beam clusters take the value of borrowFrom
    """
    ohmic = runs[runs['heating'] == 'Ohmic']
    tasks = [{'stage': 'zeff', 'shotId': int(run['shotId']), 'cluster': run['cluster'], 'value': float(value), 'zeff': float(value),
              'runDir': os.path.join(projectDir, run['runOutputDir']), 'profilesPath': os.path.join(projectDir, run['runProfilesPath'])}
             for _, run in ohmic.iterrows() for value in values]
    scan = runTasks(tasks, workers, os.path.join(outDir, 'zeff_scan.csv'))
    summary = clusterValues(shotBest(scan))
    try:
        borrowed = dict(summary[borrowFrom], borrowedFrom=borrowFrom)
    except KeyError:
        raise ValueError(f'no Zeff for {borrowFrom} to give the beam clusters, clusters with a Zeff are {sorted(summary)}')
    for cluster in runs['cluster'].unique():
        summary.setdefault(cluster, borrowed)
    return summary

def fuelingStage(runs, values, constants, workers, outDir, beamSettings):
    """
    Gas fueling efficiency of each cluster from the volume averaged ne scan, Zeff from the Zeff stage and NBI absorbed
    fraction from line averaged density
    """
    # Beam shots saved at full absorption hold the injected NBI power
    tasks = [dict(beamSettings, stage='fueling', shotId=int(run['shotId']), cluster=run['cluster'], value=float(value), fueling=float(value),
                  zeff=constants['zeff'][run['cluster']]['value'], batchFueling=float(run['cfgGasFuelingEfficiency']),
                  injectedBeam=run['heating'] != 'Ohmic' and run.get('cfgBeamAbsorptionFraction') == 1.0, runDir=os.path.join(projectDir, run['runOutputDir']), profilesPath=os.path.join(projectDir, run['runProfilesPath']))
             for _, run in runs.iterrows() for value in values]
    scan = runTasks(tasks, workers, os.path.join(outDir, 'fueling_scan.csv'))
    return clusterValues(shotBest(scan))

def saveConstants(constants, outDir):
    """
    Writes the per cluster constants with the time they were written
    """
    with open(os.path.join(outDir, 'constants.json'), 'w') as constantsFile:
        json.dump(dict(constants, written=datetime.now().isoformat(timespec='seconds')), constantsFile, indent=1)

def loadConstants(outDir):
    """
    Per cluster constants from an earlier calibration, empty when there are none
    """
    try:
        with open(os.path.join(outDir, 'constants.json')) as constantsFile:
            return json.load(constantsFile)
    except FileNotFoundError:
        return {}

def calibrate(batch, stages, zeffValues, fuelingValues, workers=4, outDir=defaultOut, shots=None, zeffBorrowFrom='Ohmic_above550kA', beamEnergyKev=75.0, beamLoc=0.0, beamWidth=0.25,
              beamHeatingCoefficient=0.66, beamAbsorptionBounds=(0.2, 0.8)):
    """
    Runs the chosen stages in order on the calibration shots, each stage reads the constants of the stages before it
    """
    os.makedirs(outDir, exist_ok=True)
    runs = calibrationRuns(batch, shots)
    constants = loadConstants(outDir)
    beamSettings = {'beamEnergyKev': beamEnergyKev, 'beamLoc': beamLoc, 'beamWidth': beamWidth, 'heatingCoefficient': beamHeatingCoefficient, 'absorptionBounds': beamAbsorptionBounds}
    nOhmic, nBeam = int((runs['heating'] == 'Ohmic').sum()), int((runs['heating'] != 'Ohmic').sum())
    counts = {'zeff': nOhmic * len(zeffValues), 'fueling': len(runs) * len(fuelingValues)}
    print(f'{len(runs)} calibration runs from {batch} ({nOhmic} Ohmic, {nBeam} beam), TORAX runs per stage {", ".join(f"{s} {counts[s]}" for s in stages)}', flush=True)
    constants['batch'] = batch
    if 'zeff' in stages:
        constants['zeff'] = zeffStage(runs, zeffValues, workers, outDir, zeffBorrowFrom)
        saveConstants(constants, outDir)
    if 'fueling' in stages:
        constants['fueling'] = fuelingStage(runs, fuelingValues, constants, workers, outDir, beamSettings)
        constants['beam'] = {'energyKev': beamEnergyKev, 'heatingCoefficient': beamHeatingCoefficient, 'absorptionBounds': list(beamAbsorptionBounds)}
        saveConstants(constants, outDir)
    return constants

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calibrate per cluster Zeff and gas fueling efficiency on the calibration shots')
    parser.add_argument('--batch', default='dbv5', help='batch the calibration runs come from, by full name or by label')
    parser.add_argument('--stages', nargs='+', default=['zeff', 'fueling'], choices=['zeff', 'fueling'], help='stages to run, in order')
    parser.add_argument('--workers', type=int, default=4, help='TORAX runs at once')
    parser.add_argument('--out', default=defaultOut, help='folder for the scan tables and constants.json')
    parser.add_argument('--shots', type=int, nargs='+', help='calibration shots to use, every calibration shot when left out')
    args = parser.parse_args()

    zeffValues = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
    fuelingValues = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    try:
        constants = calibrate(args.batch, args.stages, zeffValues, fuelingValues, workers=args.workers, outDir=args.out, shots=args.shots)
        print(json.dumps(constants, indent=1))
    except KeyError as e:
        traceback.print_exc()
        raise SystemExit(f'missing constants from an earlier stage, {e}, run the earlier stage first')
