"""
Metrics of the pulse database written as text under metrics/, one folder per table with one file per batch
"""

import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr

from plot_db import heatingLabels, newestBatch, resolveBatch, runsPath, usablePath

catalogPath = os.path.join('MAST_info', 'fairmast_shots_catalog.parquet')
targetsDir = os.path.join('pulse_database', 'targets')
ratioColumns = [('Te core', 'resTeCoreRatio'), ('ne vol avg', 'resNeVolAvgRatio'), ('Ti core', 'resTiCoreRatio'), ('li', 'resLiRatio'), ('q95', 'resQ95Ratio')]

def databaseSummary(runs):
    """
    Database size, run health and median sim over measured ratios for the given runs, as name and value pairs
    """
    usable = pd.read_csv(usablePath)
    ok = runs[runs['runStatus'] == 'ok']
    metrics = [('Catalog pulses (count)', len(pd.read_parquet(catalogPath, columns=['shot_id']))), ('Usable pulses (count)', len(usable))]
    for heating, label in heatingLabels.items():
        metrics.append((f'Usable {label} pulses (count)', int((usable['heatingResolved'] == heating).sum())))
    metrics.append(('Successful runs (count)', len(ok)))
    for heating, label in heatingLabels.items():
        metrics.append((f'Successful {label} runs (count)', int((ok['shotHeating'] == heating).sum())))
    metrics.append(('Pulses ran end to end (count)', f'{ok["shotId"].nunique()}/{runs["shotId"].nunique()}'))
    metrics.append(('TokaMaker solves converged (count)', f'{int((ok["healthTmSolves"] - ok["healthTmFailed"]).sum())}/{int(ok["healthTmSolves"].sum())}'))
    metrics.append(('Median TokaMaker vs TORAX flux gap (%)', f'{100 * ok["healthFluxGap"].median():.1f}'))
    metrics.append(('Median power balance residual (%)', f'{100 * ok["healthPowerResidual"].median():.1f}'))
    # Median over runs of each run's sim over measured ratio
    for name, column in ratioColumns:
        metrics.append((f'Median {name} sim / measured (ratio)', f'{ok[column].median():.2f}'))
    fluxRatio = ok['resFluxConsumedSimWb'] / ok['resFluxConsumedMeasWb']
    metrics.append(('Median flux consumed sim / measured (ratio)', f'{fluxRatio.median():.2f}'))
    return metrics

def latestRuns(runs):
    """
    Newest successful run of each shot
    """
    return runs[runs['runStatus'] == 'ok'].sort_values('runTimestamp').groupby('shotId').tail(1)

def beamOnsets(times, beamPower, onFraction=0.5, offFraction=0.1, minGap=0.1):
    """
    Times the beam power crosses onFraction of its peak after sitting below offFraction of it
    """
    peak = np.nanmax(beamPower)
    onsets, below = [], True
    for t, power in zip(times, beamPower):
        if power < offFraction * peak:
            below = True
        elif below and power >= onFraction * peak:
            if not onsets or t - onsets[-1] > minGap:
                onsets.append(float(t))
            below = False
    return onsets

def riseTime(times, values, onset, before=0.02, settleStart=0.08, settleEnd=0.15):
    """
    Time after onset for values to cover 63 percent of the step from the level before onset to the settled level after
    it, NaN when the window misses either level or the step is too small to read
    """
    pre = values[(times >= onset - before) & (times < onset)]
    post = values[(times >= onset + settleStart) & (times <= onset + settleEnd)]
    try:
        start, end = np.nanmedian(pre), np.nanmedian(post)
        step = end - start
        if abs(step) < 0.1 * abs(start):
            return np.nan
        after = (times >= onset) & (times <= onset + settleEnd)
        crossed = np.flatnonzero((values[after] - start) / step >= 0.63)
        return float(times[after][crossed[0]] - onset)
    except (ValueError, IndexError):
        return np.nan

def slowingDownTime(te, ne, beamMass=2.0, coulombLog=17.0):
    """
    Fast ion slowing down time on electrons in s, Te in keV and ne in m^-3
    """
    return 0.2 * beamMass * te ** 1.5 / (ne / 1e20 * coulombLog)

def beamResponse(runs):
    """
    How fast volume averaged Te answers each beam turn on, measured against simulated with the fast ion slowing down
    time from the Te and ne before the turn on, as name and value pairs with one line per turn on
    """
    rows = []
    for _, run in latestRuns(runs[runs['shotHeating'] != 'Ohmic']).iterrows():
        try:
            profiles = xr.load_dataset(run['runProfilesPath'])
        except (OSError, ValueError) as e:
            print(f'no profiles file for shot {run["shotId"]}: {e}')
            continue
        txTimes, thomsonTimes = profiles['time'].values, profiles['thomson_time'].values
        for onset in beamOnsets(txTimes, profiles['P_aux_total'].values):
            pre = (thomsonTimes >= onset - 0.02) & (thomsonTimes < onset)
            te, ne = np.nanmedian(profiles['teVolAvgMeas'].values[pre]), np.nanmedian(profiles['neVolAvgMeas'].values[pre])
            rows.append({'shotId': run['shotId'], 'onset': onset, 'riseMeas': riseTime(thomsonTimes, profiles['teVolAvgMeas'].values, onset),
                         'riseSim': riseTime(txTimes, profiles['T_e_volume_avg'].values, onset), 'slowingDown': slowingDownTime(te, ne) if np.isfinite(te * ne) else np.nan})
    table = pd.DataFrame(rows, columns=['shotId', 'onset', 'riseMeas', 'riseSim', 'slowingDown'])
    both = table.dropna(subset=['riseMeas', 'riseSim'])
    metrics = [('Beam turn ons found (count)', len(table)), ('Beam turn ons with both rise times (count)', len(both)),
               ('Median measured Te rise time (ms)', f'{1e3 * both["riseMeas"].median():.0f}'), ('Median simulated Te rise time (ms)', f'{1e3 * both["riseSim"].median():.0f}'),
               ('Median fast ion slowing down time (ms)', f'{1e3 * both["slowingDown"].median():.0f}')]
    for row in table.itertuples():
        metrics.append((f'Shot {row.shotId} beam on at {row.onset:.3f} s, measured, simulated and slowing down (ms)', f'{1e3 * row.riseMeas:.0f}, {1e3 * row.riseSim:.0f}, {1e3 * row.slowingDown:.0f}'))
    return metrics

def channelPoints(profiles, targets, simName, measName, prefix, timeName, floorFraction):
    """
    Simulated and measured values, effective error and normalized error at every diagnostic point the loss uses,
    simulated profiles read at each point's time and rho
    """
    times = targets[timeName].values
    txTimes = profiles['time'].values
    at = xr.DataArray(np.clip(times, txTimes[0], txTimes[-1]), dims=timeName)
    simProfiles = profiles[simName].interp(time=at).values
    rho = profiles['rho_norm'].values
    pointRho = targets[f'{prefix}Rho'].values
    simulated = np.array([np.interp(pointRho[i], rho, simProfiles[i], left=np.nan, right=np.nan) for i in range(len(times))])
    measured, sigma = targets[f'{measName}Meas'].values, targets[f'{measName}Sigma'].values
    mask = (targets[f'{prefix}LossMask'].values == 1) & np.isfinite(simulated)
    sigmaEff = np.sqrt(sigma ** 2 + (floorFraction * measured) ** 2)
    return simulated[mask], measured[mask], sigmaEff[mask], ((simulated - measured) / sigmaEff)[mask]

def errorMetrics(runs, targetsPath=targetsDir, floorFraction=0.05):
    """
    Sim against measured at the raw diagnostic points for Te, ne and Ti, per channel the median over pulses of each
    pulse's median ratio, the RMSE in physical units and the reduced chi squared of the normalized error, as name and
    value pairs
    """
    channels = [('Te', 'T_e', 'te', 'thomson', 'thomson_time', 'keV', 1.0), ('ne', 'n_e', 'ne', 'thomson', 'thomson_time', '1e19 m^-3', 1e19),
                ('Ti', 'T_i', 'ti', 'cxrs', 'cxrs_time', 'keV', 1.0)]
    perChannel = {name: {'ratios': [], 'squared': [], 'normalized': []} for name, *_ in channels}
    withTargets = 0
    for _, run in latestRuns(runs).iterrows():
        try:
            profiles = xr.load_dataset(run['runProfilesPath'])
            targets = xr.load_dataset(os.path.join(targetsPath, f'{run["shotId"]}_targets.nc'))
        except (OSError, ValueError):
            continue
        withTargets += 1
        for name, simName, measName, prefix, timeName, _, scale in channels:
            if f'{prefix}LossMask' not in targets:
                continue
            simulated, measured, _, normalized = channelPoints(profiles, targets, simName, measName, prefix, timeName, floorFraction)
            if len(measured):
                perChannel[name]['ratios'].append(np.median(simulated / measured))
                perChannel[name]['squared'].append(((simulated - measured) / scale) ** 2)
                perChannel[name]['normalized'].append(normalized ** 2)
    metrics = [('Runs with a targets file (count)', f'{withTargets}/{len(latestRuns(runs))}'), ('Model error floor epsilon (fraction)', floorFraction)]
    for name, _, _, _, _, units, _ in channels:
        result = perChannel[name]
        if not result['ratios']:
            metrics.append((f'{name} pulses with points (count)', 0))
            continue
        squared, normalized = np.concatenate(result['squared']), np.concatenate(result['normalized'])
        metrics += [(f'{name} pulses with points (count)', len(result['ratios'])), (f'{name} points (count)', len(squared)),
                    (f'{name} median over pulses of the median sim / measured (ratio)', f'{np.median(result["ratios"]):.2f}'),
                    (f'{name} RMSE ({units})', f'{np.sqrt(np.mean(squared)):.3g}'), (f'{name} reduced chi squared (no units)', f'{np.mean(normalized):.2f}')]
    return metrics

def writeMetrics(metrics, path):
    """
    Writes one metric per line as name (units): value
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as metricsFile:
        metricsFile.write(''.join(f'{name}: {value}\n' for name, value in metrics))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write pulse database metrics as text under metrics/')
    parser.add_argument('--batch', help='this batch, by full name or by the label it ran with, the newest batch when left out')
    parser.add_argument('--all-batches', action='store_true', help='every run of every batch together, saved as all_batches.txt')
    parser.add_argument('--out', default='metrics', help='root folder for every metrics file, metrics when left out')
    parser.add_argument('--targets', default=targetsDir, help='folder with the measurement targets files for the error metrics')
    args = parser.parse_args()

    allRuns = pd.read_csv(runsPath, dtype={'configHash': str, 'codeHash': str})
    okRuns = allRuns[allRuns['runStatus'] == 'ok']
    batchName = None if args.all_batches else (resolveBatch(okRuns, args.batch) if args.batch else newestBatch(okRuns))
    runs = allRuns if batchName is None else allRuns[allRuns['batchName'] == batchName]
    path = os.path.join(args.out, 'database_summary', f'{batchName or "all_batches"}.txt')
    writeMetrics(databaseSummary(runs), path)
    print(f'{len(runs)} runs in {batchName or "every batch"}, metrics written to {path}')
    for table, metrics in (('beam_response', beamResponse(runs)), ('error_metrics', errorMetrics(runs, args.targets))):
        path = os.path.join(args.out, table, f'{batchName or "all_batches"}.txt')
        writeMetrics(metrics, path)
        print(f'{table} written to {path}')
