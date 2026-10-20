"""
Plots of the pulse database, every figure saved twice under plots/, once in an svg tree and once in a matching pdf tree
"""

import argparse
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import requests
import xarray as xr

BASE_URL = 'https://mastapp.site'
runsPath = os.path.join('pulse_database', 'runs.csv')
shotParametersPath = os.path.join('pulse_database', 'shot_parameters.csv')
usablePath = os.path.join('MAST_info', 'usable_pulses.csv')
heatingColors = {'Ohmic': 'tab:blue', 'SS Beam': 'tab:orange', 'SW Beam': 'tab:green', '2 Beams,SS Beam,SW Beam': 'tab:red'}
heatingLabels = {'Ohmic': 'Ohmic', 'SS Beam': 'SS beam', 'SW Beam': 'SW beam', '2 Beams,SS Beam,SW Beam': '2 beams'}
ratioColumns = [('resTeCoreRatio', 'core Te'), ('resTeVolAvgRatio', 'vol avg Te'), ('resNeCoreRatio', 'core ne'), ('resNeVolAvgRatio', 'vol avg ne'),
                ('resWeRatio', 'electron stored energy'), ('resTiCoreRatio', 'core Ti'), ('resLiRatio', 'li'), ('resQ95Ratio', 'q95')]

def loadRuns(batchName=None):
    """
    Successful runs from runs.csv, optionally only one batch
    """
    # Read hashes as text
    runs = pd.read_csv(runsPath, dtype={'configHash': str, 'codeHash': str})
    runs = runs[runs['runStatus'] == 'ok']
    if batchName is not None:
        runs = runs[runs['batchName'] == batchName]
    return runs.reset_index(drop=True)

def saveFigure(fig, outDir, name, folder=''):
    """
    Saves one figure as outDir/svg/folder/name.svg and outDir/pdf/folder/name.pdf, then closes it
    """
    for kind in ('svg', 'pdf'):
        path = os.path.join(outDir, kind, folder)
        os.makedirs(path, exist_ok=True)
        fig.savefig(os.path.join(path, f'{name}.{kind}'), format=kind, bbox_inches='tight')
    plt.close(fig)

def heatingLegend(ax):
    """
    One legend entry per heating type
    """
    handles = [plt.Line2D([], [], marker='o', linestyle='', color=color, label=heatingLabels[heating]) for heating, color in heatingColors.items()]
    ax.legend(handles=handles, fontsize=8)

def loadProfiles(run):
    """
    The run's saved profiles file, read fully into memory so the file closes right away
    """
    return xr.load_dataset(run['runProfilesPath'])

def burstAverage(times, *series, maxGap=0.002):
    """
    Averages Thomson lasers fired in a burst into one point at the burst's mean time, a new burst starts wherever the
    gap to the previous laser is over maxGap seconds, each series holds one value or one profile per time
    """
    times = np.asarray(times, dtype=float)
    burst = np.cumsum(np.concatenate([[False], np.diff(times) > maxGap]))[:len(times)]
    groups = [burst == b for b in np.unique(burst)]
    # Profile points outside Thomson coverage are NaN for a whole burst, numpy warns about those
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        averaged = [np.array([np.nanmean(np.asarray(values, dtype=float)[group], axis=0) for group in groups]) for values in series]
    return [np.array([times[group].mean() for group in groups])] + averaged

def pairedProfiles(profiles):
    """
    Measured Thomson profiles and the TORAX profiles at the same Thomson times on the TORAX rho grid, bursts averaged.
    Returns times, measured Te, simulated Te, measured ne, simulated ne
    """
    times = profiles['thomson_time'].values
    txTimes = profiles['time'].values
    # TORAX can stop a few ns before the last Thomson time from rounding, clipping keeps interp from returning NaN
    at = xr.DataArray(np.clip(times, txTimes[0], txTimes[-1]), dims='thomson_time')
    teSim = profiles['T_e'].interp(time=at).values
    neSim = profiles['n_e'].interp(time=at).values
    return burstAverage(times, profiles['teMeasProfile'].values, teSim, profiles['neMeasProfile'].values, neSim)

def safeRatio(numerator, denominator):
    """
    Numerator over denominator wherever the denominator is positive, NaN elsewhere
    """
    denominator = np.asarray(denominator, dtype=float)
    return np.asarray(numerator, dtype=float) / np.where(denominator > 0, denominator, np.nan)

def onGrid(rhoNew, rho, values):
    """
    Interpolates a profile onto another rho grid, NaN outside the stretch where the profile has values
    """
    finite = np.isfinite(values)
    return np.interp(rhoNew, rho[finite], values[finite], left=np.nan, right=np.nan) if finite.sum() >= 2 else np.full(len(rhoNew), np.nan)

def quietMedian(values, axis=None):
    """
    nanmedian without the warning numpy gives for an all NaN slice, which then comes back as NaN
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmedian(values, axis=axis)

def coveredMedian(values, axis=None, minFraction=0.5):
    """
    Median over the times that have a value, NaN when fewer than minFraction of the times do, so a point Thomson only
    reaches now and then (the core of a plasma sitting below the midplane) does not stand in for the whole pulse
    """
    values = np.asarray(values, dtype=float)
    enough = np.mean(np.isfinite(values), axis=axis) >= minFraction
    return np.where(enough, quietMedian(values, axis=axis), np.nan)

def ratioAxis(ax, low=0.25, high=4.0):
    """
    Log axis for sim over measured with plain number ticks, so 2x high and 2x low sit the same distance from 1
    """
    ax.set_yscale('log')
    ax.set_ylim(low, high)
    ticks = [tick for tick in (0.125, 0.25, 0.5, 1, 2, 4, 8) if low <= tick <= high]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{tick:g}' for tick in ticks])
    ax.minorticks_off()
    ax.axhline(1.0, color='k', lw=0.8)

def plotRatioOverview(runs, outDir):
    """
    Sim over measured for every quantity, one point per shot grouped by heating type, log axis so 2x high and 2x low look the same
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    heatingOrder = list(heatingColors)
    rng = np.random.default_rng(0)
    for ax, (column, label) in zip(axes.ravel(), ratioColumns):
        for i, heating in enumerate(heatingOrder):
            values = runs.loc[runs['shotHeating'] == heating, column].dropna()
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(values)), values, color=heatingColors[heating], s=20)
        ax.axhline(1.0, color='k', lw=0.8)
        ax.set_yscale('log')
        ax.set_xticks(range(len(heatingOrder)))
        ax.set_xticklabels([heatingLabels[heating] for heating in heatingOrder], fontsize=8)
        ax.set_title(f'{label} (n={runs[column].notna().sum()})')
        ax.set_ylabel('sim / measured')
    fig.suptitle('Sim over measured for each quantity, one point per run (median over its pulse), horizontal jitter added')
    fig.tight_layout()
    saveFigure(fig, outDir, 'ratio_overview', 'accuracy')

def plotRatioVsShot(runs, outDir):
    """
    Sim over measured against peak current and beam power, to see whether the misses depend on the operating point
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, (xColumn, xLabel) in enumerate((('shotIpMaxKA', 'peak Ip [kA]'), ('shotNbiMaxMW', 'peak NBI power [MW]'))):
        for col, (column, label) in enumerate((('resTeCoreRatio', 'core Te'), ('resNeVolAvgRatio', 'vol avg ne'), ('resWeRatio', 'electron stored energy'))):
            ax = axes[row, col]
            ax.scatter(runs[xColumn], runs[column], c=runs['shotHeating'].map(heatingColors), s=25)
            ax.axhline(1.0, color='k', lw=0.8)
            ax.set_yscale('log')
            ax.set_xlabel(xLabel)
            ax.set_ylabel(f'{label} sim / measured')
    heatingLegend(axes[0, 0])
    fig.suptitle('Simulated / measured against operating point, median over each pulse. Top: peak plasma current. Bottom: peak beam power (Ohmic at 0 MW)')
    fig.tight_layout()
    saveFigure(fig, outDir, 'ratio_vs_operating_point', 'accuracy')

def pulseMeans(runs):
    """
    Median measured and simulated values over each pulse from the per-run profile files
    """
    rows = []
    for _, run in runs.iterrows():
        try:
            with xr.open_dataset(run['runProfilesPath']) as profiles:
                row = {'shotId': run['shotId'], 'shotHeating': run['shotHeating']}
                for name in ('teCore', 'teVolAvg', 'neVolAvg', 'we'):
                    row[name + 'Meas'] = float(np.nanmedian(profiles[name + 'Meas'].values))
                    row[name + 'Sim'] = float(np.nanmedian(profiles[name + 'Sim'].values))
                rows.append(row)
        except (FileNotFoundError, KeyError, ValueError):
            print(f'no usable profiles file for shot {run["shotId"]}')
    return pd.DataFrame(rows)

def plotParity(runs, outDir):
    """
    Simulated against measured with the 1:1 line and a factor of 2 band, one point per shot
    """
    means = pulseMeans(runs)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    quantities = [('teCore', 'core Te [keV]', 1.0), ('teVolAvg', 'vol avg Te [keV]', 1.0), ('neVolAvg', 'vol avg ne [1e19 m^-3]', 1e19), ('we', 'electron stored energy [kJ]', 1e3)]
    for ax, (name, label, scale) in zip(axes, quantities):
        measured, simulated = means[name + 'Meas'] / scale, means[name + 'Sim'] / scale
        ax.scatter(measured, simulated, c=means['shotHeating'].map(heatingColors), s=25)
        # How strongly the simulation follows the measurement from shot to shot, fit in log space where a slope of 1
        # follows it and a slope of 0 gives the same value whatever was measured
        good = (measured > 0) & (simulated > 0)
        lines = []
        for groupName, members in [('all', good)] + [(heatingLabels[heating], good & (means['shotHeating'] == heating)) for heating in heatingColors]:
            if members.sum() >= 3:
                slope = np.polyfit(np.log(measured[members]), np.log(simulated[members]), 1)[0]
                correlation = np.corrcoef(np.log(measured[members]), np.log(simulated[members]))[0, 1]
                lines.append(f'{groupName}: slope {slope:.2f}, correlation {correlation:.2f}, n {int(members.sum())}')
        ax.text(0.97, 0.03, '\n'.join(lines), transform=ax.transAxes, ha='right', va='bottom', fontsize=7, bbox=dict(facecolor='white', edgecolor='0.7'))
        for shotId, x, y in zip(means['shotId'], measured, simulated):
            ax.annotate(str(shotId), (x, y), fontsize=6, alpha=0.6)
        top = 1.1 * np.nanmax([measured.max(), simulated.max()])
        line = np.linspace(0, top, 50)
        ax.plot(line, line, 'k-', lw=0.8)
        ax.fill_between(line, line / 2, line * 2, color='gray', alpha=0.15)
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_xlabel(f'measured {label}')
        ax.set_ylabel(f'simulated {label}')
    heatingLegend(axes[0])
    fig.suptitle('Sim vs measured, median over each pulse, shaded band is within a factor of 2')
    fig.tight_layout()
    saveFigure(fig, outDir, 'parity', 'accuracy')

def plotRatioInTime(runs, outDir):
    """
    Sim over measured through each pulse on a normalized time axis, to show where in the pulse the misses happen
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    quantities = [('teCore', 'core Te'), ('neVolAvg', 'vol avg ne'), ('we', 'electron stored energy')]
    bins = np.linspace(0, 1, 21)
    for ax, (name, label) in zip(axes, quantities):
        allTimes, allRatios = [], []
        for _, run in runs.iterrows():
            try:
                with xr.open_dataset(run['runProfilesPath']) as profiles:
                    t = profiles['thomson_time'].values
                    ratio = profiles[name + 'Sim'].values / np.where(profiles[name + 'Meas'].values > 0, profiles[name + 'Meas'].values, np.nan)
            except (FileNotFoundError, KeyError, ValueError):
                continue
            tNorm = (t - run['shotWindowStart']) / (run['shotWindowEnd'] - run['shotWindowStart'])
            ax.plot(tNorm, ratio, color=heatingColors.get(run['shotHeating'], 'gray'), lw=0.6, alpha=0.5)
            allTimes.append(tNorm)
            allRatios.append(ratio)
        allTimes, allRatios = np.concatenate(allTimes), np.concatenate(allRatios)
        # Median across shots in 5% wide time bins
        which = np.digitize(allTimes, bins)
        medians = [np.nanmedian(allRatios[which == i]) if np.any(which == i) else np.nan for i in range(1, len(bins))]
        ax.plot(0.5 * (bins[1:] + bins[:-1]), medians, 'k-', lw=2)
        ax.axhline(1.0, color='k', lw=0.8, ls='--')
        ax.set_yscale('log')
        ax.set_xlabel('time through the simulated window (0 = start, 1 = end)')
        ax.set_ylabel(f'{label} sim / measured')
        handles = [plt.Line2D([], [], color=color, lw=1, label=f'{heatingLabels[heating]} run') for heating, color in heatingColors.items()]
        handles += [plt.Line2D([], [], color='k', lw=2, label='median over runs'), plt.Line2D([], [], color='k', lw=0.8, ls='--', label='sim = measured')]
        ax.legend(handles=handles, fontsize=8)
    fig.suptitle('Simulated / measured against normalized time. Thin lines: runs. Thick line: median in 5% time bins')
    fig.tight_layout()
    saveFigure(fig, outDir, 'ratio_in_time', 'accuracy')

def plotHealth(runs, outDir):
    """
    Numerical health of every run, coupling gap, energy balance, failed TokaMaker solves, last surface and run time
    """
    fig, axes = plt.subplots(1, 5, figsize=(24, 4.6))
    # Column, title, y label, scale applied to the stored value, y range (None lets matplotlib pick)
    panels = [('healthFluxGap', 'TokaMaker vs TORAX consumed flux gap', 'largest gap / largest consumed flux [%]', 100.0, None),
              ('healthPowerResidual', 'TORAX energy balance residual', 'median |power in - power out| / power in [%]', 100.0, None),
              ('healthTmFailed', 'failed TokaMaker solves', 'solves that did not converge [count]', 1.0, None),
              ('healthLastSurface', 'TORAX boundary (last surface factor)', 'boundary flux surface [psiN]', 1.0, (0.8, 1.0)),
              ('runSeconds', 'run time', 'wall time [minutes]', 1.0 / 60.0, None)]
    order = runs.sort_values('shotId')
    for ax, (column, title, ylabel, scale, limits) in zip(axes, panels):
        ax.bar(range(len(order)), order[column] * scale, color=order['shotHeating'].map(heatingColors))
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order['shotId'], rotation=90, fontsize=7)
        ax.set_xlabel('shot')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if limits:
            ax.set_ylim(*limits)
    heatingLegend(axes[-1])
    fig.suptitle('Numerical health of each run')
    fig.tight_layout()
    saveFigure(fig, outDir, 'run_health', 'checks')

def plotDatabaseCoverage(runs, outDir):
    """
    Peak plasma current of every usable pulse for each heating type as a gray violin, with the shots run in the plotted
    batch as markers on top, each row labeled with how many pulses are usable and how many were run
    """
    usable = pd.read_csv(usablePath)
    usableCurrent = dict(zip(usable['shot_id'], usable['plasma_max_current'].abs()))
    ran = runs.drop_duplicates('shotId')
    markers = {'Ohmic': 'o', 'SS Beam': 's', 'SW Beam': '^', '2 Beams,SS Beam,SW Beam': 'D'}
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rowLabels = []
    for row, heat in enumerate(heatingColors):
        position = len(heatingColors) - 1 - row
        current = usable.loc[usable['heatingResolved'] == heat, 'plasma_max_current'].abs().dropna().values
        # Run shots placed at their catalog current, the same value the violin is built from
        ranCurrent = ran.loc[ran['shotHeating'] == heat, 'shotId'].map(usableCurrent).dropna().values
        if len(current) > 1:
            body = ax.violinplot(current, positions=[position], orientation='horizontal', widths=0.8, showextrema=False)['bodies'][0]
            body.set_facecolor('0.85')
            body.set_edgecolor('none')
            body.set_alpha(1.0)
        # Small vertical spread keeps shots at a similar current apart
        ax.scatter(ranCurrent, position + rng.uniform(-0.08, 0.08, len(ranCurrent)), marker=markers[heat], s=50, color=heatingColors[heat], edgecolor='k', linewidth=0.5, zorder=3)
        rowLabels.append(f'{heatingLabels[heat]}\n{len(current)} usable, {len(ranCurrent)} run')
    ax.set_yticks(range(len(heatingColors) - 1, -1, -1))
    ax.set_yticklabels(rowLabels)
    ax.set_xlabel('peak plasma current [kA]')
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='y', length=0)
    batches = runs['batchName'].unique()
    ax.set_title(f'Usable pulses (gray) and the shots run in {batches[0] if len(batches) == 1 else "every batch"} (markers)', fontsize=10)
    fig.tight_layout()
    saveFigure(fig, outDir, 'database_coverage', 'checks')

def collectRunData(runs):
    """
    Everything the pooled plots need from each run's profiles file, read once with bursts averaged
    """
    names = [name + suffix for name in ('teVolAvg', 'neVolAvg', 'we') for suffix in ('Meas', 'Sim')]
    runData = []
    for _, run in runs.iterrows():
        try:
            profiles = loadProfiles(run)
            times, teMeas, teSim, neMeas, neSim = pairedProfiles(profiles)
            _, *values = burstAverage(profiles['thomson_time'].values, *[profiles[name].values for name in names])
        except (OSError, KeyError, ValueError) as e:
            print(f'no usable profiles file for shot {run["shotId"]}: {e}')
            continue
        rho = profiles['rho_norm'].values
        edgeIndex = int(np.argmin(np.abs(rho - 0.98)))
        series = dict(zip(names, values))
        # Core and edge read straight off the measured profile, NaN where Thomson does not reach rho 0 or rho 0.98,
        # the stored core and edge columns fall back to the nearest point Thomson covers
        series.update({'teCoreMeas': teMeas[:, 0], 'teCoreSim': teSim[:, 0], 'teEdgeMeas': teMeas[:, edgeIndex], 'teEdgeSim': teSim[:, edgeIndex],
                       'neCoreMeas': neMeas[:, 0], 'neCoreSim': neSim[:, 0], 'neEdgeMeas': neMeas[:, edgeIndex], 'neEdgeSim': neSim[:, edgeIndex]})
        runData.append({'run': run, 'rho': rho, 'times': times,
                        'tNorm': (times - run['shotWindowStart']) / (run['shotWindowEnd'] - run['shotWindowStart']),
                        'teMeas': teMeas, 'teSim': teSim, 'neMeas': neMeas, 'neSim': neSim, 'series': series,
                        'txTimes': profiles['time'].values, 'teBoundary': profiles['T_e'].values[:, -1], 'edgePsiN': edgePsiN(profiles, run)})
    return runData

def edgePsiN(profiles, run, rhoEdge=0.98):
    """
    True normalized poloidal flux of the rho 0.98 edge point, median over the pulse. TORAX normalizes psi to its own
    boundary, which sits at the last surface factor of the real flux, so TORAX psi_norm times that factor is the real psiN
    """
    rhoFace = profiles['rho_face_norm'].values
    return float(np.median([np.interp(rhoEdge, rhoFace, psiRow) for psiRow in profiles['psi_norm'].values])) * float(run['healthLastSurface'])

def edgeLabel(runData):
    """
    Where the edge values sit, rho 0.98 of the TORAX grid and the span of real psiN that is across the runs
    """
    psiN = [item['edgePsiN'] for item in runData]
    return f'rho 0.98, psiN {min(psiN):.2f} to {max(psiN):.2f}'

def drawBand(ax, rho, stack, color, label):
    """
    Median line with a shaded 10 to 90 percent band across a stack of profiles
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        low, middle, high = np.nanpercentile(stack, [10, 50, 90], axis=0) if len(stack) else (np.full(len(rho), np.nan),) * 3
    ax.fill_between(rho, low, high, color=color, alpha=0.18, lw=0)
    ax.plot(rho, middle, color=color, lw=1.6, label=label)

def plotRatioVsRadius(runData, outDir):
    """
    Sim over measured Te and ne along the radius, one thin line per run (median over its Thomson times) and a bold
    median over runs for each heating type, to show where along the radius the simulation misses
    """
    rho = runData[0]['rho']
    heating = np.array([item['run']['shotHeating'] for item in runData])
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, (simName, measName, label) in zip(axes, (('teSim', 'teMeas', 'Te'), ('neSim', 'neMeas', 'ne'))):
        curves = np.array([onGrid(rho, item['rho'], coveredMedian(safeRatio(item[simName], item[measName]), axis=0)) for item in runData])
        for curve, heat in zip(curves, heating):
            ax.plot(rho, curve, color=heatingColors.get(heat, 'gray'), lw=0.6, alpha=0.3)
        for heat, color in heatingColors.items():
            group = curves[heating == heat]
            # Median only where at least 3 runs of this heating type have a value, no legend entry when it never does
            enough = np.sum(np.isfinite(group), axis=0) >= 3
            if enough.any():
                ax.plot(rho[enough], quietMedian(group, axis=0)[enough], color=color, lw=2.4, label=f'{heatingLabels[heat]} median (n={len(group)})')
        ax.axvline(0.1, color='0.45', ls=':', lw=1.2, label='inner patch edge, rho 0.1')
        ax.axvline(0.98, color='0.45', ls='--', lw=1.2, label=f'edge value taken here, {edgeLabel(runData)}')
        ratioAxis(ax)
        ax.set_xlim(0, 1)
        ax.set_xlabel('rho, normalized toroidal flux radius (1 is the TORAX boundary)')
        ax.set_ylabel(f'{label} sim / measured')
        ax.set_title(f'{label} along the radius')
    axes[0].legend(fontsize=8)
    fig.suptitle('Simulated / measured against radius. Thin lines: runs (median over each pulse). Bold lines: median over runs of each heating type,\n'
                 'where at least 3 runs are measured')
    fig.tight_layout()
    saveFigure(fig, outDir, 'ratio_vs_radius', 'accuracy')

def plotRatioMap(runData, outDir, timeBins=10):
    """
    Median sim over measured Te and ne on a grid of time through the pulse against radius, each run counts once per
    cell and cells with fewer than 3 runs are left blank
    """
    rho = runData[0]['rho']
    timeEdges = np.linspace(0, 1, timeBins + 1)
    rhoEdges = np.concatenate([[rho[0]], 0.5 * (rho[1:] + rho[:-1]), [rho[-1]]])
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for ax, (simName, measName, label) in zip(axes, (('teSim', 'teMeas', 'Te'), ('neSim', 'neMeas', 'ne'))):
        perRun = np.full((len(runData), timeBins, len(rho)), np.nan)
        for k, item in enumerate(runData):
            ratio = safeRatio(item[simName], item[measName])
            which = np.clip(np.digitize(item['tNorm'], timeEdges) - 1, 0, timeBins - 1)
            for b in np.unique(which):
                perRun[k, b] = onGrid(rho, item['rho'], quietMedian(ratio[which == b], axis=0))
        enough = np.sum(np.isfinite(perRun), axis=0) >= 3
        median = np.ma.masked_invalid(np.where(enough, np.log2(quietMedian(perRun, axis=0)), np.nan))
        mesh = ax.pcolormesh(timeEdges, rhoEdges, median.T, cmap='RdBu_r', vmin=-1, vmax=1, shading='flat')
        colorbar = fig.colorbar(mesh, ax=ax)
        colorbar.set_ticks(np.log2([0.5, 0.71, 1, 1.41, 2]))
        colorbar.set_ticklabels(['0.5', '0.71', '1', '1.41', '2'])
        colorbar.set_label(f'{label} sim / measured, median over runs')
        ax.set_xlabel('time through the simulated window (0 = start, 1 = end)')
        ax.set_ylabel('rho, normalized toroidal flux radius (1 is the TORAX boundary)')
        ax.set_title(f'{label}')
    fig.suptitle('Median simulated / measured against time and radius, color scale saturates at 0.5 and 2,\n'
                 'blank cells have fewer than 3 measured runs')
    fig.tight_layout()
    saveFigure(fig, outDir, 'ratio_map', 'accuracy')

def plotPeaking(runData, outDir):
    """
    Profile peaking, core over volume average, simulated against measured with one point per run (median over the pulse),
    catches a profile whose average is right but whose shape is wrong
    """
    colors = [heatingColors.get(item['run']['shotHeating'], 'gray') for item in runData]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, (name, label) in zip(axes, (('te', 'Te'), ('ne', 'ne'))):
        measured = np.array([coveredMedian(safeRatio(item['series'][name + 'CoreMeas'], item['series'][name + 'VolAvgMeas'])) for item in runData])
        simulated = np.array([coveredMedian(safeRatio(item['series'][name + 'CoreSim'], item['series'][name + 'VolAvgSim'])) for item in runData])
        # Simulated peaking only where the measured one exists, so both axes use the same runs
        simulated = np.where(np.isfinite(measured), simulated, np.nan)
        ax.scatter(measured, simulated, c=colors, s=30)
        for item, x, y in zip(runData, measured, simulated):
            ax.annotate(str(item['run']['shotId']), (x, y), fontsize=6, alpha=0.6)
        low, high = 0.9 * np.nanmin([measured, simulated]), 1.1 * np.nanmax([measured, simulated])
        ax.plot([low, high], [low, high], 'k-', lw=0.8)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xlabel(f'measured {label} peaking, core / volume average (no units)')
        ax.set_ylabel(f'simulated {label} peaking, core / volume average (no units)')
        ax.set_title(f'{label} (n={np.sum(np.isfinite(measured) & np.isfinite(simulated))} runs with the core measured at over half their times)', fontsize=10)
    heatingLegend(axes[0])
    fig.suptitle('Peaking factor (core / volume average), median over each pulse')
    fig.tight_layout()
    saveFigure(fig, outDir, 'peaking', 'shape')

def plotEdgeCheck(runData, outDir):
    """
    Edge sanity check, left is simulated Te at the TORAX boundary over the input edge Te (exactly 1 when the boundary
    condition is honored), middle and right are simulated against measured Te and ne at rho 0.98 at every Thomson time
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.8))
    worst = 0.0
    for item in runData:
        run = item['run']
        txNorm = (item['txTimes'] - run['shotWindowStart']) / (run['shotWindowEnd'] - run['shotWindowStart'])
        boundaryRatio = item['teBoundary'] / run['cfgEdgeTe']
        worst = max(worst, float(np.nanmax(np.abs(boundaryRatio - 1))))
        axes[0].plot(txNorm, boundaryRatio, color=heatingColors.get(run['shotHeating'], 'gray'), lw=0.8, alpha=0.6)
    inputs = ', '.join(f'{1e3 * value:g} eV' for value in sorted({item['run']['cfgEdgeTe'] for item in runData}))
    axes[0].axhline(1.0, color='k', lw=0.8, ls='--')
    axes[0].set_ylim(0.9, 1.1)
    axes[0].set_xlabel('time through the simulated window (0 = start, 1 = end)')
    axes[0].set_ylabel('simulated Te at the TORAX boundary (rho 1) / input edge Te (no units)')
    axes[0].set_title(f'Simulated boundary Te / input edge Te ({inputs}),\nlargest deviation {worst:.1e}, {len(runData)} runs', fontsize=10)
    heatingLegend(axes[0])
    location = edgeLabel(runData)
    for ax, (name, label, scale) in zip(axes[1:], (('teEdge', f'edge Te at {location} [keV]', 1.0), ('neEdge', f'edge ne at {location} [1e19 m^-3]', 1e19))):
        measured = np.concatenate([item['series'][name + 'Meas'] for item in runData]) / scale
        simulated = np.concatenate([item['series'][name + 'Sim'] for item in runData]) / scale
        colors = np.concatenate([[heatingColors.get(item['run']['shotHeating'], 'gray')] * len(item['series'][name + 'Meas']) for item in runData])
        keep = np.isfinite(measured) & np.isfinite(simulated)
        ax.scatter(measured[keep], simulated[keep], c=colors[keep], s=7, alpha=0.5)
        top = 1.05 * np.nanpercentile(np.concatenate([measured[keep], simulated[keep]]), 99)
        line = np.linspace(0, top, 50)
        ax.plot(line, line, 'k-', lw=0.8)
        ax.fill_between(line, line / 2, line * 2, color='gray', alpha=0.12)
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ratio = safeRatio(simulated[keep], measured[keep])
        ax.set_title(f'Each point is one Thomson measurement time, {keep.sum()} of them across {len(runData)} runs (bursts averaged,\nonly times where Thomson measures this point), {np.nanmean((ratio > 0.5) & (ratio < 2)):.0%} within 2x, median ratio {quietMedian(ratio):.2f}', fontsize=9)
        ax.set_xlabel(f'measured {label}')
        ax.set_ylabel(f'simulated {label}')
    fig.suptitle('Edge check. Left: simulated boundary Te / input edge Te. Middle and right: simulated against measured edge Te and ne,\n'
                 '1:1 line and factor of 2 band')
    fig.tight_layout()
    saveFigure(fig, outDir, 'edge_check', 'checks')

def plotParityAllTimes(runData, outDir):
    """
    Simulated against measured at every Thomson time of every run (bursts averaged) as a density of points on log axes,
    with the 1:1 line and dashed lines a factor of 2 either side
    """
    quantities = [('teCore', 'core Te [keV]', 1.0), ('teVolAvg', 'vol avg Te [keV]', 1.0), ('neCore', 'core ne [1e19 m^-3]', 1e19),
                  ('neVolAvg', 'vol avg ne [1e19 m^-3]', 1e19), ('we', 'electron stored energy W_e [kJ]', 1e3)]
    fig, axes = plt.subplots(1, 5, figsize=(29, 5.6))
    for ax, (name, label, scale) in zip(axes, quantities):
        measured = np.concatenate([item['series'][name + 'Meas'] for item in runData]) / scale
        simulated = np.concatenate([item['series'][name + 'Sim'] for item in runData]) / scale
        keep = np.isfinite(measured) & np.isfinite(simulated) & (measured > 0) & (simulated > 0)
        measured, simulated = measured[keep], simulated[keep]
        low, high = 0.8 * min(measured.min(), simulated.min()), 1.25 * max(measured.max(), simulated.max())
        extent = (np.log10(low), np.log10(high), np.log10(low), np.log10(high))
        cells = ax.hexbin(measured, simulated, gridsize=30, xscale='log', yscale='log', extent=extent, mincnt=1, cmap='Blues', norm=LogNorm())
        line = np.array([low, high])
        ax.plot(line, line, 'k-', lw=0.9)
        ax.plot(line, 2 * line, 'k--', lw=0.7)
        ax.plot(line, line / 2, 'k--', lw=0.7)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ratio = simulated / measured
        ax.set_title(f'{label}\n{np.mean((ratio > 0.5) & (ratio < 2)):.0%} of {len(ratio)} Thomson measurement times within 2x, median ratio {np.median(ratio):.2f}', fontsize=10)
        ax.set_xlabel(f'measured {label}')
        ax.set_ylabel(f'simulated {label}')
        fig.colorbar(cells, ax=ax, label='Thomson times per cell')
    fig.suptitle(f'Simulated against measured, each point is one Thomson measurement time of one of {len(runData)} runs (bursts averaged), color is how many points fall in each hexagon,\n'
                 'solid line is a match and dashed lines a factor of 2, core values only where Thomson reaches the center')
    fig.tight_layout()
    saveFigure(fig, outDir, 'parity_all_times', 'accuracy')

def plotRatioHistograms(runData, outDir):
    """
    How often each quantity lands within 1.25x and 2x of the measurement, every Thomson time of every run with each run
    weighted to 1 so long pulses do not dominate, Ohmic and beam heated runs separately
    """
    location = edgeLabel(runData)
    quantities = [('teCore', 'core Te'), ('teVolAvg', 'vol avg Te'), ('teEdge', f'edge Te ({location})'), ('we', 'electron stored energy W_e'),
                  ('neCore', 'core ne'), ('neVolAvg', 'vol avg ne'), ('neEdge', f'edge ne ({location})')]
    # Solid blue against dashed black so the two groups differ in color and line style
    groups = [('Ohmic', 'tab:blue', '-', True), ('beam heated', 'k', '--', False)]
    bins = np.linspace(-2, 2, 33)
    fig, axes = plt.subplots(2, 4, figsize=(21, 8.5))
    for ax, (name, label) in zip(axes.ravel(), quantities):
        for groupLabel, color, style, ohmic in groups:
            logRatios, weights = [], []
            for item in runData:
                if (item['run']['shotHeating'] == 'Ohmic') != ohmic:
                    continue
                ratio = safeRatio(item['series'][name + 'Sim'], item['series'][name + 'Meas'])
                ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
                logRatios.append(np.log2(ratio))
                weights.append(np.full(len(ratio), 1.0 / max(len(ratio), 1)))
            # A group can have runs yet no measured values, an off axis shot has no measured core
            if logRatios and np.concatenate(weights).sum() > 0:
                logRatios, weights = np.concatenate(logRatios), np.concatenate(weights)
                within125 = weights[np.abs(logRatios) < np.log2(1.25)].sum() / weights.sum()
                within2 = weights[np.abs(logRatios) < 1].sum() / weights.sum()
                # Values beyond 4x either way pile into the end bins
                ax.hist(np.clip(logRatios, -2, 2), bins=bins, weights=weights, histtype='step', lw=1.6, color=color, linestyle=style,
                        label=f'{groupLabel}: {within125:.0%} within 1.25x, {within2:.0%} within 2x')
        ax.axvline(0, color='0.5', lw=0.8)
        ax.set_xticks(np.log2([0.25, 0.5, 1, 2, 4]))
        ax.set_xticklabels(['0.25', '0.5', '1', '2', '4'])
        ax.set_xlabel(f'{label} sim / measured', fontsize=9)
        ax.set_ylabel('runs (each run weighted to 1)')
        ax.legend(fontsize=7)
    axes.ravel()[-1].axis('off')
    fig.suptitle('Spread of sim over measured at every Thomson time, each run weighted equally, values beyond 4x either way are piled into the end bins')
    fig.tight_layout()
    saveFigure(fig, outDir, 'ratio_histograms', 'accuracy')

def ratioAxes(ax, low=0.2, high=5.0):
    """
    Log x and y axes for sim over measured on both, plain number ticks and lines through 1 on each axis
    """
    ticks = [0.25, 0.5, 1, 2, 4]
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{tick:g}' for tick in ticks])
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{tick:g}' for tick in ticks])
    ax.minorticks_off()
    ax.axhline(1.0, color='0.5', lw=0.5)
    ax.axvline(1.0, color='0.5', lw=0.5)

def plotCompensatingErrors(runs, outDir):
    """
    Whether the Te and ne misses cancel in the electron stored energy, which is proportional to ne times Te. Left is each
    run's Te miss against its ne miss with lines where the product is constant, right checks that the product tracks the
    stored energy miss
    """
    colors = runs['shotHeating'].map(heatingColors).fillna('gray')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.scatter(runs['resNeVolAvgRatio'], runs['resTeVolAvgRatio'], c=colors, s=30)
    line = np.geomspace(0.2, 5, 50)
    for product, style in ((0.5, ':'), (1.0, '--'), (2.0, '-.')):
        ax.plot(line, product / line, color='0.4', ls=style, lw=1, label=f'Te ratio x ne ratio = {product:g}')
    ax.set_xlabel('vol avg ne sim / measured')
    ax.set_ylabel('vol avg Te sim / measured')
    ax.set_title('Volume averaged Te ratio against ne ratio', fontsize=10)
    ratioAxes(ax)
    ax.legend(fontsize=7, loc='lower left')

    ax = axes[1]
    ax.scatter(runs['resTeVolAvgRatio'] * runs['resNeVolAvgRatio'], runs['resWeRatio'], c=colors, s=30)
    ax.plot([0.2, 5], [0.2, 5], 'k-', lw=0.8)
    ax.set_xlabel('Te ratio x ne ratio (no units)')
    ax.set_ylabel('electron stored energy W_e sim / measured')
    ax.set_title('Te ratio x ne ratio against stored energy ratio, 1:1 line', fontsize=10)
    ratioAxes(ax)
    heatingLegend(ax)
    fig.suptitle('Volume averaged Te, ne and electron stored energy ratios, one point per run (median over each pulse)')
    fig.tight_layout()
    saveFigure(fig, outDir, 'compensating_errors', 'shape')

def plotDensityVsGreenwald(runs, outDir):
    """
    Volume averaged density against the Greenwald density from the plasma current and the EFIT minor radius, measured
    and simulated, one point per run
    """
    rows = []
    for _, run in runs.iterrows():
        try:
            profiles = loadProfiles(run)
            # Minor radius from the EFIT boundary at each TokaMaker time, pad boundary with NaN
            lcfsR = profiles['efitLcfsR'].values
            minorRadius = float(np.nanmedian((np.nanmax(lcfsR, axis=1) - np.nanmin(lcfsR, axis=1)) / 2))
            greenwald = float(np.nanmedian(np.abs(profiles['Ip'].values))) / 1e6 / (np.pi * minorRadius ** 2) * 1e20
            rows.append({'shotHeating': run['shotHeating'], 'greenwald': greenwald, 'neMeas': float(np.nanmedian(profiles['neVolAvgMeas'].values)),
                         'neSim': float(np.nanmedian(profiles['neVolAvgSim'].values))})
        except (OSError, KeyError, ValueError) as e:
            print(f'no Greenwald density for run {run["runId"]}: {e}')
    table = pd.DataFrame(rows)
    if table.empty:
        print('no runs with the EFIT boundary saved, skipping the density against Greenwald plot')
        return
    colors = table['shotHeating'].map(heatingColors).fillna('gray')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    labels = []
    for column, marker, fill, name in (('neMeas', 'o', True, 'measured'), ('neSim', 's', False, 'simulated')):
        ax.scatter(table['greenwald'] / 1e19, table[column] / 1e19, marker=marker, s=40, c=colors if fill else 'none', edgecolors=colors, linewidths=1.2)
        correlation = np.corrcoef(np.log(table['greenwald']), np.log(table[column]))[0, 1] if len(table) >= 3 else np.nan
        labels.append(f'{name} ({"filled circles" if fill else "open squares"}): correlation {correlation:.2f} with the Greenwald density')
    top = 1.1 * np.nanmax(table[['greenwald', 'neMeas', 'neSim']].values) / 1e19
    ax.plot([0, top], [0, top], 'k:', lw=0.8, label='ne = n_GW')
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel('Greenwald density from Ip and the EFIT minor radius [1e19 m^-3]')
    ax.set_ylabel('volume averaged ne [1e19 m^-3]')
    ax.set_title('\n'.join(labels), fontsize=9)
    heatingLegend(ax)

    ax = axes[1]
    ax.scatter(table['neMeas'] / table['greenwald'], table['neSim'] / table['greenwald'], c=colors, s=40)
    topFraction = 1.1 * np.nanmax([(table['neMeas'] / table['greenwald']).max(), (table['neSim'] / table['greenwald']).max()])
    ax.plot([0, topFraction], [0, topFraction], 'k-', lw=0.8)
    ax.set_xlim(0, topFraction)
    ax.set_ylim(0, topFraction)
    ax.set_xlabel('measured Greenwald fraction (no units)')
    ax.set_ylabel('simulated Greenwald fraction (no units)')
    ax.set_title('Greenwald fraction, simulated against measured, 1:1 line', fontsize=10)
    fig.suptitle('Volume averaged ne against Greenwald density, one point per run (median over each pulse)')
    fig.tight_layout()
    saveFigure(fig, outDir, 'density_vs_greenwald', 'shape')

def plotCoreVsAverage(runData, outDir):
    """
    Each run's core miss against its volume average miss for Te and ne, with the 1:1 line and a shaded corner for a high
    average with a low core
    """
    colors = [heatingColors.get(item['run']['shotHeating'], 'gray') for item in runData]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (name, label) in zip(axes, (('te', 'Te'), ('ne', 'ne'))):
        average = np.array([coveredMedian(safeRatio(item['series'][name + 'VolAvgSim'], item['series'][name + 'VolAvgMeas'])) for item in runData])
        core = np.array([coveredMedian(safeRatio(item['series'][name + 'CoreSim'], item['series'][name + 'CoreMeas'])) for item in runData])
        ax.scatter(average, core, c=colors, s=30)
        ax.plot([0.2, 5], [0.2, 5], 'k-', lw=0.8)
        ax.fill_between([1, 5], 0.2, 1, color='gray', alpha=0.12)
        for item, x, y in zip(runData, average, core):
            if x > 1 and y < 1:
                ax.annotate(str(item['run']['shotId']), (x, y), fontsize=6, alpha=0.7)
        ax.set_xlabel(f'vol avg {label} sim / measured')
        ax.set_ylabel(f'core {label} sim / measured')
        ax.set_title(f'{label}, {np.sum(np.isfinite(core))} runs with the core measured at over half their times,\nshaded: volume average ratio > 1 and core ratio < 1', fontsize=10)
        ratioAxes(ax)
    heatingLegend(axes[0])
    fig.suptitle('Core ratio against volume average ratio, one point per run, 1:1 line')
    fig.tight_layout()
    saveFigure(fig, outDir, 'core_vs_average', 'shape')

def plotTimeTraceGrid(runData, outDir, perPage=42):
    """
    Small multiples of one quantity against time, one panel per run sorted by shot, simulated line over measured points
    with bursts averaged, saved in pages of perPage runs
    """
    location = edgeLabel(runData)
    # Key in the series, axis label, scale, file name
    quantities = [('neVolAvg', 'vol avg ne [1e19 m^-3]', 1e19, 'ne_volume_average'), ('neCore', 'core ne at rho 0 [1e19 m^-3]', 1e19, 'ne_core'),
                  ('neEdge', f'edge ne at {location} [1e19 m^-3]', 1e19, 'ne_edge'), ('teVolAvg', 'vol avg Te [keV]', 1.0, 'te_volume_average'),
                  ('teCore', 'core Te at rho 0 [keV]', 1.0, 'te_core'), ('teEdge', f'edge Te at {location} [keV]', 1.0, 'te_edge'),
                  ('we', 'electron stored energy W_e [kJ]', 1e3, 'electron_stored_energy')]
    ordered = sorted(runData, key=lambda item: item['run']['shotId'])
    pages = [ordered[start:start + perPage] for start in range(0, len(ordered), perPage)]
    handles = [plt.Line2D([], [], color=color, lw=1.4, label=f'{heatingLabels[heating]} simulated') for heating, color in heatingColors.items()]
    handles.append(plt.Line2D([], [], color='k', marker='.', linestyle='', label='measured (Thomson)'))
    columns = 6
    for name, label, scale, fileName in quantities:
        for page, items in enumerate(pages, 1):
            rows = int(np.ceil(len(items) / columns))
            fig, axes = plt.subplots(rows, columns, figsize=(18, 2.3 * rows + 1.4), squeeze=False)
            for ax, item in zip(axes.ravel(), items):
                run = item['run']
                ax.plot(item['times'], item['series'][name + 'Meas'] / scale, 'k.', ms=3)
                ax.plot(item['times'], item['series'][name + 'Sim'] / scale, color=heatingColors.get(run['shotHeating'], 'gray'), lw=1.4)
                ax.set_title(str(run['shotId']), fontsize=8)
                ax.tick_params(labelsize=7)
                ax.set_ylim(bottom=0)
            for ax in axes.ravel()[len(items):]:
                ax.axis('off')
            pageNote = f', page {page} of {len(pages)}' if len(pages) > 1 else ''
            fig.suptitle(f'{label}, one panel per run (shot number on top){pageNote}')
            fig.supxlabel('time [s]')
            fig.supylabel(label)
            fig.legend(handles=handles, loc='upper right', fontsize=8, ncol=5)
            fig.tight_layout(rect=(0.01, 0.01, 1, 0.95))
            saveFigure(fig, outDir, f'traces_{fileName}' + (f'_{page}' if len(pages) > 1 else ''), 'traces')

def burstMeanSpread(times, values, maxGap=0.002):
    """
    One point per laser burst at the burst's mean time with the spread inside the burst (0 for a single laser),
    the same grouping the batch figures use for their error bars
    """
    times, values = np.asarray(times, dtype=float), np.asarray(values, dtype=float)
    burst = np.cumsum(np.concatenate([[False], np.diff(times) > maxGap]))[:len(times)]
    grouped = pd.DataFrame({'time': times, 'value': values}).groupby(burst)
    return grouped['time'].mean().values, grouped['value'].mean().values, grouped['value'].std().fillna(0.0).values

def plotPulseKinetics(profiles, run, outDir, folder=''):
    """
    Te and ne against time at the core, the volume average and the edge for one run, the batch's Te and ne figures rebuilt
    from the profiles file, measured bursts as one point with the spread as the error bar, simulated at every TORAX time
    """
    rho = profiles['rho_norm'].values
    edgeIndex = int(np.argmin(np.abs(rho - 0.98)))
    edgeTitle = f'edge (rho 0.98, psiN {edgePsiN(profiles, run):.2f})'
    thomsonTimes, txTimes = profiles['thomson_time'].values, profiles['time'].values
    for simName, volAvgName, measName, scale, unit, title, fileName in (('T_e', 'T_e_volume_avg', 'te', 1.0, 'Te [keV]', 'Electron temperature', 'Te'),
                                                                         ('n_e', 'n_e_volume_avg', 'ne', 1e19, 'ne [1e19 m^-3]', 'Electron density', 'ne')):
        panels = [('core (rho 0)', profiles[measName + 'MeasProfile'].values[:, 0], profiles[simName].values[:, 0]),
                  ('volume average', profiles[measName + 'VolAvgMeas'].values, profiles[volAvgName].values),
                  (edgeTitle, profiles[measName + 'MeasProfile'].values[:, edgeIndex], profiles[simName].values[:, edgeIndex])]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
        for ax, (panelTitle, measured, simulated) in zip(axes, panels):
            burstTimes, means, spread = burstMeanSpread(thomsonTimes, measured / scale)
            ax.errorbar(burstTimes, means, yerr=spread, fmt='k.', elinewidth=0.8, capsize=2, label='measured, Thomson (burst mean and spread)')
            ax.plot(txTimes, simulated / scale, 'r-', label='TORAX (simulated)')
            ax.set_title(panelTitle)
            ax.set_xlabel('time [s]')
            ax.set_ylabel(unit)
            ax.set_ylim(bottom=0)
            if not np.isfinite(measured).any():
                ax.text(0.5, 0.92, 'Thomson does not reach this point on this shot', transform=ax.transAxes, ha='center', fontsize=8, bbox=dict(facecolor='white', edgecolor='0.7'))
        axes[0].legend(fontsize=7)
        fig.suptitle(f'{title}, shot {run["shotId"]} ({heatingLabels.get(run["shotHeating"], run["shotHeating"])}, {run["batchName"]})')
        fig.tight_layout()
        saveFigure(fig, outDir, f'{run["shotId"]}_{fileName}', folder)

def plotPulseEnergyRadiation(profiles, run, outDir, folder=''):
    """
    Electron stored energy measured against simulated and the radiated power for one run, the batch's stored energy and
    radiation figure rebuilt from the profiles file, with the bolometer trace when the run saved it
    """
    txTimes = profiles['time'].values
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    burstTimes, means, spread = burstMeanSpread(profiles['thomson_time'].values, profiles['weMeas'].values / 1e3)
    axes[0].errorbar(burstTimes, means, yerr=spread, fmt='k.', elinewidth=0.8, capsize=2, label='measured, from Thomson ne and Te (burst mean and spread)')
    axes[0].plot(txTimes, profiles['W_thermal_e'].values / 1e3, 'r-', label='TORAX (simulated)')
    axes[0].set_title('electron stored energy, W_e = 3/2 times the volume integral of ne Te')
    axes[0].set_ylabel('W_e [kJ]')
    axes[0].set_ylim(bottom=0)
    # TORAX stores radiation as a loss, negative, so the sign is flipped to plot radiated power
    axes[1].plot(txTimes, -profiles['P_radiation_e'].values / 1e6, 'r-', label='TORAX impurity radiation (simulated)')
    # Older profiles files were saved before the bolometer trace was added to them
    if 'pradMeas' in profiles:
        axes[1].plot(profiles['bolometer_time'].values, profiles['pradMeas'].values / 1e6, 'k-', lw=0.6, alpha=0.7, label='bolometer (measured, includes divertor and SOL radiation)')
        axes[1].set_title('radiated power')
    else:
        axes[1].set_title('radiated power, this run has no bolometer trace saved')
    axes[1].set_ylabel('P_rad [MW]')
    for ax in axes:
        ax.set_xlabel('time [s]')
        ax.legend(fontsize=7)
    fig.suptitle(f'Stored energy and radiation, shot {run["shotId"]} ({heatingLabels.get(run["shotHeating"], run["shotHeating"])}, {run["batchName"]})')
    fig.tight_layout()
    saveFigure(fig, outDir, f'{run["shotId"]}_energy_radiation', folder)

def plotPulseProfileEvolution(profiles, run, outDir, folder='', nTimes=10):
    """
    Simulated profiles at nTimes evenly spaced times colored by time against rho, with measured Te and ne at the nearest
    Thomson time dashed
    """
    rho, rhoFace = profiles['rho_norm'].values, profiles['rho_face_norm'].values
    txTimes = profiles['time'].values
    picks = np.unique([int(np.argmin(np.abs(txTimes - t))) for t in np.linspace(txTimes[0], txTimes[-1], nTimes)])
    norm = plt.Normalize(vmin=txTimes[0], vmax=txTimes[-1])
    # Thermal pressure from electrons and main ions, ne Te in m^-3 keV times 1.602e-16 J per keV gives Pa
    pressure = (profiles['n_e'].values * profiles['T_e'].values + profiles['n_i'].values * profiles['T_i'].values) * 1.602e-16 / 1e3
    # Title, axis label, values, rho grid, measured profile to overlay, its scale
    panels = [('n_e', 'n_e [1e19 m^-3]', profiles['n_e'].values / 1e19, rho, 'neMeasProfile', 1e19),
              ('T_e', 'T_e [keV]', profiles['T_e'].values, rho, 'teMeasProfile', 1.0),
              ('pressure', 'p from ne Te + ni Ti [kPa]', pressure, rho, None, None),
              ('q', 'safety factor q (no units)', profiles['q'].values, rhoFace, None, None),
              ('n_i', 'n_i [1e19 m^-3]', profiles['n_i'].values / 1e19, rho, None, None),
              ('T_i', 'T_i [keV]', profiles['T_i'].values, rho, 'tiMeasProfile' if 'tiMeasProfile' in profiles else None, 1.0),
              ('psi', 'poloidal flux psi [Wb]', profiles['psi'].values, rho, None, None),
              ('j_total', 'total current density [MA m^-2]', profiles['j_total'].values / 1e6, rho, None, None)]
    fig, axes = plt.subplots(2, 4, figsize=(23, 9.5))
    for ax, (title, label, values, grid, measName, measScale) in zip(axes.ravel(), panels):
        for k in picks:
            color = plt.cm.plasma(norm(txTimes[k]))
            ax.plot(grid, values[k], color=color, lw=1.5)
            if measName:
                # Thomson times for Te and ne, CXRS times for Ti, whichever axis the measured profile is stored on
                measTimes = profiles[profiles[measName].dims[0]].values
                nearest = int(np.argmin(np.abs(measTimes - txTimes[k])))
                ax.plot(rho, profiles[measName].values[nearest] / measScale, color=color, ls='--', lw=1.2)
        ax.set_title(title)
        ax.set_xlabel('rho (1 is the TORAX boundary)')
        ax.set_ylabel(label)
        ax.set_xlim(0, 1)
    fig.legend(handles=[plt.Line2D([], [], color='k', label='simulated (TORAX)'), plt.Line2D([], [], color='k', ls='--', label='measured (Thomson for Te and ne, CXRS for Ti when saved, nearest time)')],
               loc='upper center', bbox_to_anchor=(0.45, 0.955), ncol=2, fontsize=11)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.plasma), ax=axes.ravel().tolist(), shrink=0.8, aspect=30, pad=0.02)
    colorbar.set_label('time [s]')
    fig.suptitle(f'Profile evolution, shot {run["shotId"]} ({heatingLabels.get(run["shotHeating"], run["shotHeating"])}, {run["batchName"]}), {len(picks)} times', fontsize=14)
    saveFigure(fig, outDir, f'{run["shotId"]}_profile_evolution', folder)

def plotPulseProfileSlices(profiles, run, outDir, folder='', nSlices=6):
    """
    Te and ne against rho for one run in nSlices time slices across the window, the line is the median and the band the
    10 to 90 percent spread of every profile in the slice, measured at each Thomson time and simulated at each TORAX time
    """
    rowSpecs = (('teMeasProfile', 'T_e', 1.0, 'Te [keV]'), ('neMeasProfile', 'n_e', 1e19, 'ne [1e19 m^-3]'))
    rho = profiles['rho_norm'].values
    thomsonTimes, txTimes = profiles['thomson_time'].values, profiles['time'].values
    edges = np.linspace(thomsonTimes.min(), thomsonTimes.max(), nSlices + 1)
    fig, axes = plt.subplots(2, nSlices, figsize=(3.1 * nSlices, 7), sharey='row', squeeze=False)
    for col in range(nSlices):
        inMeasured = (thomsonTimes >= edges[col]) & (thomsonTimes <= edges[col + 1])
        inSimulated = (txTimes >= edges[col]) & (txTimes <= edges[col + 1])
        for row, (measName, simName, scale, label) in enumerate(rowSpecs):
            ax = axes[row, col]
            drawBand(ax, rho, profiles[measName].values[inMeasured] / scale, 'k', f'measured, {inMeasured.sum()} Thomson measurement times in this slice')
            drawBand(ax, rho, profiles[simName].values[inSimulated] / scale, 'tab:red', f'simulated, {inSimulated.sum()} TORAX output times in this slice')
            ax.axvline(0.1, color='0.5', ls=':', lw=0.9)
            ax.axvline(0.98, color='0.5', ls='--', lw=0.9)
            ax.set_xlim(0, 1)
            ax.legend(fontsize=6, loc='lower left')
            if row == 0:
                ax.set_title(f'{edges[col]:.3f} to {edges[col + 1]:.3f} s', fontsize=9)
            if row == 1:
                ax.set_xlabel('rho (1 is the TORAX boundary)')
            if col == 0:
                ax.set_ylabel(label)
    # Each row shares one y axis, the floor is set once every slice is drawn so no slice gets clipped
    for row in range(len(rowSpecs)):
        axes[row, 0].set_ylim(bottom=0)
    fig.suptitle(f'Shot {run["shotId"]} ({heatingLabels.get(run["shotHeating"], run["shotHeating"])}, {run["batchName"]}), line is the median and band the 10 to 90 percent spread in each slice\n'
                 'dotted line is the inner patch edge, dashed line where the edge value is taken, rho 1 is the TORAX boundary set by the edge condition', fontsize=10)
    fig.tight_layout()
    saveFigure(fig, outDir, f'{run["shotId"]}_profile_slices', folder)

def plotPulseEquilibrium(profiles, run, outDir, folder=''):
    """
    li, q95 and consumed boundary flux from TokaMaker at each solve time against EFIT for one run, the batch's
    equilibrium figure rebuilt from the profiles file, a failed TokaMaker solve shows as a gap in the simulated line
    """
    times = profiles['tm_time'].values
    panels = [('liSim', 'liMeas', 'internal inductance li(1) (no units)', 'internal inductance'),
              ('q95Sim', 'q95Meas', 'edge safety factor q95 (no units)', 'q95'),
              ('fluxSim', 'fluxMeas', 'consumed boundary flux [Wb]', 'flux consumed since the window start')]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, (simName, measName, label, title) in zip(axes, panels):
        ax.plot(times, profiles[measName].values, 'k.-', label='EFIT (measured)')
        ax.plot(times, profiles[simName].values, 'r-o', ms=3, label='TokaMaker (simulated)')
        ax.set_title(title)
        ax.set_xlabel('time [s]')
        ax.set_ylabel(label)
    axes[0].legend(fontsize=7)
    failed = int(np.sum(profiles['tmSolved'].values == 0))
    fig.suptitle(f'Equilibrium, shot {run["shotId"]} ({heatingLabels.get(run["shotHeating"], run["shotHeating"])}, {run["batchName"]}), '
                 f'{len(times)} TokaMaker solve times, {failed} failed')
    fig.tight_layout()
    saveFigure(fig, outDir, f'{run["shotId"]}_equilibrium', folder)

def plotPulseBoundary(profiles, run, outDir, folder=''):
    """
    Plasma boundary shape at each TokaMaker solve time colored by time, TokaMaker solid and EFIT dashed inside the wall,
    the batch's boundary evolution figure rebuilt from the profiles file
    """
    times = profiles['tm_time'].values
    norm = plt.Normalize(vmin=times.min(), vmax=max(times.max(), times.min() + 1e-9))
    # Mirror mode stores Z first, swapped here so the first coordinate is always R
    flip = int(profiles['tmMirrorMode'].values) == 1
    wallR, wallZ = (profiles['wallZ'].values, profiles['wallR'].values) if flip else (profiles['wallR'].values, profiles['wallZ'].values)
    lcfsR, lcfsZ = (profiles['lcfsZ'].values, profiles['lcfsR'].values) if flip else (profiles['lcfsR'].values, profiles['lcfsZ'].values)
    fig, ax = plt.subplots(figsize=(6.5, 8.5))
    for k in np.unique(profiles['wallContour'].values):
        inContour = profiles['wallContour'].values == k
        ax.plot(wallR[inContour], wallZ[inContour], 'k-', lw=1.5)
    for i, t in enumerate(times):
        color = plt.cm.plasma(norm(t))
        atTime = profiles['lcfsTmIndex'].values == i
        ax.plot(lcfsR[atTime], lcfsZ[atTime], color=color, lw=1.4)
        ax.plot(profiles['efitLcfsR'].values[i], profiles['efitLcfsZ'].values[i], color=color, lw=1.0, ls='--')
    ax.set_aspect('equal')
    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')
    ax.legend(handles=[plt.Line2D([], [], color='0.3', label='TokaMaker (simulated)'), plt.Line2D([], [], color='0.3', ls='--', label='EFIT (measured)'),
                       plt.Line2D([], [], color='k', lw=1.5, label='wall')], fontsize=8, loc='upper right')
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.plasma), ax=ax, shrink=0.7, label='time [s]')
    # A TokaMaker surface that would not trace at psiN 1 was saved from just inside it
    tracedAt = profiles['lcfsTracedPsiN'].values if 'lcfsTracedPsiN' in profiles else np.ones(len(times))
    inside = tracedAt < 1
    insideNote = f'\n{int(inside.sum())} of {len(times)} TokaMaker boundaries traced just inside at psiN {", ".join(f"{p:.2f}" for p in np.unique(tracedAt[inside]))}, psiN 1 would not trace' if inside.any() else ''
    fig.suptitle(f'Plasma boundary, shot {run["shotId"]} ({heatingLabels.get(run["shotHeating"], run["shotHeating"])}, {run["batchName"]}){insideNote}', fontsize=10 if inside.any() else 12)
    saveFigure(fig, outDir, f'{run["shotId"]}_boundary', folder)

def plotPulseIonTemperature(profiles, run, outDir, folder=''):
    """
    Ti against time at the core, volume average, and edge for one run, CXRS against TORAX, core and edge read off the
    measured profile
    """
    cxrsTimes, txTimes = profiles['cxrs_time'].values, profiles['time'].values
    rho = profiles['rho_norm'].values
    edgeIndex = int(np.argmin(np.abs(rho - 0.98)))
    panels = [('core (rho 0)', profiles['tiMeasProfile'].values[:, 0], txTimes, profiles['T_i'].values[:, 0]),
              ('volume average', profiles['tiVolAvgMeas'].values, cxrsTimes, profiles['tiVolAvgSim'].values),
              (f'edge (rho 0.98, psiN {edgePsiN(profiles, run):.2f})', profiles['tiMeasProfile'].values[:, edgeIndex], txTimes, profiles['T_i'].values[:, edgeIndex])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, (title, measured, simTimes, simulated) in zip(axes, panels):
        burstTimes, means, spread = burstMeanSpread(cxrsTimes, measured)
        ax.errorbar(burstTimes, means, yerr=spread, fmt='k.', elinewidth=0.8, capsize=2, label='measured, CXRS frame nearest each Thomson burst (mean and spread)')
        ax.plot(simTimes, simulated, 'r-', label='TORAX (simulated)')
        ax.set_title(title)
        ax.set_xlabel('time [s]')
        ax.set_ylabel('Ti [keV]')
        ax.set_ylim(bottom=0)
        if not np.isfinite(measured).any():
            ax.text(0.5, 0.92, 'CXRS does not reach this point on this shot', transform=ax.transAxes, ha='center', fontsize=8, bbox=dict(facecolor='white', edgecolor='0.7'))
    axes[0].legend(fontsize=7)
    fig.suptitle(f'Ion temperature, shot {run["shotId"]} ({heatingLabels.get(run["shotHeating"], run["shotHeating"])}, {run["batchName"]})')
    fig.tight_layout()
    saveFigure(fig, outDir, f'{run["shotId"]}_Ti', folder)

def plotRunFigures(runs, outDir, profileLines=10, byShot=False):
    """
    Every per run figure, Te and ne against time, stored energy and radiation, profile evolution and profile slices, plus
    equilibrium, boundary and Ti for runs that saved those. In a batch folder each run goes in runs/<shot>, with byShot
    each shot gets its own folder under outDir with one subfolder per batch
    """
    for _, run in runs.iterrows():
        try:
            profiles = loadProfiles(run)
        except (OSError, KeyError, ValueError) as e:
            print(f'no usable profiles file for run {run["runId"]}: {e}')
            continue
        runOut, folder = (os.path.join(outDir, str(run['shotId'])), run['batchName']) if byShot else (outDir, os.path.join('runs', str(run['shotId'])))
        plotPulseKinetics(profiles, run, runOut, folder)
        plotPulseEnergyRadiation(profiles, run, runOut, folder)
        plotPulseProfileEvolution(profiles, run, runOut, folder, nTimes=profileLines)
        plotPulseProfileSlices(profiles, run, runOut, folder)
        # Older profiles files were saved before the equilibrium, boundary and CXRS data were added to them
        if 'tm_time' in profiles.dims:
            plotPulseEquilibrium(profiles, run, runOut, folder)
        if 'lcfs_point' in profiles.dims:
            plotPulseBoundary(profiles, run, runOut, folder)
        if 'cxrs_time' in profiles.dims:
            plotPulseIonTemperature(profiles, run, runOut, folder)

def plotEdgeEquivalence(runs, outDir, fitHalfWidth=0.08):
    """
    Temporary edge check for Oak, to be deleted once shown. For each run that saved its edge values, how one Thomson
    edge reading is made, the readings and the values given to TORAX against the simulated boundary over time, and the
    simulated over given ratio, plus one figure pooling every run. With the measured edge the simulation equals the
    measurement at the TORAX boundary by construction, this shows it
    """
    folder = os.path.join('checks', 'edge_equivalence')
    pooled = {'te': [], 'ne': [], 'teSim': [], 'neSim': [], 'runs': 0}
    for _, run in runs.iterrows():
        try:
            profiles = loadProfiles(run)
        except (OSError, KeyError, ValueError) as e:
            print(f'no usable profiles file for run {run["runId"]}: {e}')
            continue
        if 'edge_bc_time' not in profiles.dims:
            print(f'run {run["runId"]} was saved before runs kept their edge values, skipped in the edge check')
            continue
        txTimes, bcTimes = profiles['time'].values, profiles['edge_bc_time'].values
        boundary = float(run['healthLastSurface'])
        # TORAX interpolates the edge values it is given linearly in time, done the same way here at each TORAX output time
        teGiven = np.interp(txTimes, bcTimes, profiles['teEdgeBc'].values)
        neGiven = np.interp(txTimes, bcTimes, profiles['neEdgeBc'].values)
        teSim, neSim = profiles['T_e'].values[:, -1], profiles['n_e'].values[:, -1]
        measured = 'teEdgeFit' in profiles
        fig, axes = plt.subplots(1, 4, figsize=(27, 5.4))

        ax = axes[0]
        if measured:
            k = int(np.argmin(np.abs(bcTimes - np.median(bcTimes))))
            atTime = np.isclose(profiles['edgePointTime'].values, bcTimes[k])
            psiN, te = profiles['edgePointPsiN'].values[atTime], profiles['edgePointTe'].values[atTime]
            span = np.linspace(boundary - fitHalfWidth, boundary + fitHalfWidth, 20)
            ax.axvspan(boundary - fitHalfWidth, boundary + fitHalfWidth, color='0.92', label=f'fit window, psiN within {fitHalfWidth} of the boundary')
            ax.plot(psiN, te, 'ko', ms=5, label=f'Thomson points at {bcTimes[k]:.4f} s')
            ax.plot(span, np.polyval(np.polyfit(psiN, te, 1), span), 'b-', label='straight line fit through them')
            ax.plot(boundary, profiles['teEdgeFit'].values[k], 'b*', ms=15, label='reading, the line at the boundary')
            ax.plot(boundary, profiles['teEdgeBc'].values[k], 'gD', ms=8, label='after the running median, given to TORAX')
            ax.plot(boundary, np.interp(bcTimes[k], txTimes, teSim), 'ro', mfc='none', ms=13, mew=2, label='TORAX boundary Te at that time')
            ax.axvline(boundary, color='0.4', ls='--', lw=1)
            ax.set_xlabel('psiN')
            ax.set_ylabel('Te [keV]')
            ax.set_title(f'How one edge value is made, TORAX boundary at psiN {boundary:.2f}', fontsize=10)
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, 'this run used the constant edge values,\nso there is no Thomson reading to show', ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()

        for ax, (fitName, givenName, simulated, scale, label) in zip(axes[1:3], (('teEdgeFit', 'teEdgeBc', teSim, 1.0, 'Te at the TORAX boundary [keV]'),
                                                                                  ('neEdgeFit', 'neEdgeBc', neSim, 1e19, 'ne at the TORAX boundary [1e19 m^-3]'))):
            if measured:
                ax.plot(bcTimes, profiles[fitName].values / scale, '.', color='0.6', ms=4, label='Thomson reading at each Thomson time')
            ax.plot(bcTimes, profiles[givenName].values / scale, 'k.', ms=6, label='value given to TORAX')
            ax.plot(txTimes, simulated / scale, 'r-', lw=1.2, label='TORAX boundary value (simulated)')
            ax.set_xlabel('time [s]')
            ax.set_ylabel(label)
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=7)

        ax = axes[3]
        teRatio, neRatio = safeRatio(teSim, teGiven), safeRatio(neSim, neGiven)
        ax.plot(txTimes, teRatio, 'r-', label='Te')
        ax.plot(txTimes, neRatio, 'b--', label='ne')
        ax.axhline(1.0, color='k', lw=0.8)
        ax.set_ylim(0.9, 1.1)
        worst = float(np.nanmax(np.abs(np.concatenate([teRatio, neRatio]) - 1)))
        ax.set_xlabel('time [s]')
        ax.set_ylabel('simulated / given at the boundary (no units)')
        ax.set_title(f'Simulated over given, largest deviation {worst:.1e}', fontsize=10)
        ax.legend(fontsize=7)
        source = 'measured from Thomson' if measured else 'constants'
        fig.suptitle(f'Edge check, shot {run["shotId"]} ({run["batchName"]}), edge values {source}')
        fig.tight_layout()
        saveFigure(fig, outDir, f'edge_equivalence_{run["shotId"]}', folder)
        pooled['te'].append(teGiven)
        pooled['teSim'].append(teSim)
        pooled['ne'].append(neGiven)
        pooled['neSim'].append(neSim)
        pooled['runs'] += 1

    if pooled['runs']:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        for ax, (name, scale, unit) in zip(axes, (('te', 1.0, 'Te [keV]'), ('ne', 1e19, 'ne [1e19 m^-3]'))):
            given, simulated = np.concatenate(pooled[name]) / scale, np.concatenate(pooled[name + 'Sim']) / scale
            top = 1.05 * np.nanmax(np.concatenate([given, simulated]))
            ax.plot([0, top], [0, top], 'k-', lw=0.8)
            ax.scatter(given, simulated, s=6, color='tab:blue', alpha=0.6)
            ax.set_xlim(0, top)
            ax.set_ylim(0, top)
            worst = float(np.nanmax(np.abs(safeRatio(simulated, given) - 1)))
            ax.set_xlabel(f'value given to TORAX at the boundary, {unit}')
            ax.set_ylabel(f'TORAX boundary value (simulated), {unit}')
            ax.set_title(f'{len(given)} TORAX output times from {pooled["runs"]} runs, largest deviation {worst:.1e}', fontsize=10)
        fig.suptitle('Simulated boundary value against input edge value, each point one TORAX output time, 1:1 line')
        fig.tight_layout()
        saveFigure(fig, outDir, 'edge_equivalence_all_runs', folder)

def plotShotRuns(allRuns, shotId, outDir, folder=''):
    """
    Every successful run of one shot against the same measurement, one simulated line per run labeled by batch,
    transport model and coupling loops, to show whether a change made the shot better
    """
    quantities = [('teCore', 'core Te at rho 0 [keV]', 1.0), ('teVolAvg', 'vol avg Te [keV]', 1.0), ('neCore', 'core ne at rho 0 [1e19 m^-3]', 1e19),
                  ('neVolAvg', 'vol avg ne [1e19 m^-3]', 1e19), ('we', 'electron stored energy W_e [kJ]', 1e3)]
    runColors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive']
    fig, axes = plt.subplots(1, 5, figsize=(27, 5))
    for index, item in enumerate(collectRunData(allRuns[allRuns['shotId'] == shotId].sort_values('runTimestamp'))):
        run = item['run']
        for ax, (name, label, scale) in zip(axes, quantities):
            # Measurement is the same for every run of the shot, drawn once
            if index == 0:
                ax.plot(item['times'], item['series'][name + 'Meas'] / scale, 'k.', ms=4, label='measured (Thomson)')
            ax.plot(item['times'], item['series'][name + 'Sim'] / scale, color=runColors[index % len(runColors)], lw=1.5,
                    label=f'{run["batchName"]}, {run["cfgTurbulentTransport"]}, loop {run["cfgMaxLoop"]:g}')
    for ax, (name, label, scale) in zip(axes, quantities):
        ax.set_xlabel('time [s]')
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=7)
    fig.suptitle(f'Shot {shotId}, every run of this shot against the same measurement (bursts averaged), core only where Thomson reaches the center')
    fig.tight_layout()
    saveFigure(fig, outDir, f'shot_{shotId}_runs', folder)

def plotBatchComparison(allRuns, batchA, batchB, outDir):
    """
    Shot by shot comparison of two batches, how far each quantity is from the measurement for batch A against batch B,
    below the 1:1 line batch B is closer to the experiment. Returns the shots both batches ran
    """
    columns = [('resTeCoreRatio', 'core Te'), ('resTeVolAvgRatio', 'vol avg Te'), ('resNeVolAvgRatio', 'vol avg ne'), ('resWeRatio', 'electron stored energy')]
    first, second = [allRuns[allRuns['batchName'] == batch].sort_values('runTimestamp').groupby('shotId').tail(1).set_index('shotId') for batch in (batchA, batchB)]
    shared = sorted(set(first.index) & set(second.index))
    colors = first.loc[shared, 'shotHeating'].map(heatingColors).fillna('gray')
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    for ax, (column, label) in zip(axes, columns):
        # Distance from perfect in factors of 2, 0 is a perfect match and 1 is off by 2x either way
        missA = np.abs(np.log2(first.loc[shared, column].astype(float).values))
        missB = np.abs(np.log2(second.loc[shared, column].astype(float).values))
        ax.scatter(missA, missB, c=colors, s=35)
        for shotId, x, y in zip(shared, missA, missB):
            ax.annotate(str(shotId), (x, y), fontsize=6, alpha=0.6)
        top = 1.1 * np.nanmax(np.concatenate([missA, missB, [0.1]]))
        ax.plot([0, top], [0, top], 'k-', lw=0.8)
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_xlabel('|log2 sim / measured|, batch A')
        ax.set_ylabel('|log2 sim / measured|, batch B')
        ax.set_title(f'{label}', fontsize=10)
    heatingLegend(axes[0])
    fig.suptitle(f'A = {batchA}, B = {batchB}, |log2 sim / measured| per shot, 1:1 line')
    fig.tight_layout()
    saveFigure(fig, outDir, 'batch_comparison')
    return shared

def openShotGroup(shotId, group):
    """
    One FAIR-MAST diagnostic group for a shot, data only downloads when read
    """
    response = requests.get(f'{BASE_URL}/json/shots/{shotId}', timeout=(10, 60))
    response.raise_for_status()
    shot = response.json()
    return xr.open_zarr(shot['url'].replace('s3:/', shot['endpoint_url']), group=group)

def fetchShotParameters(shotId, zeff=1.7):
    """
    Normalized beta, electron collisionality, Greenwald fraction and q95 as medians over the current flattop,
    from EFIT and the Thomson midplane chord, zeff is the same value the runs assume
    """
    efm = openShotGroup(shotId, 'efm')[['plasma_current_c', 'betan', 'betap', 'li', 'q_95', 'minor_radius', 'geom_axis_rc']].load()
    try:
        thomson = openShotGroup(shotId, 'ayc')[['te', 'ne', 'radius']].load()
    except (KeyError, FileNotFoundError):
        thomson = openShotGroup(shotId, 'atm')[['te', 'ne', 'radius']].load()

    # Flattop is every reliable EFIT slice within 10% of peak current, same reliability rule as the simulated window,
    # max raises ValueError when no slice is reliable
    ip = np.abs(efm['plasma_current_c'].values)
    reliable = (efm['betap'].values > 0) & (efm['li'].values < 2.0) & np.isfinite(ip)
    flattop = reliable & (ip >= 0.9 * ip[reliable].max())
    flat = efm.isel(time=np.flatnonzero(flattop))
    tStart, tEnd = float(flat['time'].min()), float(flat['time'].max())
    rGeo, minorRadius = float(flat['geom_axis_rc'].median()), float(flat['minor_radius'].median())

    # Line averages along the Thomson midplane chord at each flattop Thomson time, only channels inside the plasma,
    # picked by index since older Thomson repeats each time stamp with an empty copy
    neLines, teLines = [], []
    for i in np.flatnonzero((thomson['time'].values >= tStart) & (thomson['time'].values <= tEnd)):
        profile = thomson.isel(time=i)
        r, te, ne = np.ravel(profile['radius'].values), profile['te'].values, profile['ne'].values
        inside = np.isfinite(r) & np.isfinite(te) & np.isfinite(ne) & (te > 0) & (ne > 0) & (np.abs(r - rGeo) < minorRadius)
        if np.sum(inside) < 5:
            continue
        order = np.argsort(r[inside])
        rIn, teIn, neIn = r[inside][order], te[inside][order], ne[inside][order]
        neLines.append(np.trapezoid(neIn, rIn) / (rIn[-1] - rIn[0]))
        teLines.append(np.trapezoid(teIn, rIn) / (rIn[-1] - rIn[0]))
    if not neLines:
        raise ValueError('no Thomson profile with 5 channels inside the plasma during the flattop')
    neLine, teLine = float(np.median(neLines)), float(np.median(teLines))

    ipMA = float(np.median(ip[flattop])) / 1e6
    q95 = float(np.median(np.abs(flat['q_95'].values)))
    epsilon = minorRadius / rGeo
    # Electron collisionality with ne in m^-3 and Te in eV, Coulomb log for Te above 10 eV
    coulombLog = 31.3 - np.log(np.sqrt(neLine) / teLine)
    nuStar = 6.921e-18 * q95 * rGeo * neLine * zeff * coulombLog / (teLine ** 2 * epsilon ** 1.5)
    greenwaldDensity = 1e20 * ipMA / (np.pi * minorRadius ** 2)
    return {'shotId': shotId, 'flattopStart': tStart, 'flattopEnd': tEnd, 'ipMA': ipMA, 'betaN': float(flat['betan'].median()), 'q95': q95,
            'neLine': neLine, 'teLine': teLine, 'greenwaldFraction': neLine / greenwaldDensity, 'nuStar': float(nuStar), 'note': ''}

def loadShotParameters(shotIds, workers=8):
    """
    Shot parameters from pulse_database/shot_parameters.csv, fetching and adding any shots not in it yet.
    Shots missing data are saved with a note so they are skipped next time, network failures are tried again next run
    """
    try:
        saved = pd.read_csv(shotParametersPath)
        known = set(saved['shotId'])
    except FileNotFoundError:
        saved, known = None, set()
    missing = sorted(set(int(shotId) for shotId in shotIds) - known)
    if missing:
        print(f'fetching the operating point of {len(missing)} shots from FAIR-MAST')
        rows = []
        failedInRow = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetchShotParameters, shotId): shotId for shotId in missing}
            for done, future in enumerate(as_completed(futures), 1):
                shotId = futures[future]
                try:
                    rows.append(future.result())
                    failedInRow = 0
                except requests.RequestException as e:
                    failedInRow += 1
                    print(f'could not reach FAIR-MAST for shot {shotId}: {e}')
                except (KeyError, FileNotFoundError, ValueError, IndexError) as e:
                    rows.append({'shotId': shotId, 'note': f'{type(e).__name__}: {e}'})
                    failedInRow = 0
                except Exception as e:
                    failedInRow += 1
                    print(f'could not fetch shot {shotId}: {type(e).__name__}: {e}')
                # Saved as it goes so an interrupted fetch keeps what it has
                if done % 50 == 0 or done == len(missing) or failedInRow >= 20:
                    if rows:
                        pd.concat([saved, pd.DataFrame(rows)], ignore_index=True).to_csv(shotParametersPath, index=False)
                    print(f'{done} of {len(missing)} shots checked')
                # Stop when FAIR-MAST is likely down
                if failedInRow >= 20:
                    print('20 network failures in a row, stopping the fetch, rerun later to fill in the rest')
                    for pending in futures:
                        pending.cancel()
                    break
        if rows:
            saved = pd.concat([saved, pd.DataFrame(rows)], ignore_index=True)
    if saved is None:
        return pd.DataFrame(columns=['shotId', 'betaN', 'nuStar', 'q95', 'greenwaldFraction'])
    return saved[saved['shotId'].isin(set(int(shotId) for shotId in shotIds))]

def plotOperatingSpace(batchName, outDir, workers=8):
    """
    Where the simulated shots sit in plasma regime space against every usable shot in gray, colored by how
    each run went, collisionality against normalized beta and q95 against Greenwald fraction
    """
    runs = pd.read_csv(runsPath, dtype={'configHash': str, 'codeHash': str})
    if batchName is not None:
        runs = runs[runs['batchName'] == batchName]
    usable = set(pd.read_csv(usablePath)['shot_id'])
    params = loadShotParameters(usable | set(runs['shotId']), workers).set_index('shotId')

    # Latest run of each shot, a failed run, a run that finished with numerical warnings, or a healthy run
    latest = runs.sort_values('runTimestamp').groupby('shotId').tail(1).set_index('shotId')
    warning = (latest['healthTmFailed'] > 0) | (latest['healthFluxGap'] > 0.1)
    latest['runOutcome'] = np.select([latest['runStatus'] != 'ok', warning], ['failed', 'warning'], default='healthy')
    outcomes = {'healthy': ('tab:blue', 'o', 'ran, numerically healthy'),
                'warning': ('tab:orange', '^', 'ran, failed TokaMaker solve or flux gap over 10%'),
                'failed': ('tab:red', 'X', 'did not finish')}

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    panels = [('nuStar', 'betaN', r'electron collisionality $\nu^*_e$ (no units)', r'normalized beta $\beta_N$ [% m T / MA]'),
              ('q95', 'greenwaldFraction', 'edge safety factor q95 (no units)', 'Greenwald fraction, density over the density limit (no units)')]
    background = params[params.index.isin(usable)]
    for ax, (xName, yName, xLabel, yLabel) in zip(axes, panels):
        shown = background[[xName, yName]].dropna()
        ax.scatter(shown[xName], shown[yName], s=8, color='0.8', label=f'usable shots (n={len(shown)})', zorder=1)
        for outcome, (color, marker, label) in outcomes.items():
            points = params.reindex(latest.index[latest['runOutcome'] == outcome])[[xName, yName]].dropna()
            ax.scatter(points[xName], points[yName], s=45, color=color, marker=marker, edgecolors='white', linewidths=0.8, label=f'{label} (n={len(points)})', zorder=3)
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        ax.grid(color='0.92', lw=0.6)
        ax.set_axisbelow(True)
    # Log scale needs at least one positive collisionality, an empty table stays linear
    axes[0].set_xscale('log' if (params['nuStar'] > 0).any() else 'linear')
    axes[0].legend(fontsize=8)
    noPoint = sorted(set(latest.index) - set(params.dropna(subset=['betaN', 'nuStar']).index))
    if noPoint:
        print(f'no operating point for simulated shots {noPoint}')
    fig.suptitle('Simulated shots against every usable shot, medians over the current flattop')
    fig.tight_layout()
    saveFigure(fig, outDir, 'operating_space', 'checks')

def newestBatch(runs):
    """
    Name of the batch with the most recent successful run
    """
    try:
        return runs.sort_values('runTimestamp')['batchName'].iloc[-1]
    except IndexError:
        raise SystemExit('runs.csv has no successful runs yet, run a batch first')

def resolveBatch(runs, name):
    """
    Full batch name from either the full name or just the label the batch ran with, the newest one when a label was
    used more than once
    """
    names = sorted(runs['batchName'].unique())
    # Batch names are the date and time, YYYY-MM-DD_HHMM_, then the label
    matches = [batch for batch in names if batch == name or batch[16:] == name]
    try:
        chosen = matches[-1]
    except IndexError:
        raise SystemExit(f'no successful runs in a batch called {name}, batches with successful runs: {", ".join(names) or "none yet"}')
    if len(matches) > 1:
        print(f'{len(matches)} batches are labeled {name}, using the newest, {chosen}')
    return chosen

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the pulse database, every figure saved under plots/ in an svg tree and a matching pdf tree')
    parser.add_argument('--batch', help='plot this batch, by full name or by the label it ran with, the newest batch when left out')
    parser.add_argument('--all-batches', action='store_true', help='plot every successful run of every batch together, saved in plots/all_batches')
    parser.add_argument('--out', default='plots', help='root folder for every figure, plots when left out')
    parser.add_argument('--operating-space', action='store_true', help='also plot shots in plasma regime space, fetching any shots not yet in pulse_database/shot_parameters.csv')
    parser.add_argument('--run-figures', action='store_true', help='per run figures for every run in the batch, saved in plots/<batch>/svg/runs/<shot>')
    parser.add_argument('--shots', type=int, nargs='+', help='per run figures for every run of these shot numbers in any batch, saved in plots/shots/<shot>/svg/<batch>, plus one figure per shot overlaying all of its runs')
    parser.add_argument('--profile-lines', type=int, default=10, help='how many times the profile evolution figures draw, any number works since every TORAX output time is saved')
    parser.add_argument('--compare', nargs=2, metavar=('BATCH_A', 'BATCH_B'), help='compare two batches shot by shot, by full name or label, saved in plots/comparisons/<A>_vs_<B>')
    parser.add_argument('--edge-equivalence', action='store_true', help='temporary edge check showing the simulated boundary equals the edge values TORAX was given, saved in svg/checks/edge_equivalence')
    args = parser.parse_args()

    allRuns = loadRuns()
    batchName = None if args.all_batches else (resolveBatch(allRuns, args.batch) if args.batch else newestBatch(allRuns))
    runs = loadRuns(batchName)
    outDir = os.path.join(args.out, batchName or 'all_batches')
    print(f'{len(runs)} successful runs in {batchName or "every batch"}, plots go to {outDir}')
    plotRatioOverview(runs, outDir)
    plotRatioVsShot(runs, outDir)
    plotParity(runs, outDir)
    plotRatioInTime(runs, outDir)
    plotHealth(runs, outDir)
    plotDatabaseCoverage(runs, outDir)
    plotCompensatingErrors(runs, outDir)
    plotDensityVsGreenwald(runs, outDir)
    runData = collectRunData(runs)
    if runData:
        plotRatioVsRadius(runData, outDir)
        plotRatioMap(runData, outDir)
        plotPeaking(runData, outDir)
        plotCoreVsAverage(runData, outDir)
        plotEdgeCheck(runData, outDir)
        plotParityAllTimes(runData, outDir)
        plotRatioHistograms(runData, outDir)
        plotTimeTraceGrid(runData, outDir)
    else:
        print('no readable profiles files, skipping the profile based plots')
    if args.run_figures:
        plotRunFigures(runs, outDir, args.profile_lines)
    if args.edge_equivalence:
        plotEdgeEquivalence(runs, outDir)
    if args.operating_space:
        plotOperatingSpace(batchName, outDir)
    if args.shots:
        shotsDir = os.path.join(args.out, 'shots')
        print(f'no successful runs for shots {sorted(set(args.shots) - set(allRuns["shotId"]))}' if set(args.shots) - set(allRuns['shotId']) else f'shot figures for {sorted(args.shots)} go to {shotsDir}')
        plotRunFigures(allRuns[allRuns['shotId'].isin(args.shots)], shotsDir, args.profile_lines, byShot=True)
        for shotId in sorted(set(args.shots) & set(allRuns['shotId'])):
            plotShotRuns(allRuns, shotId, os.path.join(shotsDir, str(shotId)))
    if args.compare:
        batchA, batchB = (resolveBatch(allRuns, name) for name in args.compare)
        compareDir = os.path.join(args.out, 'comparisons', f'{batchA}_vs_{batchB}')
        shared = plotBatchComparison(allRuns, batchA, batchB, compareDir)
        print(f'{len(shared)} shots in both batches, comparison saved to {compareDir}')
        for shotId in shared:
            plotShotRuns(allRuns, shotId, compareDir, 'shots')
