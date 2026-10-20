"""
Measurement targets for the learned residual, one file per shot

For each shot the raw Thomson channels with their errors, the CXRS carbon Ti and rotation with their errors, the
interferometer line density and the EFIT boundary flux, current and minor radius over the simulated window. Every
Thomson and CXRS point carries its psiN and rho on the TORAX boundary at the chosen psiN, a burst index, and the masks
the loss uses
"""

import argparse
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

projectDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, projectDir)
from MAST_pulse_batch import loadShotWindow, openShotGroup, openThomsonGroup, psiNormAtMidplane, rhoTorNormFromPsiNorm

usablePath = os.path.join(projectDir, 'MAST_info', 'usable_pulses.csv')
defaultOut = os.path.join(projectDir, 'pulse_database', 'targets')

# netCDF writes run one thread at a time
writeLock = threading.Lock()

def burstIndex(times, maxGap=0.002):
    """
    Burst number of each laser time, a new burst starts wherever the gap to the previous laser is over maxGap seconds
    """
    return np.cumsum(np.concatenate([[False], np.diff(times) > maxGap])).astype(int)

def mapToFlux(r, eq, boundary):
    """
    psiN and rho on the TORAX boundary for midplane major radii, NaN where the radius is missing
    """
    psiN, rho = np.full(r.shape, np.nan), np.full(r.shape, np.nan)
    finite = np.isfinite(r)
    try:
        if finite.any():
            psiN[finite] = psiNormAtMidplane(r[finite], eq)
            rho[finite] = rhoTorNormFromPsiNorm(psiN[finite], eq, boundary)
    except ValueError:
        # EFIT slice with an empty flux grid, its points stay NaN and drop out as invalid
        pass
    return psiN, rho

def lossMask(valid, psiN, boundary, fitHalfWidth):
    """
    Points the loss uses, valid, inside the TORAX boundary, and outside the band the edge boundary values were fit in
    """
    with np.errstate(invalid='ignore'):
        return valid & (psiN < boundary) & (np.abs(psiN - boundary) > fitHalfWidth)

def thomsonTargets(shotId, efm, tStart, tEnd, boundary, fitHalfWidth):
    """
    Raw Thomson Te and ne with errors at every Thomson time in the window, Te in keV and ne in m^-3
    """
    thomson = openThomsonGroup(shotId)[['te', 'te_error', 'ne', 'ne_error', 'radius']].load()
    te, ne = thomson['te'].values, thomson['ne'].values
    teError, neError = thomson['te_error'].values, thomson['ne_error'].values
    # Older Thomson stores one radius per channel, newer stores one per time and channel
    radius = np.broadcast_to(thomson['radius'].values, te.shape)
    times = thomson['time'].values
    # Index by position, older Thomson repeats each time stamp with an empty copy that the valid check drops
    keep = np.flatnonzero(~np.isnan(te).all(axis=1) & (times >= tStart) & (times <= tEnd))
    psiN, rho = np.full(te[keep].shape, np.nan), np.full(te[keep].shape, np.nan)
    for row, i in enumerate(keep):
        psiN[row], rho[row] = mapToFlux(radius[i].astype(float), efm.sel(time=times[i], method='nearest'), boundary)
    with np.errstate(invalid='ignore'):
        valid = np.isfinite(te[keep]) & np.isfinite(ne[keep]) & (te[keep] > 0) & (ne[keep] > 0) & (teError[keep] > 0) & (neError[keep] > 0) & np.isfinite(psiN)
    dims = ('thomson_time', 'thomson_channel')
    return xr.Dataset({
        'teMeas': (dims, te[keep] / 1000.0), 'teSigma': (dims, teError[keep] / 1000.0),
        'neMeas': (dims, ne[keep]), 'neSigma': (dims, neError[keep]),
        'thomsonRadius': (dims, radius[keep]), 'thomsonPsiN': (dims, psiN), 'thomsonRho': (dims, rho),
        'thomsonValid': (dims, valid.astype(np.int8)), 'thomsonLossMask': (dims, lossMask(valid, psiN, boundary, fitHalfWidth).astype(np.int8)),
        'thomsonBurst': ('thomson_time', burstIndex(times[keep])),
    }, coords={'thomson_time': times[keep]})

def cxrsTargets(shotId, efm, tStart, tEnd, boundary, fitHalfWidth):
    """
    CXRS carbon Ti in keV and toroidal velocity in m/s with errors at every CXRS frame in the window, empty when the
    shot has no act group
    """
    try:
        act = openShotGroup(shotId, 'act', level=1)
    except KeyError:
        return xr.Dataset()
    names = [name for name in ('pla_temperature', 'pla_temperature_error', 'pla_velocity', 'pla_velocity_error') if name in act.data_vars]
    if 'pla_temperature' not in names or 'pla_temperature_error' not in names:
        return xr.Dataset()
    act = act[names + ['majorradius']].sortby('time').sel(time=slice(tStart, tEnd)).load()
    times = act['time'].values
    ti, tiError = act['pla_temperature'].values, act['pla_temperature_error'].values
    radius = np.broadcast_to(act['majorradius'].values.astype(float), ti.shape)
    psiN, rho = np.full(ti.shape, np.nan), np.full(ti.shape, np.nan)
    for i, t in enumerate(times):
        psiN[i], rho[i] = mapToFlux(radius[i], efm.sel(time=t, method='nearest'), boundary)
    # Same quality filter the pulse setup uses for CXRS Ti
    with np.errstate(invalid='ignore'):
        valid = np.isfinite(ti) & (ti > 0) & (ti < 5000) & (tiError > 0) & (tiError < 0.3 * ti) & np.isfinite(psiN)
    dims = ('cxrs_time', 'cxrs_channel')
    targets = xr.Dataset({
        'tiMeas': (dims, ti / 1000.0), 'tiSigma': (dims, tiError / 1000.0), 'cxrsRadius': (dims, radius), 'cxrsPsiN': (dims, psiN), 'cxrsRho': (dims, rho),
        'cxrsValid': (dims, valid.astype(np.int8)), 'cxrsLossMask': (dims, lossMask(valid, psiN, boundary, fitHalfWidth).astype(np.int8)),
    }, coords={'cxrs_time': times})
    if 'pla_velocity' in names and 'pla_velocity_error' in names:
        targets['vTorMeas'] = (dims, act['pla_velocity'].values)
        targets['vTorSigma'] = (dims, act['pla_velocity_error'].values)
    return targets

def interferometerTargets(shotId, tStart, tEnd):
    """
    Interferometer line integrated density in m^-2 over the window, empty when the shot has no ane group
    """
    try:
        density = openShotGroup(shotId, 'ane', level=1)['density'].sel(time=slice(tStart, tEnd)).load()
    except KeyError:
        return xr.Dataset()
    return xr.Dataset({'lineDensity': ('ane_time', density.values)}, coords={'ane_time': density['time'].values})

def efitTargets(efm, validTimes, tStart, tEnd):
    """
    EFIT boundary and axis flux, plasma current and minor radius at every reliable EFIT time in the window
    """
    times = validTimes[(validTimes >= tStart) & (validTimes <= tEnd)]
    window = efm.sel(time=times)
    return xr.Dataset({
        'efitPsiBoundary': ('efit_time', window['psi_boundary'].values), 'efitPsiAxis': ('efit_time', window['psi_axis'].values),
        'efitIp': ('efit_time', np.abs(window['plasma_current_c'].values)), 'efitMinorRadius': ('efit_time', window['minor_radius'].values),
    }, coords={'efit_time': times})

def buildShotTargets(shotId, outDir, boundary=0.9, fitHalfWidth=0.08):
    """
    Builds and writes one shot's targets file over the same window the pulse runs use
    """
    efm, validTimes, tStart, tEnd, _, _, _, _, _ = loadShotWindow(shotId)
    parts = [thomsonTargets(shotId, efm, tStart, tEnd, boundary, fitHalfWidth), cxrsTargets(shotId, efm, tStart, tEnd, boundary, fitHalfWidth),
             interferometerTargets(shotId, tStart, tEnd), efitTargets(efm, validTimes, tStart, tEnd)]
    targets = xr.merge(parts)
    targets.attrs.update({'shotId': int(shotId), 'windowStart': tStart, 'windowEnd': tEnd, 'boundaryPsiN': boundary, 'edgeFitHalfWidth': fitHalfWidth,
                          'sigmaSource': 'Thomson te_error and ne_error, CXRS pla_temperature_error', 'created': datetime.now().isoformat(timespec='seconds')})
    path = os.path.join(outDir, f'{shotId}_targets.nc')
    with writeLock:
        targets.to_netcdf(path + '.tmp')
        os.replace(path + '.tmp', path)
    return {'shotId': shotId, 'status': 'ok', 'thomsonTimes': targets.sizes.get('thomson_time', 0), 'lossPoints': int(targets['thomsonLossMask'].sum()),
            'cxrsTimes': targets.sizes.get('cxrs_time', 0), 'error': ''}

def buildTargets(shotIds, outDir, workers=6, boundary=0.9, fitHalfWidth=0.08):
    """
    Targets for every shot without a file yet on threads, outcomes appended to targets_log.csv in outDir
    """
    os.makedirs(outDir, exist_ok=True)
    toBuild = [shotId for shotId in shotIds if not os.path.exists(os.path.join(outDir, f'{shotId}_targets.nc'))]
    print(f'{len(shotIds) - len(toBuild)} shots already have targets, {len(toBuild)} to build on {workers} threads')
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(buildShotTargets, shotId, outDir, boundary, fitHalfWidth): shotId for shotId in toBuild}
        for done, future in enumerate(as_completed(futures), 1):
            shotId = futures[future]
            try:
                rows.append(future.result())
                print(f'[{done}/{len(toBuild)}] shot {shotId} ok, {rows[-1]["lossPoints"]} Thomson points in the loss')
            except Exception as e:
                rows.append({'shotId': shotId, 'status': 'failed', 'error': f'{type(e).__name__}: {e}'[:300]})
                print(f'[{done}/{len(toBuild)}] shot {shotId} failed: {rows[-1]["error"]}')
                traceback.print_exc()
    if rows:
        logPath = os.path.join(outDir, 'targets_log.csv')
        pd.DataFrame(rows).to_csv(logPath, mode='a', header=not os.path.exists(logPath), index=False)
    return rows

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build the measurement target files for the learned residual')
    parser.add_argument('--shots', type=int, nargs='+', help='shot numbers, every usable shot when left out')
    parser.add_argument('--out', default=defaultOut, help='folder for the targets files')
    parser.add_argument('--workers', type=int, default=6, help='threads fetching shots at once')
    parser.add_argument('--boundary', type=float, default=0.9, help='TORAX boundary in psiN, the same as the pulse runs')
    args = parser.parse_args()

    shotIds = args.shots or sorted(int(shotId) for shotId in pd.read_csv(usablePath)['shot_id'])
    buildTargets(shotIds, args.out, workers=args.workers, boundary=args.boundary)
