"""
Local copy of the FAIR-MAST data the pulse runs read, for every usable pulse

Each shot's needed groups and variables are copied from the FAIR-MAST S3 store into MAST_info/fairmast_cache, laid out
like the store, level1/<shot>.zarr and level2/<shot>.zarr. An empty file complete/<shot> marks a shot with every group
copied. A log records every shot, a rerun skips finished shots and retries failed ones
"""

import argparse
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import xarray as xr

projectDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
usablePath = os.path.join(projectDir, 'MAST_info', 'usable_pulses.csv')
defaultOut = os.path.join(projectDir, 'MAST_info', 'fairmast_cache')
S3_URL = 'https://s3.echo.stfc.ac.uk'

# Groups and variables the pulse runs, plot_db and the targets read, None keeps the whole group
neededGroups = {
    (1, 'efm'): None,
    (1, 'ayc'): ['te', 'te_error', 'ne', 'ne_error', 'radius'],
    (1, 'atm'): ['te', 'te_error', 'ne', 'ne_error', 'radius'],
    (1, 'act'): ['majorradius', 'pla_temperature', 'pla_temperature_error', 'pla_velocity', 'pla_velocity_error'],
    (1, 'anb'): None,
    (1, 'abm'): ['prad_pol'],
    (1, 'ane'): ['density'],
    (2, 'gas_injection'): None,
    (2, 'pf_active'): None,
}

def cachePath(outDir, shotId, level):
    """
    Local zarr store for one shot and level
    """
    return os.path.join(outDir, f'level{level}', f'{shotId}.zarr')

def copyGroup(shotId, level, group, names, outDir):
    """
    Copies one group's variables for one shot, returns the copied size in MB or None when the shot has no such group
    """
    try:
        source = xr.open_zarr(f'{S3_URL}/mast/level{level}/shots/{shotId}.zarr', group=group)
    except KeyError:
        return None
    if names is not None:
        source = source[[name for name in names if name in source.data_vars]]
    data = source.load()
    # Drop the S3 encoding
    for name in data.variables:
        data[name].encoding = {}
    data.to_zarr(cachePath(outDir, shotId, level), group=group, mode='w')
    return data.nbytes / 1e6

def cacheShot(shotId, outDir):
    """
    Copies every needed group for one shot, the older Thomson group only when the newer one is missing
    """
    start = time.time()
    copied, missing, sizeMB = [], [], 0.0
    for (level, group), names in neededGroups.items():
        if group == 'atm' and 'ayc' in copied:
            continue
        size = copyGroup(shotId, level, group, names, outDir)
        if size is None:
            missing.append(group)
        else:
            copied.append(group)
            sizeMB += size
    # Marker file read by the pulse runs, only shots with every group copied get one
    os.makedirs(os.path.join(outDir, 'complete'), exist_ok=True)
    open(os.path.join(outDir, 'complete', str(shotId)), 'w').close()
    return {'shotId': shotId, 'status': 'ok', 'groups': ' '.join(copied), 'missingGroups': ' '.join(missing),
            'sizeMB': round(sizeMB, 2), 'seconds': round(time.time() - start, 1), 'cachedAt': datetime.now().isoformat(timespec='seconds'), 'error': ''}

def cacheDatabase(shotIds, outDir, workers=4):
    """
    Copies every shot missing from the log or failed in it on threads, each outcome appended to the log as it finishes
    """
    os.makedirs(outDir, exist_ok=True)
    logPath = os.path.join(outDir, 'cache_log.csv')
    try:
        log = pd.read_csv(logPath)
        done = set(log.loc[log['status'] == 'ok', 'shotId'])
    except FileNotFoundError:
        done = set()
    toCache = [shotId for shotId in shotIds if shotId not in done]
    print(f'{len(shotIds) - len(toCache)} shots already cached, {len(toCache)} to copy on {workers} threads into {outDir}')
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(cacheShot, shotId, outDir): shotId for shotId in toCache}
        for finished, future in enumerate(as_completed(futures), 1):
            shotId = futures[future]
            try:
                row = future.result()
                outcome = f'ok, {row["sizeMB"]:.1f} MB in {row["seconds"]:.0f} s' + (f', no {row["missingGroups"]}' if row['missingGroups'] else '')
            except Exception as e:
                row = {'shotId': shotId, 'status': 'failed', 'error': f'{type(e).__name__}: {e}'[:300], 'cachedAt': datetime.now().isoformat(timespec='seconds')}
                outcome = f'failed: {row["error"]}'
                traceback.print_exc()
            pd.DataFrame([row]).to_csv(logPath, mode='a', header=not os.path.exists(logPath), index=False)
            # Time left from the average pace so far, threads included
            remaining = (time.time() - start) / finished * (len(toCache) - finished)
            print(f'[{finished}/{len(toCache)}] shot {shotId} {outcome} | about {remaining / 3600:.1f} h left')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Copy the FAIR-MAST data the pulse runs read into a local cache')
    parser.add_argument('--shots', type=int, nargs='+', help='shot numbers, every usable shot when left out')
    parser.add_argument('--out', default=defaultOut, help='cache folder')
    parser.add_argument('--workers', type=int, default=4, help='threads copying shots at once')
    args = parser.parse_args()

    shotIds = args.shots or sorted(int(shotId) for shotId in pd.read_csv(usablePath)['shot_id'])
    cacheDatabase(shotIds, args.out, workers=args.workers)
