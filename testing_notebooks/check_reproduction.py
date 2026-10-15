"""
Reruns one database shot in TORAX alone from its saved config and seed equilibria, and compares the result with the
profiles file the coupled run saved
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torax
import xarray as xr

projectDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runsPath = os.path.join(projectDir, 'pulse_database', 'runs.csv')

def pickRun(batch, shotId=None):
    """
    Newest successful run in a batch, found by full batch name or by the label it ran with
    """
    runs = pd.read_csv(runsPath, dtype={'configHash': str, 'codeHash': str})
    # Batch names start with the date and time, YYYY-MM-DD_HHMM_, then the label
    keep = (runs['runStatus'] == 'ok') & ((runs['batchName'] == batch) | (runs['batchName'].str[16:] == batch))
    if shotId is not None:
        keep &= runs['shotId'] == shotId
    try:
        return runs[keep].sort_values('runTimestamp').iloc[-1]
    except IndexError:
        raise SystemExit(f'no successful run in batch {batch}' + (f' for shot {shotId}' if shotId is not None else ''))

def rerunTorax(run):
    """
    Loads the saved TORAX config, points its geometry at the seed files in the run folder and runs TORAX on its own
    """
    runDir = os.path.join(projectDir, run['runOutputDir'])
    with open(os.path.join(runDir, 'torax_config.pkl'), 'rb') as configFile:
        config = pickle.load(configFile)
    # Seed paths are relative to the run folder, so the rerun still finds them after the folder moves
    config['geometry']['geometry_directory'] = runDir
    dataTree, history = torax.run_simulation(torax.ToraxConfig.from_dict(config), log_timestep_info=False)
    print(f'TORAX alone finished with {history.sim_error}')
    return dataTree.profiles.to_dataset()

def compareProfiles(rerun, profilesPath, tolerance=1e-3):
    """
    Largest difference between the rerun and the saved profiles for each quantity, as a fraction of the largest saved
    value, and whether every quantity falls within tolerance
    """
    passed = True
    with xr.open_dataset(profilesPath) as saved:
        for name in ('T_e', 'n_e', 'T_i', 'psi', 'q'):
            try:
                rerunValues, savedValues = xr.align(rerun[name], saved[name], join='inner')
                difference = float(np.abs(rerunValues - savedValues).max()) / float(np.abs(savedValues).max())
            except KeyError:
                print(f'{name}: not in both files, skipped')
                continue
            except ValueError:
                print(f'{name}: no times and radii in common between the two files')
                passed = False
                continue
            within = difference <= tolerance
            passed &= within
            print(f'{name}: {rerunValues.size} points compared, largest difference {difference:.1e} of the largest value, {"ok" if within else "FAILED"}')
    return passed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rerun one database shot in TORAX alone and compare it with the saved coupled run')
    parser.add_argument('--batch', default='repro', help='batch to check, by full name or by the label it ran with')
    parser.add_argument('--shot', type=int, help='shot to check, the newest successful run in the batch when left out')
    args = parser.parse_args()

    run = pickRun(args.batch, args.shot)
    print(f'checking shot {run["shotId"]} from {run["batchName"]}')
    rerun = rerunTorax(run)
    passed = compareProfiles(rerun, os.path.join(projectDir, run['runProfilesPath']))
    print('reproduction passed, the saved config and seeds rebuild the database run' if passed else 'reproduction failed, the saved config and seeds miss something the coupled run used')
