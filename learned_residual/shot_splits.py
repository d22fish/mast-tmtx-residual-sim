"""
Shot clusters and frozen train, validation, test and calibration splits for the learned residual

Every usable shot gets a cluster from its heating and its peak plasma current, a family of near repeat shots
from the reference shots it took its settings from, and one split. Campaign M9 is held out. The test set is
fixed by a hash 
"""

import argparse
import hashlib
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

projectDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
usablePath = os.path.join(projectDir, 'MAST_info', 'usable_pulses.csv')
catalogPath = os.path.join(projectDir, 'MAST_info', 'fairmast_shots_catalog.parquet')
defaultOut = os.path.join(projectDir, 'MAST_info', 'shot_splits.csv')

heatingShort = {'Ohmic': 'Ohmic', 'SS Beam': 'SS', 'SW Beam': 'SW', '2 Beams,SS Beam,SW Beam': '2B'}
splitFractions = {'test': 1 / 7, 'validation': 1 / 7, 'train': 5 / 7}

def loadShots():
    """
    Usable shots with campaign, heating, peak current, reference shot, and EFIT magnetic axis height
    """
    shots = pd.read_csv(usablePath)
    catalog = pd.read_parquet(catalogPath, columns=['shot_id', 'rad_magnetic_height_efit'])
    return shots.merge(catalog, on='shot_id', how='left')

def shotCluster(heating, peakCurrent, cut=550.0):
    """
    Cluster from heating and peak plasma current in kA, split at the cut between the 400 kA and 700 kA scenario groups,
    beam shots below the cut pooled into one cluster since not many beam shots below threshold
    """
    group = f'below{cut:g}kA' if abs(peakCurrent) < cut else f'above{cut:g}kA'
    if heating == 'Ohmic':
        return f'Ohmic_{group}'
    if abs(peakCurrent) < cut:
        return f'beam_{group}'
    return f'{heatingShort[heating]}_{group}'

def shotFamilies(shots):
    """
    Joins each shot to the shot it took its settings from, every connected group is one family named by its root shot
    """
    parent = {}

    def root(shotId):
        """
        Root of the family a shot belongs to, path compressed
        """
        parent.setdefault(shotId, shotId)
        while parent[shotId] != shotId:
            parent[shotId] = parent[parent[shotId]]
            shotId = parent[shotId]
        return shotId

    for shotId, referenceShot in zip(shots['shot_id'], shots['reference_shot']):
        root(shotId)
        if pd.notna(referenceShot):
            parent[root(shotId)] = root(int(referenceShot))
    return shots['shot_id'].map(root)

def assignFamilies(pool, seed):
    """
    Grouped and stratified split of the pool. Families go whole into one split, largest first, each to the split whose
    cluster counts sit furthest below its share, so every split holds every cluster in proportion
    """
    rng = np.random.default_rng(seed)
    clusterTotals = pool['cluster'].value_counts()
    counts = {name: pd.Series(0.0, index=clusterTotals.index) for name in splitFractions}
    families = pool.groupby('family')['cluster'].value_counts().unstack(fill_value=0)
    # Random order first so equal sized families are taken in a seeded order, then largest first
    order = families.index[rng.permutation(len(families))]
    order = sorted(order, key=lambda family: -families.loc[family].sum())
    assignment = {}
    for family in order:
        members = families.loc[family]
        deficits = {name: float(np.sum(members * (fraction * clusterTotals - counts[name]) / clusterTotals)) for name, fraction in splitFractions.items()}
        chosen = max(deficits, key=deficits.get)
        counts[chosen] = counts[chosen].add(members, fill_value=0)
        assignment[family] = chosen
    return pool['family'].map(assignment)

def pickCalibration(shots, perCluster, seed):
    """
    Calibration shots, perCluster train shots of each cluster from as many different families, taken out of train
    """
    rng = np.random.default_rng(seed)
    picked = []
    for _, members in shots[shots['split'] == 'train'].groupby('cluster'):
        oneEach = members.groupby('family').sample(1, random_state=seed)
        picked += list(oneEach.index[rng.permutation(len(oneEach))[:perCluster]])
    return picked

def pickStability(shots, count, seed):
    """
    Train shots for the numerical stability reruns, spread over the clusters with at most one per family
    """
    rng = np.random.default_rng(seed)
    train = shots[(shots['split'] == 'train')].drop_duplicates('family')
    clusters = list(train['cluster'].value_counts().index)
    picked = []
    # Clusters are visited largest first, one shot each pass, until count shots are picked
    while len(picked) < count:
        for name in clusters:
            candidates = train[(train['cluster'] == name) & ~train.index.isin(picked)]
            if len(candidates) and len(picked) < count:
                picked.append(candidates.index[rng.integers(len(candidates))])
    return picked

def testHash(shots):
    """
    First 12 characters of the sha256 of the sorted test and held out campaign shot numbers
    """
    testShots = sorted(int(shotId) for shotId in shots.loc[shots['split'].isin(['test', 'testCampaign']), 'shot_id'])
    return hashlib.sha256(','.join(str(shotId) for shotId in testShots).encode()).hexdigest()[:12]

def buildSplits(seed=0, cut=550.0, perCluster=12, stabilityCount=10, offAxisLimit=-0.10, heldOutCampaign='M9'):
    """
    Every usable shot with its cluster, family, split, off-axis flag and stability flag
    """
    shots = loadShots()
    shots['heating'] = shots['heatingResolved']
    shots['peakCurrentKA'] = shots['plasma_max_current'].abs()
    shots['cluster'] = [shotCluster(heating, current, cut) for heating, current in zip(shots['heating'], shots['peakCurrentKA'])]
    shots['family'] = shotFamilies(shots)
    # Axis more than offAxisLimit below the midplane puts the Z=0 Thomson chord off the core, keep shots with no catalog value
    shots['offAxis'] = shots['rad_magnetic_height_efit'] < offAxisLimit

    shots['split'] = ''
    heldOut = shots['campaign'] == heldOutCampaign
    shots.loc[heldOut, 'split'] = 'testCampaign'
    pool = shots[shots['split'] == '']
    shots.loc[pool.index, 'split'] = assignFamilies(pool, seed)

    shots.loc[pickCalibration(shots, perCluster, seed), 'split'] = 'calibration'
    shots['stability'] = False
    shots.loc[pickStability(shots, stabilityCount, seed), 'stability'] = True
    columns = ['shot_id', 'campaign', 'heating', 'peakCurrentKA', 'cluster', 'family', 'split', 'offAxis', 'stability']
    return shots[columns].sort_values('shot_id').reset_index(drop=True)

def writeSplits(shots, outPath, settings):
    """
    Writes the splits once and keeps any existing splits file as it is, plus a doc with the settings, the cluster by
    split counts and the test hash
    """
    table = pd.crosstab(shots['cluster'], shots['split'], margins=True)
    manifest = {'created': datetime.now().isoformat(timespec='seconds'), 'testHash': testHash(shots), **settings,
                'counts': {name: {str(k): int(v) for k, v in row.items()} for name, row in table.iterrows()}}
    try:
        with open(outPath, 'x') as splitsFile:
            shots.to_csv(splitsFile, index=False)
    except FileExistsError:
        raise SystemExit(f'{outPath} already exists and the splits are frozen, delete it on purpose to make new ones')
    with open(os.path.splitext(outPath)[0] + '_manifest.json', 'w') as manifestFile:
        json.dump(manifest, manifestFile, indent=1)
    print(table.to_string())
    print(f'\n{len(shots)} shots written to {outPath}, test hash {manifest["testHash"]}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make the frozen shot clusters and splits for the learned residual')
    parser.add_argument('--out', default=defaultOut, help='splits csv path, a manifest json is written next to it')
    parser.add_argument('--seed', type=int, default=0, help='seed for the family order and the calibration and stability picks')
    parser.add_argument('--cut', type=float, default=550.0, help='peak current in kA between the 400 kA and 700 kA groups')
    parser.add_argument('--per-cluster', type=int, default=12, help='calibration shots per cluster')
    parser.add_argument('--dry-run', action='store_true', help='print the counts and the test hash without writing anything')
    args = parser.parse_args()

    settings = {'seed': args.seed, 'cut': args.cut, 'perCluster': args.per_cluster, 'heldOutCampaign': 'M9', 'offAxisLimit': -0.10, 'stabilityCount': 10}
    shots = buildSplits(seed=args.seed, cut=args.cut, perCluster=args.per_cluster)
    if args.dry_run:
        print(pd.crosstab(shots['cluster'], shots['split'], margins=True).to_string())
        print(f'\ntest hash {testHash(shots)}, nothing written')
    else:
        writeSplits(shots, args.out, settings)
