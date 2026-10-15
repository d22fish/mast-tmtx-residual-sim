"""
Metrics of the pulse database written as text under metrics/, one folder per table with one file per batch
"""

import argparse
import os

import pandas as pd

from plot_db import heatingLabels, newestBatch, resolveBatch, runsPath, usablePath

catalogPath = os.path.join('MAST_info', 'fairmast_shots_catalog.parquet')
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
    args = parser.parse_args()

    allRuns = pd.read_csv(runsPath, dtype={'configHash': str, 'codeHash': str})
    okRuns = allRuns[allRuns['runStatus'] == 'ok']
    batchName = None if args.all_batches else (resolveBatch(okRuns, args.batch) if args.batch else newestBatch(okRuns))
    runs = allRuns if batchName is None else allRuns[allRuns['batchName'] == batchName]
    path = os.path.join(args.out, 'database_summary', f'{batchName or "all_batches"}.txt')
    writeMetrics(databaseSummary(runs), path)
    print(f'{len(runs)} runs in {batchName or "every batch"}, metrics written to {path}')
