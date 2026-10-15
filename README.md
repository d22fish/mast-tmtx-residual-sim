# MAST Parameter Matching

Tools for pulling MAST equilibrium and diagnostic data from FAIR-MAST, building a TokaMaker mesh, and mapping shot data onto TokaMaker/TORAX input formats.

## Scope

Data comes from the public FAIR-MAST catalog (mastapp.site), which currently covers legacy MAST shots only (campaigns M5-M9, 2004-2013). 

## Setup

HDF5 and pkg-config are required before installing Python dependencies, since h5py compiles from source on recent Python versions.

    brew install hdf5 pkg-config
    pip install -r requirements.txt

## Files

- MAST_mesh_generator.ipynb: builds a TokaMaker mesh from FAIR-MAST's pf_active coil geometry and wall contour, saves to MAST_mesh.h5
- data_cleaning.ipynb: screens MAST dataset to determine usable pulses
- MAST_pulse_batch.py: runs MAST pulses through TokaMaker and TORAX, singly or in parallel batches, and builds the pulse database

- pulse_database/runs.csv: one row per run of a pulse with its inputs, run health checks and sim vs experiment comparisons
- pulse_database/data_dictionary.csv: units and descriptions for every column in runs.csv and every variable in the per-run profile files

- MAST_info/usable_pulses.csv, screened_pulses.csv, data_availability.csv: data_cleaning outputs, the shots a batch can draw from and why the rest were excluded
- MAST_info/fairmast_shots_catalog.parquet, mast_efit_passive_structure_M4.csv: FAIR-MAST shot catalog and EFIT passive structure

- testing_notebooks/MAST_parameter_matching.ipynb: loads the mesh, pulls equilibrium and diagnostic data for a shot, maps it onto TokaMaker/TORAX input formats
- testing_notebooks/MAST_pulse.ipynb: basic test of one simulated MAST pulse using input from experimental data for sim
- testing_notebooks/MAST_pulse_batch.ipynb: notebook version of MAST_pulse_batch.py, kept for testing

## Mesh resolution convergence validation

- Swept plasma_dx, coil_dx, and vac_dx together at their current values, then .5 and .25 (14440, 55414, 216756 cells) and ran a raw TokaMaker solve at each resolution. kappa, beta_pol, q_95, and W_MHD all move by less than 0.02% between the two finest levels, and the change shrinks by roughly 4x each time the mesh spacing is halved.

## Data visualization

- Shot catalog: https://mastapp.site/parquet/shots.parquet (15,969 shots x 189 features)
- efm: EFIT equilibrium reconstruction (plasma shape, current, q-profile, flux)
- ayc/atm: Thomson scattering (Te, ne profiles), atm on older shots
- act: CXRS ion temperature
- anb: NBI injected power
- ane: interferometer line integrated density
- abm: bolometer radiated power
- gas_injection, pf_active (level 2): gas puff and PF coil currents

## Run a batch

From the terminal, rerunning the same command picks an unfinished batch back up:

    python MAST_pulse_batch.py --heating Ohmic --campaigns M9 --sample 10 --workers 4 --max-loop 2

Other flags:
- --transport qlknn: QLKNN or default TGLF-NN
- --label <name>: names the batch folder, default based on shot selection
- --shots <ids>: run an explicit shot list
- --no-plots: skip the comparison figures (faster for large batches)

From a notebook cell (run from the project folder):

    from MAST_pulse_batch import selectShots, runBatch
    shotIds, screenNotes, selectionName = selectShots(heating=['Ohmic'], campaigns=['M9'], sampleSize=10)
    rows = runBatch(shotIds, screenNotes=screenNotes, selectionName=selectionName, workers=4, maxLoop=2, turbulenceModel='tglfnn')


## Plot the database

plot_db.py reads pulse_database/runs.csv and each run's profiles file, and saves every figure twice, under plots/.../svg and plots/.../pdf. Every command makes the batch figures (accuracy, shape, checks, traces) for one batch, the newest unless --batch is given, and the flags below add figures on top. --batch and --compare take a full batch name (2026-10-12_2152_dbv2) or just its label (dbv2).

    python3 plot_db.py

Batch figures for the newest batch, in plots/<batch>/

    python3 plot_db.py --batch dbv2 --run-figures

Also each run's own figures (Te, ne, Ti, stored energy and radiation, profile evolution and slices, equilibrium, plasma boundary), in plots/<batch>/svg/runs/<shot>/

    python3 plot_db.py --shots 25012 27972

Every run of these shots from any batch, one folder per batch in plots/shots/<shot>/svg/<batch>/, plus shot_<shot>_runs overlaying all runs of the shot

    python3 plot_db.py --compare dbv1 dbv2

Shot by shot comparison of the shots both batches ran, in plots/comparisons/<A>_vs_<B>/

    python3 plot_db.py --all-batches

Batch figures pooled over every successful run of every batch, in plots/all_batches/

    python3 plot_db.py --operating-space

Adds checks/operating_space, the shots in plasma regime space, fetching parameters from FAIR-MAST for shots not yet in pulse_database/shot_parameters.csv

    python3 plot_db.py --batch edge_test --edge-equivalence

Temporary check that TORAX's boundary Te and ne equal the edge values it was given, in plots/<batch>/svg/checks/edge_equivalence/

Other flags:
- --out <folder>: root folder in place of plots
- --profile-lines <n>: how many times the profile evolution figures draw, default 10

Runs from batches before the profiles files kept equilibrium, boundary, CXRS Ti, bolometer and edge values get the Te, ne, stored energy and profile figures only, and are skipped by --edge-equivalence.

## Calculate metrics

Batch metrics for a given batch in metrics/database_summary

    python3 calc_metrics.py --batch pilot

Batch figures pooled over every successful run of every batch, in plots/all_batches/

    python3 calc_metrics.py --all-batches