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
