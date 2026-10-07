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
- MAST_parameter_matching.ipynb: loads the mesh, pulls equilibrium and diagnostic data for a shot, maps it onto TokaMaker/TORAX input formats
- MAST_pulse.ipynb: basic test of one simulated MAST pulse using input from experimental data for sim
- data_cleaning.ipynb: screens MAST dataset to determine usable pulses
- MAST_pulse_batch.ipynb: refined MAST_pulse.ipynb to make more robust, added ability to run multiple pulses of differing heating types back to back

## Mesh resolution convergence validation

- Swept plasma_dx, coil_dx, and vac_dx together at their current values, then .5 and .25 (14440, 55414, 216756 cells) and ran a raw TokaMaker solve at each resolution. kappa, beta_pol, q_95, and W_MHD all move by less than 0.02% between the two finest levels, and the change shrinks by roughly 4x each time the mesh is .5.

## Data visualization

Data parquet retrieved from https://mastapp.site/parquet/shots.parquet
15,969 shots x 189 features
efm — Basic EFIT, equilibrium reconstruction code's output (plasma shape, current, q-profile, flux)
ayc — Core Thomson scattering data (Te, ne profile measurements)