"""
MAST pulse batching

Takes a shot id (or list of shot id's) plus every parameter that a decision is made for and runs tmtx loops
"""

# campaigns M5-M9, 2004-2013
import numpy as np
import requests
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import copy
import argparse
import functools
import hashlib
import inspect
import json
import multiprocessing
import time
import traceback
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from unittest import mock

BASE_URL = 'https://mastapp.site'
S3_URL = 'https://s3.echo.stfc.ac.uk'
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MAST_info', 'fairmast_cache')

apiReachable = True

# Pulse database, runs.csv gets one row per run, pending holds per-run rows until a batch merges them
databaseDir = 'pulse_database'
runsPath = os.path.join(databaseDir, 'runs.csv')
pendingDir = os.path.join(databaseDir, 'pending')
dictionaryPath = os.path.join(databaseDir, 'data_dictionary.csv')

tokamaker_python_path = os.getenv('OFT_ROOTPATH')
if tokamaker_python_path is not None:
    sys.path.append(os.path.join(tokamaker_python_path, 'python'))
import OpenFUSIONToolkit
from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
from OpenFUSIONToolkit.TokaMaker.pulse_design import TokaMaker_TORAX, BASE_CONFIG
import torax

# Turbulent transport surrogates
turbulenceConfigs = {
    'qlknn': copy.deepcopy(BASE_CONFIG['transport']),
    'tglfnn': {**{k: v for k, v in BASE_CONFIG['transport'].items() if k not in ('include_ITG', 'include_TEM', 'include_ETG', 'avoid_big_negative_s')}, 'model_name': 'tglfnn-ukaea', 'machine': 'multimachine'},
}

def getOftEnv(nthreads=2):
    """
    OFT_env only allows one instance per kernel, reuse it if this kernel already made one
    """
    return OFT_env.instance if hasattr(OFT_env, 'instance') else OFT_env(nthreads=nthreads)

def openShotGroup(shotId, group, level=1):
    """
    Opens a diagnostic group (efm, ayc, anb, summary, etc) for a shot,
    download of pulse only happen upon a read
    """
    global apiReachable
    # Open cached shots from disk, a group missing from the cache is missing from the shot
    if os.path.exists(os.path.join(CACHE_DIR, 'complete', str(shotId))):
        return xr.open_zarr(os.path.join(CACHE_DIR, f'level{level}', f'{shotId}.zarr'), group=group)
    # Fixed S3 path pattern the API returns for every shot, used whenever the API does not answer
    shotUrl = f'{S3_URL}/mast/level{level}/shots/{shotId}.zarr'
    if apiReachable:
        path = f'json/shots/{shotId}' if level == 1 else f'json/level2/shots/{shotId}'
        try:
            response = requests.get(f'{BASE_URL}/{path}', timeout=5)
            response.raise_for_status()
            shot = response.json()
            shotUrl = shot['url'].replace('s3:/', shot['endpoint_url'])
        except (requests.ConnectionError, requests.Timeout) as e:
            apiReachable = False
            print(f'FAIR-MAST API unreachable ({type(e).__name__}), opening shots straight from S3 for the rest of this process')
        except requests.RequestException as e:
            raise ConnectionError(f'failed to fetch shot {shotId}: {e}') from e
    return xr.open_zarr(shotUrl, group=group)

def openThomsonGroup(shotId, level=1):
    """
    Thomson group for a shot, ayc in later campaigns and atm in earlier ones
    """
    try:
        return openShotGroup(shotId, 'ayc', level=level)
    except KeyError:
        try:
            return openShotGroup(shotId, 'atm', level=level)
        except KeyError:
            raise KeyError(f'shot {shotId} has no Thomson group under ayc or atm')

##### Geometry and profile helpers #####

def buildGeqdsk(eq):
    """
    Builds a dict matching read_eqdsk()'s keys from an efm equilibrium slice.
    """
    validCols = ~np.isnan(eq['psirz'].values).any(axis=0)
    gridR, gridZ = eq['gridr'].values, eq['gridz'].values
    lcfsR, lcfsZ = eq['lcfs_r'].values, eq['lcfs_z'].values
    lcfsValid = ~np.isnan(lcfsR) & ~np.isnan(lcfsZ)
    return {
        'nr': len(gridR),
        'nz': len(gridZ),
        'rdim': float(gridR.max() - gridR.min()),
        'zdim': float(gridZ.max() - gridZ.min()),
        'rleft': float(gridR.min()),
        'zmid': float((gridZ.max() + gridZ.min()) / 2.0),
        'rcentr': float(eq['rvtor']),
        'bcentr': float(eq['bvac_val']),
        'raxis': float(eq['magnetic_axis_r']),
        'zaxis': float(eq['magnetic_axis_z']),
        'psimag': float(eq['psi_axis']),
        'psibry': float(eq['psi_boundary']),
        'ip': float(eq['plasma_current_c']),
        'fpol': eq['fpsi_c'].values,
        'pres': eq['ppsi_c'].values,
        'ffprim': eq['ffprime'].values,
        'pprime': eq['pprime'].values,
        'psirz': eq['psirz'].values[:, validCols],
        'qpsi': eq['qpsi_c'].values,
        'rzout': np.column_stack([lcfsR[lcfsValid], lcfsZ[lcfsValid]]),
        'rzlim': np.column_stack([eq['limiterr'].values, eq['limiterz'].values]),
    }

def psiNormAtMidplane(rQuery, eq):
    """
    Maps real-space midplane R to normalized poloidal flux using the cleaned psirz grid
    """
    validCols = ~np.isnan(eq['psirz'].values).any(axis=0)
    gridR, gridZ = eq['gridr'].values, eq['gridz'].values
    psirzClean = eq['psirz'].values[:, validCols]
    interp = RegularGridInterpolator((gridZ, gridR), psirzClean, bounds_error=False, fill_value=None)
    # Thomson chords at Z=0
    psi = interp(np.column_stack([np.zeros_like(rQuery), rQuery]))
    return (psi - float(eq['psi_axis'])) / (float(eq['psi_boundary']) - float(eq['psi_axis']))

def rhoTorNormFromPsiNorm(psiNormQuery, eq, lastSurface=1.0):
    """
    Converts normalized poloidal flux to normalized toroidal flux
    """
    psiNorm65 = eq['psi_norm'].values
    q = eq['qpsi_c'].values
    torFlux = np.concatenate([[0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(psiNorm65))])
    torFluxNorm = torFlux / np.interp(lastSurface, psiNorm65, torFlux)
    return np.interp(np.clip(psiNormQuery, 0, 1), psiNorm65, torFluxNorm) ** 0.5

def mapThomsonToRho(shotId, refTime, eq, lastSurface=1.0):
    """
    Throw out broken/NaN channels, relabel measurements by flux surface in format TORAX likes
    """
    thomson = openThomsonGroup(shotId, level=1).sel(time=refTime, method='nearest')
    r, te, ne = thomson['radius'].values, thomson['te'].values, thomson['ne'].values
    valid = ~np.isnan(r) & ~np.isnan(te) & ~np.isnan(ne)
    rho = rhoTorNormFromPsiNorm(psiNormAtMidplane(r[valid], eq), eq, lastSurface)
    return rho, te[valid], ne[valid]

@functools.lru_cache(maxsize=2)
def loadCxrs(shotId):
    """
    CXRS radius, Ti and Ti error for the whole shot. None if the shot has no act group or no Ti
    """
    try:
        act = openShotGroup(shotId, 'act', level=1)
    except KeyError:
        return None
    if act is None or 'pla_temperature' not in act.data_vars or 'pla_temperature_error' not in act.data_vars:
        return None
    return act[['majorradius', 'pla_temperature', 'pla_temperature_error']].sortby('time').load()

def mapCxrsToRho(shotId, refTime, eq, lastSurface=1.0):
    """
    CXRS isnt on every shot even when the beam is, not every era has the act group
    Fall back to Ti = Te
    """
    act = loadCxrs(shotId)
    if act is None:
        return None
    cxrs = act.sel(time=refTime, method='nearest')
    r, ti, tiError = cxrs['majorradius'].values, cxrs['pla_temperature'].values, cxrs['pla_temperature_error'].values
    valid = ~np.isnan(r) & ~np.isnan(ti) & (ti > 0) & (ti < 5000) & (tiError > 0) & (tiError < 0.3 * ti)
    if not np.any(valid):
        return None
    rho = rhoTorNormFromPsiNorm(psiNormAtMidplane(r[valid], eq), eq, lastSurface)
    # CXRS only measures with a beam on, before that there are too few good channels inside the plasma
    if np.sum(rho < 1.0) < 3:
        return None
    return rho, ti[valid]

def buildNodeProfile(rho, values, rhoNodeCount=10, medianHalfWidth=0.1):
    """
    Running median onto a fixed rho grid, shared by every kinetic profile (Te, ne, Ti)
    smooths noisy scattered measurements onto a clean grid
    """
    sortOrder = np.argsort(rho)
    rhoSorted, valSorted = rho[sortOrder], values[sortOrder]
    rhoNodes = np.linspace(0.0, 1.0, rhoNodeCount)
    nodes = np.array([np.median(valSorted[np.abs(rhoSorted - x) <= medianHalfWidth]) if np.any(np.abs(rhoSorted - x) <= medianHalfWidth) else np.nan for x in rhoNodes])
    covered = ~np.isnan(nodes)
    return rhoNodes[covered], nodes[covered]

def gasPuffRate(shotId, coarseDt=0.01):
    """
    Need a particle rate for set_fueling. gas_injection only gives a cumulative
    injected count, resample onto a coarser time grid before differentiating.
    turns a running total into a rate
    """
    ds = openShotGroup(shotId, 'gas_injection', level=2)
    try:
        t, total = ds['time'].values, ds['total_injected'].values
    except KeyError:
        raise KeyError(f'shot {shotId} gas_injection has no total_injected, only raw valve signals')
    tCoarse = np.arange(t.min(), t.max(), coarseDt) # resampled to 10 ms
    totalCoarse = np.interp(tCoarse, t, total)
    rateCoarse = np.clip(np.gradient(totalCoarse, tCoarse), 0, None)
    return dict(zip(tCoarse.tolist(), rateCoarse.tolist()))

channelToCoilSet = {
    'SOL': 'SOL', 'P2IL FEED': 'P2_INNER_LOWER', 'P2IU FEED': 'P2_INNER_UPPER',
    'P2OL FEED': 'P2_OUTER_LOWER', 'P2OU FEED': 'P2_OUTER_UPPER',
    'P3L FEED': 'P3_LOWER', 'P3U FEED': 'P3_UPPER',
    'P4L FEED': 'P4_LOWER', 'P4U FEED': 'P4_UPPER',
    'P5L FEED': 'P5_LOWER', 'P5U FEED': 'P5_UPPER',
    'P6L': 'P6_LOWER', 'P6U': 'P6_UPPER',
}

##### Shot window #####

def thomsonChannelsInside(ayc, efm, t, psiLimit=0.85):
    """
    Good Thomson channels inside psiLimit at one Thomson time, mapped with the EFIT slice nearest that time
    """
    thomsonNow = ayc.sel(time=t)
    r, te, ne = np.ravel(thomsonNow['radius'].values), thomsonNow['te'].values, thomsonNow['ne'].values
    good = np.isfinite(r) & np.isfinite(te) & np.isfinite(ne) & (te > 0) & (ne > 0)
    return int(np.sum(psiNormAtMidplane(r[good], efm.sel(time=t, method='nearest')) < psiLimit))

# Keep the last two shots in memory
@functools.lru_cache(maxsize=2)
def loadShotWindow(shotId, minChannels=5, maxStartDelay=0.05):
    """
    Loads EFIT and Thomson for a shot and picks the simulated window, the longest stretch of reliable EFIT slices,
    starting at the first Thomson time in it with at least minChannels good channels inside the plasma
    """
    efmGroup = openShotGroup(shotId, 'efm', level=1)
    if efmGroup is None:
        raise KeyError(f'shot {shotId} efm fetch failed, see the network error printed above')
    try:
        efm = efmGroup.load()
    except KeyError:
        raise KeyError(f'shot {shotId} has no efm group at all')
    if 'plasma_current_c' not in efm.data_vars:
        efm['plasma_current_c'] = efm['plasma_current_x']
    validTimes = efm['time'].values[~np.isnan(efm['plasma_current_c'].values)]

    # EFIT reconstruction only between startup and ramp down, poloidal beta < 0 and li > 2 when plasma is small
    try:
        efmReliable = (efm['betap'].sel(time=validTimes).values > 0) & (efm['li'].sel(time=validTimes).values < 2.0)
    except KeyError:
        raise KeyError(f'shot {shotId} efm has no derived equilibrium (betap/li missing), EFIT reconstruction looks incomplete')

    # Longest stretch of reliable slices
    reliableTimes = validTimes[efmReliable]
    reliableRuns = np.split(reliableTimes, np.where(np.diff(reliableTimes) > 2.5 * np.median(np.diff(validTimes)))[0] + 1)
    pulseWindow = max(reliableRuns, key=lambda run: run[-1] - run[0])
    tPulseStart = float(pulseWindow[0])
    tPulseEnd = float(pulseWindow[-1])
    refTime = tPulseStart
    equilibrium = efm.sel(time=refTime, method='nearest')

    ayc = openThomsonGroup(shotId, level=1)[['te', 'ne', 'radius']].load()
    teValidTimeMask = ~np.isnan(ayc['te'].values).all(axis=1)
    thomsonValidTimes = ayc['time'].values[teValidTimeMask]

    # Start the simulation at the first Thomson time with a usable profile, the Thomson time nearest the EFIT
    # window start can come before it or before the plasma has formed
    startCandidates = thomsonValidTimes[(thomsonValidTimes >= tPulseStart) & (thomsonValidTimes <= tPulseStart + maxStartDelay)]
    try:
        thomsonRefTime = next(float(t) for t in startCandidates if thomsonChannelsInside(ayc, efm, t) >= minChannels)
    except StopIteration:
        raise ValueError(f'shot {shotId} has no Thomson time within {maxStartDelay * 1000:.0f} ms of the window start with {minChannels} good channels inside the plasma')
    tPulseStart = thomsonRefTime
    refTime = tPulseStart
    equilibrium = efm.sel(time=refTime, method='nearest')

    return efm, validTimes, tPulseStart, tPulseEnd, refTime, equilibrium, ayc, thomsonValidTimes, thomsonRefTime

def loadBeamPower(shotId, tPulseStart, tPulseEnd, powerThresholdMw=0.01):
    """
    NBI power trajectory. None for Ohmic shots or shots with no beam diagnostic. tot_sum_power 
    sums SS and SW so this check covers every supported heating type (SS Beam, SW Beam, 2 Beams). 
    Threshold sits above noise floor on confirmed Ohmic shots (sub 1 kW) and below real beam 
    power (100s of kW+).
    """
    try:
        anb = openShotGroup(shotId, 'anb', level=1)
    except KeyError:
        return None
    if anb is None or 'tot_sum_power' not in anb.data_vars:
        return None
    power = anb['tot_sum_power'].sel(time=slice(tPulseStart, tPulseEnd))
    if np.nanmax(power.values) < powerThresholdMw:
        return None
    return {float(t): float(p) * 1e6 for t, p in zip(power['time'].values, power.values) if not np.isnan(p)}

def loadBeamFueling(shotId, tPulseStart, tPulseEnd, beamEnergyKev=75.0):
    """
    NBI particle source in atoms/s. Full, half, and third energy atoms carry E, E/2, and E/3
    """
    anb = openShotGroup(shotId, 'anb', level=1).sel(time=slice(tPulseStart, tPulseEnd))
    # Energy of one full energy atom in J
    atomEnergy = beamEnergyKev * 1e3 * 1.602e-19
    rate = xr.zeros_like(anb['tot_sum_power'])
    for beam in ('ss', 'sw'):
        for fraction, energyShare in (('full', 1.0), ('half', 1 / 2), ('third', 1 / 3)):
            try:
                rate = rate + anb[f'{beam}_{fraction}_power'].fillna(0.0) * 1e6 / (energyShare * atomEnergy)
            except KeyError:
                continue
    # Clip the small negative readings while beams are off
    rate = rate.clip(min=0.0)
    return {float(t): float(s) for t, s in zip(rate['time'].values, rate.values) if not np.isnan(s)}

def lineAverageDensity(ayc, efm, tStart, tEnd, minChannels=3):
    """
    Thomson ne averaged along the midplane chord inside the LCFS at each Thomson time from tStart to tEnd, in 1e19 m^-3
    """
    times, density = [], []
    validTimes = np.flatnonzero(~np.isnan(ayc['ne'].values).all(axis=1))
    for i in validTimes:
        thomsonNow = ayc.isel(time=i)
        t = float(thomsonNow['time'])
        if t < tStart or t > tEnd:
            continue
        r, ne = np.ravel(thomsonNow['radius'].values), thomsonNow['ne'].values
        good = np.isfinite(r) & np.isfinite(ne) & (ne > 0)
        try:
            inside = psiNormAtMidplane(r[good], efm.sel(time=t, method='nearest')) < 1.0
        except ValueError:
            continue
        order = np.argsort(r[good][inside])
        rInside, neInside = r[good][inside][order], ne[good][inside][order]
        if len(rInside) >= minChannels:
            times.append(t)
            density.append(np.trapezoid(neInside, rInside) / (rInside[-1] - rInside[0]) / 1e19)
    return np.array(times), np.array(density)

def beamAbsorbedFraction(beamPower, efm, ayc, heatingCoefficient=0.66, bounds=(0.2, 0.8)):
    """
    Absorbed fraction of injected NBI power at each beam power time, heatingCoefficient MW of beam heating per 1e19 m^-3
    of line averaged density over the injected power in MW, clipped to bounds
    """
    times = np.array(list(beamPower))
    injected = np.array(list(beamPower.values())) / 1e6
    densityTimes, density = lineAverageDensity(ayc, efm, times[0], times[-1])
    try:
        density = np.interp(times, densityTimes, density)
    except ValueError:
        raise ValueError('no Thomson time with enough channels inside the LCFS for the line averaged density')
    fraction = heatingCoefficient * density / np.maximum(injected, 1e-6)
    return times, np.clip(fraction, *bounds)

##### Seed equilibria #####

def buildSeedEquilibria(mygs, shotId, efm, seedTimes, channelToCoilSet, coilBoundAmps, fluxWeightCandidates=(10.0, 1.0, 0.1)):
    """
    One TokaMaker seed equilibrium per seed time from the EFIT boundary, current and PF coil currents, saved as gEQDSK files for TORAX
    """
    mygs.set_coil_bounds({name: [-coilBoundAmps, coilBoundAmps] for name in mygs.coil_sets})  # Loose bounds, not real coil limits

    # Generate and save one seed eqdsk per seed time
    seedFiles = []
    os.makedirs('seeds', exist_ok=True)

    # Same PF current record for every seed time, get once
    pfActiveCurrents = openShotGroup(shotId, 'pf_active', level=2)

    for i, seedTime in enumerate(seedTimes):
        seedEq = efm.sel(time=seedTime, method='nearest')
        seedGeqdsk = buildGeqdsk(seedEq)

        regTerms = []
        for channelName, coilSetName in channelToCoilSet.items():
            realCurrentAmps = float(pfActiveCurrents['coil_current'].sel(current_channel=channelName).sel(time=seedTime, method='nearest'))
            netTurns = mygs.coil_sets[coilSetName]['net_turns']
            targetPerTurn = 0.5 * realCurrentAmps if coilSetName == 'SOL' else realCurrentAmps / netTurns
            regTerms.append(mygs.coil_reg_term({coilSetName: 1.0}, target=targetPerTurn, weight=1.0e-2))
        mygs.set_coil_reg(reg_terms=regTerms)

        mygs.set_targets(Ip=abs(seedGeqdsk['ip']), pax=abs(seedGeqdsk['pres'][0]))

        rzout = seedGeqdsk['rzout']
        r0 = float(seedEq['magnetic_axis_r'])
        z0 = float(seedEq['magnetic_axis_z'])
        a = float((rzout[:, 0].max() - rzout[:, 0].min()) / 2.0)
        kappa = float((rzout[:, 1].max() - rzout[:, 1].min()) / (2.0 * a))

        # A plasma this early has almost no flux depth between axis and boundary
        for attempt, weight in enumerate(fluxWeightCandidates):
            # Set boundary flux to EFIT value so seed has physical absolute flux
            mygs.set_psi_constraints(rzout, np.full(len(rzout), float(seedEq['psi_boundary'])), weights=np.full(len(rzout), weight))
            mygs.init_psi(r0=r0, z0=z0, a=a, kappa=kappa, delta=0.0)
            try:
                mygs.solve()
                break
            except Exception as e:
                if attempt == len(fluxWeightCandidates) - 1:
                    raise
                print(f'seed {i} failed at flux weight={weight}, retrying at {fluxWeightCandidates[attempt + 1]}: {e}')

        seedPath = f'seeds/seed_{i}.eqdsk'
        mygs.save_eqdsk(seedPath, cocos=2)  # save_eqdsk defaults to cocos=7, TokaMaker_TORAX assumes cocos=2 internally
        seedFiles.append(seedPath)

    return seedFiles

def pickLastSurfaceFactor(seedFiles, candidates=(0.9,)):
    """
    TORAX cant build flux surface geometry to separatrix of diverted plasmas, use largest last surface factor it accepts
    """
    for factor in candidates:
        try:
            for seedFile in seedFiles:
                geometryCheck = copy.deepcopy(BASE_CONFIG)
                geometryCheck['geometry'] = {'geometry_type': 'eqdsk', 'geometry_directory': os.getcwd(), 'last_surface_factor': factor, 'Ip_from_parameters': False, 'geometry_file': seedFile, 'cocos': 2}
                torax.ToraxConfig.from_dict(geometryCheck)
            return factor
        except Exception:
            continue
    raise ValueError('TORAX rejects the seed equilibria at every last surface factor tried')

##### Initial profiles and edge conditions #####

def measuredEdgeConditions(shotId, lastSurface, tPulseStart, tPulseEnd, fitHalfWidth=0.08, smoothSamples=5):
    """
    Measured edge Te and ne for TORAX's boundary. At each Thomson time, take the Thomson points whose psiN is within
    fitHalfWidth of the boundary (both sides of the plasma), fit a straight line of Te against psiN and of ne against
    psiN, and read both lines at the boundary psiN. Those readings are then smoothed in time with a running median over
    smoothSamples readings. Also returns the raw readings and the points behind them for the edge check figure
    """
    efm, _, _, _, _, _, ayc, thomsonValidTimes, _ = loadShotWindow(shotId)
    times, teEdge, neEdge = [], [], []
    points = {'pointTime': [], 'pointPsiN': [], 'pointTe': [], 'pointNe': []}
    for t in thomsonValidTimes[(thomsonValidTimes >= tPulseStart) & (thomsonValidTimes <= tPulseEnd)]:
        thomsonNow = ayc.sel(time=t)
        r, te, ne = np.ravel(thomsonNow['radius'].values), thomsonNow['te'].values, thomsonNow['ne'].values
        valid = np.isfinite(r) & np.isfinite(te) & np.isfinite(ne) & (te > 0) & (ne > 0)
        psiN = psiNormAtMidplane(r[valid], efm.sel(time=t, method='nearest'))
        near = np.abs(psiN - lastSurface) <= fitHalfWidth
        if np.sum(near) < 3:
            continue
        times.append(float(t))
        # Thomson points behind each reading, kept for the edge check figure
        points['pointTime'] += [float(t)] * int(np.sum(near))
        points['pointPsiN'] += list(psiN[near])
        points['pointTe'] += list(te[valid][near] / 1000.0)
        points['pointNe'] += list(ne[valid][near])
        teEdge.append(np.polyval(np.polyfit(psiN[near], te[valid][near] / 1000.0, 1), lastSurface))
        neEdge.append(np.polyval(np.polyfit(psiN[near], ne[valid][near], 1), lastSurface))
    if len(times) < smoothSamples:
        raise ValueError(f'shot {shotId} has Thomson near psiN {lastSurface} at only {len(times)} times, too few for a measured edge')
    # Running median in time removes noisy fits, floor keep boundary positive 
    teSmooth = pd.Series(teEdge).rolling(smoothSamples, center=True, min_periods=1).median().clip(lower=0.005)
    neSmooth = pd.Series(neEdge).rolling(smoothSamples, center=True, min_periods=1).median().clip(lower=1e17)
    detail = {'fitTime': times, 'teFit': teEdge, 'neFit': neEdge, **points}
    return {t: float(value) for t, value in zip(times, teSmooth)}, {t: float(value) for t, value in zip(times, neSmooth)}, detail

def buildInitialProfiles(shotId, thomsonRefTime, equilibrium, lastSurface, efm, validTimes, tPulseStart, tPulseEnd,
                          edgeTe, edgeDensityFraction, gasPuffCoarseDt=0.01, rhoNodeCount=10, medianHalfWidth=0.1, edgeSource='measured'):
    """
    Starting Te, ne and Ti profiles for TORAX plus the edge boundary conditions and gas puff rate
    """
    # Initial kinetic profiles, Te/ne from the first Thomson slice in the window
    rhoInit, teInit, neInit = mapThomsonToRho(shotId, thomsonRefTime, equilibrium, lastSurface)

    gasPuffSTotal = gasPuffRate(shotId, coarseDt=gasPuffCoarseDt)

    # Edge conditions measured from Thomson at the TORAX boundary, or from constants with density a fixed
    # fraction of Greenwald density from Ip and EFIT minor radius
    if edgeSource == 'measured':
        edgeTeBc, edgeNeBc, edgeDetail = measuredEdgeConditions(shotId, lastSurface, tPulseStart, tPulseEnd)
    elif edgeSource == 'constants':
        tEfm = validTimes[(validTimes >= tPulseStart) & (validTimes <= tPulseEnd)]
        aMinor = np.array([(np.nanmax(efm.sel(time=t)['lcfs_r'].values) - np.nanmin(efm.sel(time=t)['lcfs_r'].values)) / 2.0 for t in tEfm])
        ipEfm = np.abs(efm['plasma_current_c'].sel(time=tEfm).values)
        greenwaldDensity = ipEfm / 1e6 / (np.pi * aMinor ** 2) * 1e20
        edgeTeBc = {float(t): edgeTe for t in tEfm}
        edgeNeBc = {float(t): float(edgeDensityFraction * n) for t, n in zip(tEfm, greenwaldDensity)}
        edgeDetail = {}
    else:
        raise ValueError(f"edgeSource must be 'measured' or 'constants', got {edgeSource}")

    # Drop channels outside the last surface where rho > 1, convert Te to keV
    insidePlasma = rhoInit < 1.0
    rhoNodes, teNodes = buildNodeProfile(rhoInit[insidePlasma], teInit[insidePlasma] / 1000.0, rhoNodeCount, medianHalfWidth)
    _, neNodes = buildNodeProfile(rhoInit[insidePlasma], neInit[insidePlasma], rhoNodeCount, medianHalfWidth)
    neProfile = {0.0: {float(r): float(n) for r, n in zip(rhoNodes, neNodes)}}
    teProfile = {0.0: {float(r): float(t) for r, t in zip(rhoNodes, teNodes)}}

    # CXRS Ti where the shot and era have it, Te elsewhere, same as the edge value below
    cxrs = mapCxrsToRho(shotId, thomsonRefTime, equilibrium, lastSurface)
    if cxrs is not None:
        rhoTi, tiInit = cxrs
        insideTi = rhoTi < 1.0
        rhoTiNodes, tiNodes = buildNodeProfile(rhoTi[insideTi], tiInit[insideTi] / 1000.0, rhoNodeCount, medianHalfWidth)
        tiProfile = {0.0: {float(r): float(t) for r, t in zip(rhoTiNodes, tiNodes)}}
    else:
        tiProfile = teProfile

    return neProfile, teProfile, tiProfile, edgeNeBc, edgeTeBc, gasPuffSTotal, edgeDetail

##### Compare to experiment #####

def compareToExperiment(tmtx, efm, ayc, validTimes, thomsonValidTimes, tPulseStart, tPulseEnd, lastSurface, verbose=True):
    """
    Thomson vs TORAX Te and ne at the innermost Thomson channel for every Thomson time in the window, with the RMS difference
    """
    # Experimental timestamps inside simulated window
    timeSources = {
        'EFIT (efm)': validTimes,
        'core Thomson (ayc)': thomsonValidTimes,
    }
    if verbose:
        for label, times in timeSources.items():
            inWindow = times[(times >= tPulseStart) & (times <= tPulseEnd)]
            print(f'{label:20s} {len(inWindow):5d} timestamps, median step {1e3 * np.median(np.diff(inWindow)):.2f} ms')
        txTimes = tmtx._data_tree.profiles.T_e.coords['time'].values
        print(f'{"TORAX output":20s} {len(txTimes):5d} timestamps')
        print(f'{"TokaMaker solves":20s} {len(tmtx.results["psi_lcfs_tm"]["x"]):5d} timestamps')

    # Core Thomson Te/ne at every measurement time from channel nearest to magnetic axis inside the LCFS
    txTe = tmtx._data_tree.profiles.T_e
    txNe = tmtx._data_tree.profiles.n_e

    compareTimes = thomsonValidTimes[(thomsonValidTimes >= tPulseStart) & (thomsonValidTimes <= tPulseEnd)]
    usedTimes, coreRho, teMeasured, teSimulated, neMeasured, neSimulated = [], [], [], [], [], []
    for t in compareTimes:
        eq = efm.sel(time=t, method='nearest')
        thomsonNow = ayc.sel(time=t)
        r, te, ne = thomsonNow['radius'].values, thomsonNow['te'].values, thomsonNow['ne'].values
        valid = ~np.isnan(r) & ~np.isnan(te) & ~np.isnan(ne)
        rho = rhoTorNormFromPsiNorm(psiNormAtMidplane(r[valid], eq), eq, lastSurface)
        try:
            inside = np.where(rho < 1.0)[0]
            core = inside[np.argmin(rho[inside])]
        except ValueError:
            continue
        usedTimes.append(float(t))
        coreRho.append(rho[core])
        teMeasured.append(te[valid][core] / 1000.0)
        neMeasured.append(ne[valid][core])
        # TORAX output at same time and rho interpolated from output grid
        teSimulated.append(float(txTe.interp(time=t, rho_norm=rho[core])))
        neSimulated.append(float(txNe.interp(time=t, rho_norm=rho[core])))

    usedTimes, coreRho = np.array(usedTimes), np.array(coreRho)
    teMeasured, teSimulated = np.array(teMeasured), np.array(teSimulated)
    neMeasured, neSimulated = np.array(neMeasured), np.array(neSimulated)

    if verbose:
        print('time [s]   rho    measured Te [keV]   simulated Te [keV]')
        for k in range(0, len(usedTimes), max(1, len(usedTimes) // 10)):
            print(f'{usedTimes[k]:7.3f}  {coreRho[k]:5.2f}  {teMeasured[k]:12.3f}  {teSimulated[k]:15.3f}')

    teRms = float(np.sqrt(np.mean((teMeasured - teSimulated) ** 2))) if len(teMeasured) else float('nan')
    neRms = float(np.sqrt(np.mean((neMeasured - neSimulated) ** 2))) if len(neMeasured) else float('nan')

    return {
        'times': usedTimes, 'coreRho': coreRho,
        'teMeasured': teMeasured, 'teSimulated': teSimulated, 'teRms': teRms,
        'neMeasured': neMeasured, 'neSimulated': neSimulated, 'neRms': neRms,
    }

def coreEdgeVolAvg(tmtx, fieldName, volAvgField, t, edgeRho=0.98):
    """
    Simulated core (rho=0), edge (rho=edgeRho), and TORAX's volume average
    """
    field = getattr(tmtx._data_tree.profiles, fieldName)
    core = float(field.interp(time=t, rho_norm=0.0))
    edge = float(field.interp(time=t, rho_norm=edgeRho))
    volAvg = tmtx._extract_tx_scalar(tmtx._data_tree, volAvgField, t)
    return core, volAvg, edge

def measuredCoreEdgeVolAvg(rho, values, tmtx, t, edgeRho=0.98):
    """
    Core/edge from the smoothed node profile, volume average weighted by TORAX's dV/drho
    Measured and simulated averages use the same flux surface volumes
    """
    rhoNodes, valNodes = buildNodeProfile(rho, values, rhoNodeCount=10, medianHalfWidth=0.1)
    if len(rhoNodes) < 2:
        return None, None, None
    core = float(valNodes[0])
    edge = float(np.interp(edgeRho, rhoNodes, valNodes))
    vpr = tmtx._data_tree.profiles.vpr.interp(time=t)
    vprRho = vpr['rho_norm'].values
    volAvg = float(np.trapezoid(np.interp(vprRho, rhoNodes, valNodes) * vpr.values, vprRho) / np.trapezoid(vpr.values, vprRho))
    return core, volAvg, edge

##### Run one pulse #####

def runMastPulse(shotId, meshPath='MAST_info/MAST_mesh.h5', outputDir='pulse_outputs', runName=None, batchName=None,
                  gasFuelingEfficiency=0.02, edgeTe=0.049, edgeDensityFraction=0.12, edgeSource='measured', Zeff=1.7,
                  nSeeds=5, order=2, maxits=100, coilBoundAmps=1.0e7, txDt=0.005, tmNPoints=5,
                  lastSurfaceFactorCandidates=(0.9,), fluxWeightCandidates=(10.0, 1.0, 0.1),
                  bootstrapModel='sauter', transportModel='angioni_sauter', turbulenceModel='tglfnn',
                  sawtoothSCritical=1.0, maxLoop=2, outputMode='minimal', gasPuffCoarseDt=0.01,
                  rhoNodeCount=10, medianHalfWidth=0.1, beamPowerThresholdMw=0.01, beamHeatingCoefficient=0.66, beamAbsorptionBounds=(0.2, 0.8),
                  beamHeatLoc=0.0, beamHeatWidth=0.25, beamEnergyKev=75.0, oftEnv=None, verbose=True):
    """
    Runs one MAST shot with tmtx. Category 1 inputs (Ip, LCFS, profiles, gas puff, coil currents, 
    NBI power, CXRS Ti) come straight from the shot. Heating branch (Ohmic vs SS/SW/2 Beams) auto picked 
    from anb, doesnt depend on data_cleaning.ipynb having been run first. oftEnv lets
    runMastPulses share one OFT_env across a batch, since only one can exist per kernel.
    """
    inputConstants = {
        'gasFuelingEfficiency': gasFuelingEfficiency,
        'edgeTe': edgeTe,
        'edgeDensityFraction': edgeDensityFraction,
        'Zeff': Zeff,
    }
    runName = runName or f'mast_pulse_{shotId}'

    # Turbulent transport surrogate goes into BASE_CONFIG before anything copies it, each worker process has its own BASE_CONFIG
    try:
        BASE_CONFIG['transport'] = copy.deepcopy(turbulenceConfigs[turbulenceModel])
    except KeyError:
        raise ValueError(f'turbulenceModel must be one of {list(turbulenceConfigs)}, got {turbulenceModel}')

    originalCwd = os.getcwd()
    meshPathAbs = os.path.abspath(meshPath)

    myOFT = oftEnv if oftEnv is not None else getOftEnv()
    mygs = TokaMaker(myOFT)
    mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict = load_gs_mesh(meshPathAbs)
    mygs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
    mygs.setup_regions(cond_dict=cond_dict, coil_dict=coil_dict)

    efm, validTimes, tPulseStart, tPulseEnd, refTime, equilibrium, ayc, thomsonValidTimes, thomsonRefTime = loadShotWindow(shotId)
    beamPower = loadBeamPower(shotId, tPulseStart, tPulseEnd, beamPowerThresholdMw)
    beamFueling = loadBeamFueling(shotId, tPulseStart, tPulseEnd, beamEnergyKev) if beamPower is not None else None
    beamAbsorbed = beamAbsorbedFraction(beamPower, efm, ayc, beamHeatingCoefficient, beamAbsorptionBounds) if beamPower is not None else None

    F0 = float(equilibrium['rvtor'] * equilibrium['bvac_val'])
    mygs.setup(order=order, F0=F0)
    # Max solver iterations per equilibrium
    mygs.settings.maxits = maxits

    # Every run lands in a batch folder, a single run gets one named for its date and time
    batchName = batchName or makeBatchName(f'single_{shotId}')

    # seeds/outputs
    runDir = os.path.join(originalCwd, outputDir, batchName, f'shot_{shotId}')
    os.makedirs(os.path.join(runDir, 'seeds'), exist_ok=True)
    os.chdir(runDir)
    try:
        # Seed equilibria, evenly spaced through window
        seedTimes = [float(t) for t in np.linspace(tPulseStart, tPulseEnd, nSeeds)]
        seedFiles = buildSeedEquilibria(mygs, shotId, efm, seedTimes, channelToCoilSet, coilBoundAmps, fluxWeightCandidates)

        lastSurface = pickLastSurfaceFactor(seedFiles, lastSurfaceFactorCandidates)

        def attemptFly(lastSurface):
            """
            Builds tmtx at one last surface factor and runs the coupling loop
            """
            neProfile, teProfile, tiProfile, edgeNeBc, edgeTeBc, gasPuffSTotal, edgeDetail = buildInitialProfiles(
                shotId, thomsonRefTime, equilibrium, lastSurface, efm, validTimes, tPulseStart, tPulseEnd,
                inputConstants['edgeTe'], inputConstants['edgeDensityFraction'], gasPuffCoarseDt, rhoNodeCount, medianHalfWidth, edgeSource=edgeSource)

            # Build the coupled object
            tmtx = TokaMaker_TORAX(
                t_init=tPulseStart,
                t_final=tPulseEnd,
                eqtimes=seedTimes,
                g_eqdsk_arr=seedFiles,
                tokamaker_obj=mygs,
                tx_dt=txDt,  # MAST pulses run ~0.5s
                tm_times=list(np.linspace(tPulseStart, tPulseEnd, tmNPoints)),
                last_surface_factor=lastSurface,
            )

            # Use the Sauter bootstrap model and low sensitivity sawtooth trigger
            tmtx.load_TORAX_config({
                'neoclassical': {'bootstrap_current': {'model_name': bootstrapModel}, 'transport': {'model_name': transportModel}},
                'mhd': {'sawtooth': {'trigger_model': {'model_name': 'simple', 's_critical': sawtoothSCritical}}},
            })

            # Generic heat stands in for NBI, only set when anb shows real beam power in window
            if beamPower is not None:
                tmtx.set_heating(generic_heat={t: p * float(np.interp(t, *beamAbsorbed)) for t, p in beamPower.items()}, generic_heat_loc=beamHeatLoc, generic_heat_width=beamHeatWidth)

            # Ip and gas puff
            ipTrajectory = {float(t): abs(float(i)) for t, i in zip(efm['time'].values[~np.isnan(efm['plasma_current_c'].values)], efm['plasma_current_c'].values[~np.isnan(efm['plasma_current_c'].values)])}
            tmtx.set_Ip(ipTrajectory)

            # Gas puff scaled by fueling efficiency
            # Beam atoms ionize in core, absorbed atoms only
            beamParticles = {} if beamFueling is None else {'generic_particle_location': beamHeatLoc, 'generic_particle_width': beamHeatWidth, 'generic_particle_S_total': {t: s * float(np.interp(t, *beamAbsorbed)) for t, s in beamFueling.items()}}
            tmtx.set_fueling(gas_puff_S_total={t: inputConstants['gasFuelingEfficiency'] * rate for t, rate in gasPuffSTotal.items()}, gas_puff_decay_length=0.05, **beamParticles)

            tmtx.set_ne(neProfile, right_bc=edgeNeBc)
            tmtx.set_Te(teProfile, right_bc=edgeTeBc)

            # CXRS Ti from buildInitialProfiles, Te where CXRS is missing
            tmtx.set_Ti(tiProfile, right_bc=edgeTeBc)

            # carbon impurity from PFCs, Deuterium only
            tmtx.set_plasma_composition(main_ion={'D': 1.0}, Zeff=inputConstants['Zeff'], impurity='C')  # MAST is deuterium only, no tritium

            tmtx.set_evolve(density=True, Ti=True, Te=True, current=True)
            tmtx.set_TokaMaker_coil_reg()  # defaults to +-5 MA-turns per coil

            tmtx.fly(run_name=runName, output_mode=outputMode, max_loop=maxLoop)
            # Edge values TORAX was given, saved with the run for the edge check
            return tmtx, teProfile, tiProfile, {'teBc': edgeTeBc, 'neBc': edgeNeBc, **edgeDetail}

        # Heating moves boundary past seed-only check validation, fly() can fail tracing 
        # flux surfaces near the LCFS deep in the loop
        remainingFactors = [f for f in lastSurfaceFactorCandidates if f <= lastSurface]
        for attempt, factor in enumerate(remainingFactors):
            try:
                tmtx, teProfile, tiProfile, edgeConditions = attemptFly(factor)
                lastSurface = factor
                break
            except Exception as e:
                if attempt == len(remainingFactors) - 1:
                    raise
                print(f'fly() failed at last_surface_factor={factor}, retrying at {remainingFactors[attempt + 1]}: {e}')
       
        # TORAX config from the last loop with the seed geometry it read, so the shot can be rerun in TORAX alone
        # _get_tx_config hands back a built ToraxConfig, record the dict on its way into from_dict
        with mock.patch.object(torax.ToraxConfig, 'from_dict', wraps=torax.ToraxConfig.from_dict) as fromDict:
            tmtx._get_tx_config()
        toraxConfig = tmtx._numpy_to_plain_python(fromDict.call_args_list[-1].args[0])
        with open(os.path.join(runDir, 'torax_config.pkl'), 'wb') as configFile:
            pickle.dump(toraxConfig, configFile)

        if verbose:
            tmtx.summary()

        # maxLoop=0 never runs the coupling loop, so TORAX output isnt populated to compare against
        if maxLoop > 0:
            diagnostics = compareToExperiment(tmtx, efm, ayc, validTimes, thomsonValidTimes, tPulseStart, tPulseEnd, lastSurface, verbose=verbose)
        else:
            if verbose:
                print('maxLoop=0, no coupled TORAX output to compare, skipping Te/ne comparison')
            diagnostics = {'teRms': None, 'neRms': None}
        diagnostics['tPulseStart'] = tPulseStart
        diagnostics['tPulseEnd'] = tPulseEnd
        diagnostics['lastSurface'] = lastSurface
        diagnostics['beamHeating'] = beamPower is not None
        diagnostics['cxrsTiUsed'] = tiProfile is not teProfile
        diagnostics['edgeConditions'] = edgeConditions

        return tmtx, diagnostics
    finally:
        os.chdir(originalCwd)

##### Run a batch of pulses #####

def runMastPulses(shotIds, **kwargs):
    """
    Runs a list of shots one after another in this kernel, a failed shot prints its error and stores None
    """
    # OFT_env only allows one instance per kernel, reuse one if this kernel already made one
    myOFT = getOftEnv()
    # One batch folder for the whole list unless a batch name is passed in
    batchName = kwargs.pop('batchName', None) or makeBatchName(f'shotlist_{len(shotIds)}shots')
    results = {}
    for shotId in shotIds:
        print(f'--- shot {shotId} ---')
        try:
            results[shotId] = runMastPulse(shotId, oftEnv=myOFT, batchName=batchName, **kwargs)
        except Exception as e:
            print(f'shot {shotId} failed: {e}')
            results[shotId] = None
    return results

##### Batching and pulse database creation #####

def selectShots(shotIds=None, heating=None, campaigns=None, minCurrent=None, maxCurrent=None, sampleSize=None, seed=0,
                usablePath='MAST_info/usable_pulses.csv', screenedPath='MAST_info/screened_pulses.csv'):
    """
    Picks the shots for a batch, an explicit list or every usable shot matching the filters,
    and a short name for the selection that goes in the batch folder name
    """
    screened = pd.read_csv(screenedPath).set_index('shot_id')
    screenNotes = {}
    if shotIds is not None:
        for shotId in shotIds:
            try:
                screenRow = screened.loc[shotId]
                if not screenRow['usable']:
                    screenNotes[shotId] = screenRow['exclusionReasons']
                    print(f'warning: shot {shotId} is screened out ({screenRow["exclusionReasons"]}), running it anyway')
            except KeyError:
                screenNotes[shotId] = 'not in the screened catalog'
                print(f'warning: shot {shotId} is not in the screened catalog, running it anyway')
        return list(shotIds), screenNotes, f'shotlist_{len(shotIds)}shots'

    usable = pd.read_csv(usablePath)
    keep = pd.Series(True, index=usable.index)
    if heating is not None:
        keep &= usable['heatingResolved'].isin(heating)
    if campaigns is not None:
        keep &= usable['campaign'].isin(campaigns)
    if minCurrent is not None:
        keep &= usable['plasma_max_current'].abs() >= minCurrent
    if maxCurrent is not None:
        keep &= usable['plasma_max_current'].abs() <= maxCurrent
    selected = usable.loc[keep, 'shot_id']
    if sampleSize is not None and sampleSize < len(selected):
        selected = selected.sample(n=sampleSize, random_state=seed)
    print(f'{len(selected)} usable shots selected')

    # Short name for the batch folder from whichever filters were set
    nameParts = []
    if heating is not None:
        nameParts.append('-'.join(label.split(',')[0].replace(' ', '') for label in heating))
    if campaigns is not None:
        nameParts.append('-'.join(campaigns))
    if minCurrent is not None:
        nameParts.append(f'min{minCurrent:g}kA')
    if maxCurrent is not None:
        nameParts.append(f'max{maxCurrent:g}kA')
    if sampleSize is not None:
        nameParts.append(f'sample{sampleSize}')
    selectionName = '_'.join(nameParts) or 'allUsable'
    return sorted(int(shotId) for shotId in selected), screenNotes, selectionName

def makeBatchName(selectionName, label=None):
    """
    Batch folder and batchName column, date and time first so folders sort in the order they ran,
    then the label if one was given or the shot selection
    """
    return f'{datetime.now().strftime("%Y-%m-%d_%H%M")}_{label or selectionName}'

def runSettings(**kwargs):
    """
    Every runMastPulse input with its default filled in, minus the ones that only change where things go or what prints
    """
    bound = inspect.signature(runMastPulse).bind_partial(**kwargs)
    bound.apply_defaults()
    skip = ('shotId', 'oftEnv', 'verbose', 'outputDir', 'runName', 'batchName')
    return {name: value for name, value in bound.arguments.items() if name not in skip}

def configHash(settings):
    """
    Hash of the run settings
    """
    return hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()[:12]

def codeHash():
    """
    Fingerprint of given file
    """
    with open(os.path.abspath(__file__), 'rb') as codeFile:
        return hashlib.sha256(codeFile.read()).hexdigest()[:12]

def isNetworkError(error):
    """
    Whether an error is a dropped or timed out connection
    """
    module = type(error).__module__ or ''
    return isinstance(error, (ConnectionError, TimeoutError)) or module.startswith(('aiohttp', 'requests', 'urllib3', 'fsspec')) or 'fetch failed' in str(error)

def failStage(error):
    """
    Step of the pulse the error came from, read off the function directly under runMastPulse in the traceback
    """
    stageByFunction = {
        'loadShotWindow': 'data', 'loadBeamPower': 'data', 'buildSeedEquilibria': 'seeds', 'pickLastSurfaceFactor': 'lastSurface',
        'buildInitialProfiles': 'profiles', 'fly': 'fly', 'summary': 'compare', 'compareToExperiment': 'compare',
    }
    names = [frame.name for frame in traceback.extract_tb(error.__traceback__)]
    try:
        below = names[names.index('runMastPulse') + 1:]
    except ValueError:
        return 'database'
    for name in below:
        if name != 'attemptFly':
            return stageByFunction.get(name, 'setup')
    return 'setup'

def collectComparison(tmtx, efm, ayc, thomsonValidTimes, tPulseStart, tPulseEnd, lastSurface):
    """
    Measured and simulated Te, ne and electron stored energy at every Thomson time in the window, same mapping
    and averages as the comparison plots, plus the measured Te/ne profiles on the TORAX rho grid
    """
    keys = ['time', 'teCoreMeas', 'teCoreSim', 'teVolAvgMeas', 'teVolAvgSim', 'teEdgeMeas', 'teEdgeSim',
            'neCoreMeas', 'neCoreSim', 'neVolAvgMeas', 'neVolAvgSim', 'neEdgeMeas', 'neEdgeSim', 'weMeas', 'weSim']
    series = {key: [] for key in keys + ['teMeasProfile', 'neMeasProfile']}
    compareTimes = thomsonValidTimes[(thomsonValidTimes >= tPulseStart) & (thomsonValidTimes <= tPulseEnd)]
    for t in compareTimes:
        eq = efm.sel(time=t, method='nearest')
        thomsonNow = ayc.sel(time=t)
        r, te, ne = thomsonNow['radius'].values, thomsonNow['te'].values, thomsonNow['ne'].values
        valid = ~np.isnan(r) & ~np.isnan(te) & ~np.isnan(ne)
        if np.sum(valid) < 3:
            continue
        rho = rhoTorNormFromPsiNorm(psiNormAtMidplane(r[valid], eq), eq, lastSurface)
        inside = rho < 1.0
        if np.sum(inside) < 3:
            continue
        teMeas = measuredCoreEdgeVolAvg(rho[inside], te[valid][inside] / 1000.0, tmtx, float(t))
        neMeas = measuredCoreEdgeVolAvg(rho[inside], ne[valid][inside], tmtx, float(t))
        if teMeas[0] is None:
            continue
        teSim = coreEdgeVolAvg(tmtx, 'T_e', 'T_e_volume_avg', float(t))
        neSim = coreEdgeVolAvg(tmtx, 'n_e', 'n_e_volume_avg', float(t))

        rhoNodes, teNodes = buildNodeProfile(rho[inside], te[valid][inside], rhoNodeCount=10, medianHalfWidth=0.1)
        _, neNodes = buildNodeProfile(rho[inside], ne[valid][inside], rhoNodeCount=10, medianHalfWidth=0.1)
        vpr = tmtx._data_tree.profiles.vpr.interp(time=float(t))
        vprRho = vpr['rho_norm'].values
        pressure = np.interp(vprRho, rhoNodes, neNodes) * np.interp(vprRho, rhoNodes, teNodes) * 1.602e-19
        weMeas = 1.5 * float(np.trapezoid(pressure * vpr.values, vprRho))
        weSim = float(tmtx._data_tree.scalars.W_thermal_e.interp(time=float(t)))

        values = [float(t), teMeas[0], teSim[0], teMeas[1], teSim[1], teMeas[2], teSim[2],
                  neMeas[0], neSim[0], neMeas[1], neSim[1], neMeas[2], neSim[2], weMeas, weSim]
        for key, value in zip(keys, values):
            series[key].append(value)
        series['teMeasProfile'].append(np.interp(vprRho, rhoNodes, teNodes / 1000.0, left=np.nan, right=np.nan))
        series['neMeasProfile'].append(np.interp(vprRho, rhoNodes, neNodes, left=np.nan, right=np.nan))
    return {key: np.array(value, dtype=float) for key, value in series.items()}

def collectEquilibrium(tmtx, efm):
    """
    Get equilibrium data for each time step
    """
    tmTimes = np.array(tmtx._tm_times)
    out = {key: [] for key in ('liSim', 'q95Sim', 'psiLcfsSim', 'liMeas', 'q95Meas', 'psiBoundaryMeas')}
    for i, t in enumerate(tmTimes):
        try:
            if tmtx._state['Ip_tm'][i] == 0:
                raise ValueError('TokaMaker solve failed')
            equil = tmtx._state['equil'][i]
            out['liSim'].append(equil.get_stats(li_normalization='std')['l_i'])
            out['q95Sim'].append(abs(float(equil.get_q(psi=np.array([0.95]))[1][0])))
            out['psiLcfsSim'].append(float(tmtx._state['psi_lcfs_tm'][i]))
        except Exception:
            out['liSim'].append(np.nan)
            out['q95Sim'].append(np.nan)
            out['psiLcfsSim'].append(np.nan)
        eq = efm.sel(time=t, method='nearest')
        out['liMeas'].append(float(eq['li']))
        out['q95Meas'].append(float(np.interp(0.95, eq['psi_norm'].values, eq['qpsi_c'].values)))
        out['psiBoundaryMeas'].append(float(eq['psi_boundary']))
    out = {key: np.array(value, dtype=float) for key, value in out.items()}
    try:
        first = int(np.flatnonzero(np.isfinite(out['psiLcfsSim']))[0])
    except IndexError:
        first = 0
    out['fluxSim'] = -(out['psiLcfsSim'] - out['psiLcfsSim'][first]) * 2 * np.pi
    out['fluxMeas'] = -(out['psiBoundaryMeas'] - out['psiBoundaryMeas'][first]) * 2 * np.pi
    out['fluxMeas'][:first] = np.nan
    return out

def collectIonTemperature(tmtx, shotId, efm, thomsonValidTimes, tPulseStart, tPulseEnd, lastSurface, stride=1):
    """
    Ti sim vs CXRS at every stride-th Thomson time, core, volume average and edge plus measured Ti profile on
    TORAX grid, Empty if the shot has no act group
    """
    series = {key: [] for key in ('time', 'tiCoreMeas', 'tiCoreSim', 'tiVolAvgMeas', 'tiVolAvgSim', 'tiEdgeMeas', 'tiEdgeSim', 'tiMeasProfile')}
    if loadCxrs(shotId) is None:
        return {key: np.array(value) for key, value in series.items()}
    compareTimes = thomsonValidTimes[(thomsonValidTimes >= tPulseStart) & (thomsonValidTimes <= tPulseEnd)][::stride]
    for t in compareTimes:
        eq = efm.sel(time=t, method='nearest')
        cxrs = mapCxrsToRho(shotId, float(t), eq, lastSurface)
        if cxrs is None:
            continue
        rho, ti = cxrs
        inside = rho < 1.0
        tiMeas = measuredCoreEdgeVolAvg(rho[inside], ti[inside] / 1000.0, tmtx, float(t))
        if tiMeas[0] is None:
            continue
        tiSim = coreEdgeVolAvg(tmtx, 'T_i', 'T_i_volume_avg', float(t))
        # Measured Ti profile on the TORAX grid, NaN outside what CXRS covers
        rhoNodes, tiNodes = buildNodeProfile(rho[inside], ti[inside] / 1000.0)
        vprRho = tmtx._data_tree.profiles.vpr.interp(time=float(t))['rho_norm'].values
        values = (float(t), tiMeas[0], tiSim[0], tiMeas[1], tiSim[1], tiMeas[2], tiSim[2], np.interp(vprRho, rhoNodes, tiNodes, left=np.nan, right=np.nan))
        for key, value in zip(series, values):
            series[key].append(value)
    return {key: np.array(value, dtype=float) for key, value in series.items()}

def medianRatio(simulated, measured):
    """
    Median of sim/measured over every time where measurement > 0
    """
    ratio = np.asarray(simulated, dtype=float) / np.where(np.asarray(measured, dtype=float) > 0, measured, np.nan)
    return float(np.nanmedian(ratio)) if np.any(np.isfinite(ratio)) else np.nan

def summarizeRun(tmtx, diagnostics, shotId, settings, screenNote):
    """
    Builds database row for a run that finished computed from data the run already loaded
    """
    efm, validTimes, tPulseStart, tPulseEnd, _, _, ayc, thomsonValidTimes, _ = loadShotWindow(shotId)
    lastSurface = diagnostics['lastSurface']
    screened = pd.read_csv('MAST_info/screened_pulses.csv').set_index('shot_id')
    row = {}

    # Shot description, measured and independent of any pulse setting
    try:
        catalogRow = screened.loc[shotId]
        row['shotCampaign'], row['shotHeating'], row['shotScenario'] = catalogRow['campaign'], catalogRow['heatingResolved'], catalogRow['shot_scenario']
    except KeyError:
        row['shotCampaign'], row['shotHeating'], row['shotScenario'] = '', '', ''
    row['screenedOut'] = bool(screenNote)
    row['screenReasons'] = screenNote
    ipWindow = np.abs(efm['plasma_current_c'].sel(time=slice(tPulseStart, tPulseEnd)))
    peakEq = efm.sel(time=float(ipWindow['time'][int(np.nanargmax(ipWindow.values))]))
    row['shotWindowStart'], row['shotWindowEnd'] = tPulseStart, tPulseEnd
    row['shotIpMaxKA'] = float(np.nanmax(ipWindow.values)) / 1e3
    row['shotBtVac'] = abs(float(peakEq['bvac_val']))
    row['shotAspectRatio'] = float(peakEq['geom_axis_rc']) / float(peakEq['minor_radius'])
    row['shotElongation'] = float(peakEq['elongation'])
    row['shotTriangularity'] = 0.5 * (float(peakEq['triang_upper']) + float(peakEq['triang_lower']))
    beamPower = loadBeamPower(shotId, tPulseStart, tPulseEnd, settings['beamPowerThresholdMw'])
    if beamPower is not None:
        beamTimes, beamWatts = np.array(list(beamPower.keys())), np.array(list(beamPower.values()))
        row['shotNbiMaxMW'] = float(beamWatts.max()) / 1e6
        row['shotNbiEnergyMJ'] = float(np.trapezoid(beamWatts, beamTimes)) / 1e6
    else:
        row['shotNbiMaxMW'], row['shotNbiEnergyMJ'] = 0.0, 0.0
    try:
        bolometer = openShotGroup(shotId, 'abm', level=1)['prad_pol'].sel(time=slice(tPulseStart, tPulseEnd))
        row['shotPradMeasMW'] = float(np.nanmean(bolometer.values)) / 1e6
    except (KeyError, TypeError):
        row['shotPradMeasMW'] = np.nan
        bolometer = None

    # Health of the coupled run
    scalars = tmtx._data_tree.scalars.to_dataset()
    txTimes = scalars['time'].values
    inWindow = (txTimes > tPulseStart + 0.02) & (txTimes < tPulseEnd - 0.02)
    powerIn = sum(scalars[name].values[inWindow] for name in ('P_ohmic_e', 'P_aux_total', 'P_alpha_total') if name in scalars)
    powerOut = scalars['P_SOL_total'].values[inWindow] + scalars['dW_thermal_dt'].values[inWindow] - scalars['P_radiation_e'].values[inWindow]
    tmTimes = np.array(tmtx._tm_times)
    solved = np.array(tmtx._state['Ip_tm']) != 0
    psiLcfsTm = np.array(tmtx._state['psi_lcfs_tm'])
    psiLcfsTx = np.array([float(tmtx._data_tree.profiles.psi.interp(time=float(np.clip(t, txTimes[0], txTimes[-1]))).isel(rho_norm=-1)) for t in tmTimes]) * (-1.0 / (2.0 * np.pi))
    first = int(np.argmax(solved))
    fluxTm, fluxTx = psiLcfsTm - psiLcfsTm[first], psiLcfsTx - psiLcfsTx[first]
    row['healthLastSurface'] = lastSurface
    row['healthBeamHeating'] = diagnostics['beamHeating']
    row['healthCxrsTiInit'] = diagnostics['cxrsTiUsed']
    row['healthTmSolves'] = len(tmTimes)
    row['healthTmFailed'] = int(np.sum(np.array(tmtx._state['Ip_tm']) == 0))
    try:
        row['healthFluxGap'] = float(np.max(np.abs(fluxTm - fluxTx)[solved]) / max(np.max(np.abs(fluxTm[solved])), 1e-9))
    except ValueError:
        # Every TokaMaker solve failed
        row['healthFluxGap'] = np.nan
    row['healthPowerResidual'] = float(np.median(np.abs(powerIn - powerOut) / np.maximum(np.abs(powerIn), 1.0)))
    row['healthTxSteps'] = int(len(txTimes))

    # Sim vs experiment, median ratio sim/measured over the window
    comparison = collectComparison(tmtx, efm, ayc, thomsonValidTimes, tPulseStart, tPulseEnd, lastSurface)
    for quantity in ('teCore', 'teVolAvg', 'teEdge', 'neCore', 'neVolAvg', 'neEdge', 'we'):
        row['res' + quantity[0].upper() + quantity[1:] + 'Ratio'] = medianRatio(comparison[quantity + 'Sim'], comparison[quantity + 'Meas'])
    ionTemperature = collectIonTemperature(tmtx, shotId, efm, thomsonValidTimes, tPulseStart, tPulseEnd, lastSurface)
    row['resTiCoreRatio'] = medianRatio(ionTemperature['tiCoreSim'], ionTemperature['tiCoreMeas'])
    row['resTiVolAvgRatio'] = medianRatio(ionTemperature['tiVolAvgSim'], ionTemperature['tiVolAvgMeas'])
    equilibrium = collectEquilibrium(tmtx, efm)
    row['resLiRatio'] = medianRatio(equilibrium['liSim'], equilibrium['liMeas'])
    row['resQ95Ratio'] = medianRatio(equilibrium['q95Sim'], equilibrium['q95Meas'])
    try:
        lastSolved = int(np.flatnonzero(np.isfinite(equilibrium['fluxSim']))[-1])
    except IndexError:
        lastSolved = -1
    row['resFluxConsumedSimWb'] = float(equilibrium['fluxSim'][lastSolved])
    row['resFluxConsumedMeasWb'] = float(equilibrium['fluxMeas'][lastSolved])
    row['resPradSimMW'] = float(np.mean(-scalars['P_radiation_e'].values[inWindow])) / 1e6
    row['resTeCoreChannelRms'] = diagnostics['teRms']
    row['resNeCoreChannelRms'] = diagnostics['neRms']
    row['resComparisonTimes'] = int(len(comparison['time']))

    # Inputs, every runMastPulse setting plus the transport settings that live in BASE_CONFIG
    for name, value in settings.items():
        row['cfg' + name[0].upper() + name[1:]] = json.dumps(value) if isinstance(value, (list, tuple)) else value
    row['cfgTurbulentTransport'] = BASE_CONFIG['transport'].get('model_name')
    row['cfgTransportMachine'] = BASE_CONFIG['transport'].get('machine', '')
    row['cfgSmoothingWidth'] = BASE_CONFIG['transport'].get('smoothing_width')
    extras = collectRunExtras(tmtx, diagnostics, efm, equilibrium, ionTemperature, bolometer)
    return row, comparison, extras

def collectRunExtras(tmtx, diagnostics, efm, equilibrium, ionTemperature, bolometer):
    """
    What the per run figures need beyond the TORAX profiles, saved in the profiles file so plot_db can rebuild every
    batch figure later: TokaMaker li, q95 and consumed flux against EFIT at each solve time, the TokaMaker and EFIT
    plasma boundary shapes with the wall, CXRS Ti, the bolometer trace and the edge values TORAX was given
    """
    tmTimes = np.array(tmtx._tm_times)
    extras = xr.Dataset(coords={'tm_time': tmTimes})
    extras['tmSolved'] = ('tm_time', (np.array(tmtx._state['Ip_tm']) != 0).astype(int))
    for name in ('liSim', 'liMeas', 'q95Sim', 'q95Meas', 'fluxSim', 'fluxMeas'):
        extras[name] = ('tm_time', equilibrium[name])

    # TokaMaker boundary at each solve time, each point with its solve index
    lcfsIndex, lcfsR, lcfsZ, tracedAt = [], [], [], []
    for i in range(len(tmTimes)):
        lcfs, psiTraced = tmtx._state['lcfs_geo_tm'].get(i), 1.0
        # Tracing right at psiN 1 can come back empty, so step slightly inside until a surface traces
        for psi in (0.99, 0.98, 0.97):
            if lcfs is None or len(lcfs) == 0:
                try:
                    lcfs, psiTraced = tmtx._state['equil'][i].trace_surf(psi), psi
                except Exception:
                    lcfs = None
        if lcfs is not None and len(lcfs):
            lcfs = np.asarray(lcfs)[::max(1, len(lcfs) // 300)]
            lcfsIndex += [i] * len(lcfs)
            lcfsR += list(lcfs[:, 0])
            lcfsZ += list(lcfs[:, 1])
        tracedAt.append(psiTraced if lcfs is not None and len(lcfs) else np.nan)
    extras['lcfsTmIndex'] = ('lcfs_point', np.array(lcfsIndex, dtype=int))
    extras['lcfsR'] = ('lcfs_point', np.array(lcfsR, dtype=float))
    extras['lcfsZ'] = ('lcfs_point', np.array(lcfsZ, dtype=float))
    extras['lcfsTracedPsiN'] = ('tm_time', np.array(tracedAt, dtype=float))

    # EFIT boundary at the same times, EFIT pads its boundary arrays with NaN to one length
    efitAt = [efm.sel(time=t, method='nearest') for t in tmTimes]
    extras['efitLcfsR'] = (('tm_time', 'efit_lcfs_point'), np.array([eq['lcfs_r'].values for eq in efitAt], dtype=float))
    extras['efitLcfsZ'] = (('tm_time', 'efit_lcfs_point'), np.array([eq['lcfs_z'].values for eq in efitAt], dtype=float))

    # Wall outline TokaMaker uses, one or more closed contours
    wallIndex, wallR, wallZ = [], [], []
    for k, contour in enumerate(getattr(tmtx._tm, 'lim_contours', None) or []):
        contour = np.asarray(contour)
        wallIndex += [k] * len(contour)
        wallR += list(contour[:, 0])
        wallZ += list(contour[:, 1])
    extras['wallContour'] = ('wall_point', np.array(wallIndex, dtype=int))
    extras['wallR'] = ('wall_point', np.array(wallR, dtype=float))
    extras['wallZ'] = ('wall_point', np.array(wallZ, dtype=float))
    # In mirror mode TokaMaker stores Z first, plot_db swaps the columns back when this is 1
    extras['tmMirrorMode'] = int(bool(getattr(getattr(tmtx._tm, 'settings', None), 'mirror_mode', False)))

    # CXRS Ti at the Thomson times where the shot has CXRS
    if len(ionTemperature['time']):
        extras = extras.assign_coords(cxrs_time=ionTemperature['time'])
        for name in ('tiCoreMeas', 'tiCoreSim', 'tiVolAvgMeas', 'tiVolAvgSim', 'tiEdgeMeas', 'tiEdgeSim'):
            extras[name] = ('cxrs_time', ionTemperature[name])
        extras['tiMeasProfile'] = (('cxrs_time', 'rho_norm'), ionTemperature['tiMeasProfile'])

    # Bolometer radiated power over the window
    if bolometer is not None:
        extras = extras.assign_coords(bolometer_time=bolometer['time'].values)
        extras['pradMeas'] = ('bolometer_time', np.asarray(bolometer.values, dtype=float))

    # Edge values TORAX was given, Thomson points for a measured edge
    edge = diagnostics['edgeConditions']
    bcTimes = np.array(sorted(edge['teBc']))
    extras = extras.assign_coords(edge_bc_time=bcTimes)
    extras['teEdgeBc'] = ('edge_bc_time', np.array([edge['teBc'][t] for t in bcTimes]))
    extras['neEdgeBc'] = ('edge_bc_time', np.array([edge['neBc'][t] for t in bcTimes]))
    if 'fitTime' in edge:
        extras['teEdgeFit'] = ('edge_bc_time', np.array(edge['teFit'], dtype=float))
        extras['neEdgeFit'] = ('edge_bc_time', np.array(edge['neFit'], dtype=float))
        for name in ('pointTime', 'pointPsiN', 'pointTe', 'pointNe'):
            extras['edge' + name[0].upper() + name[1:]] = ('edge_point', np.array(edge[name], dtype=float))
    return extras

def saveRunProfiles(tmtx, comparison, profilesPath, attrs, extras=None):
    """
    Simulated profiles and scalars on the TORAX grids, measured Thomson values at each Thomson time, plus the extras
    from collectRunExtras that the per run figures need
    """
    profiles = tmtx._data_tree.profiles.to_dataset()
    scalars = tmtx._data_tree.scalars.to_dataset()
    profileNames = [name for name in ('T_e', 'T_i', 'n_e', 'n_i', 'psi', 'psi_norm', 'q', 'j_total', 'vpr', 'sigma_parallel', 'j_bootstrap', 'j_external') if name in profiles]
    scalarNames = [name for name in ('Ip', 'P_ohmic_e', 'P_aux_total', 'P_radiation_e', 'P_SOL_total', 'W_thermal_e', 'W_thermal_total', 'T_e_volume_avg', 'n_e_volume_avg', 'tau_E') if name in scalars]
    dataset = xr.merge([profiles[profileNames], scalars[scalarNames]])
    for key, value in comparison.items():
        if key == 'time':
            continue
        if value.ndim == 2:
            dataset[key] = (('thomson_time', 'rho_norm'), value)
        else:
            dataset[key] = (('thomson_time',), value)
    dataset = dataset.assign_coords(thomson_time=comparison['time'])
    if extras is not None:
        dataset = xr.merge([dataset, extras])
    dataset.attrs.update(attrs)
    dataset.to_netcdf(profilesPath)

def runShotForBatch(shotId, batchName, settings, screenNote='', maxRetries=2, retryWait=60, restoreConsole=True):
    """
    Runs one shot inside a batch and writes its database row to the pending folder, whether it ran or failed
    """
    originalCwd = os.getcwd()
    runStart = datetime.now()
    runId = f'{shotId}_{runStart.strftime("%Y%m%d_%H%M%S")}'
    runDir = os.path.join(originalCwd, 'pulse_outputs', batchName, f'shot_{shotId}')
    os.makedirs(runDir, exist_ok=True)
    os.makedirs(os.path.join(originalCwd, pendingDir), exist_ok=True)
    row = {'runId': runId, 'runTimestamp': runStart.isoformat(timespec='seconds'), 'batchName': batchName, 'shotId': shotId}

    # TokaMaker prints from Fortran straight to the file descriptors
    sys.stdout.flush()
    sys.stderr.flush()
    consoleLog = open(os.path.join(runDir, f'{runId}_console.log'), 'w')
    savedOut, savedErr = os.dup(1), os.dup(2)
    os.dup2(consoleLog.fileno(), 1)
    os.dup2(consoleLog.fileno(), 2)
    try:
        tmtx, diagnostics, error = None, None, None
        for attempt in range(maxRetries + 1):
            try:
                tmtx, diagnostics = runMastPulse(shotId, oftEnv=getOftEnv(), batchName=batchName, verbose=False, **settings)
                error = None
                break
            except Exception as e:
                error = e
                traceback.print_exc()
                if not isNetworkError(e) or attempt == maxRetries:
                    break
                print(f'network error on shot {shotId}, retrying in {retryWait} s: {e}')
                time.sleep(retryWait)
        row['runAttempts'] = attempt + 1

        if error is None:
            try:
                summary, comparison, extras = summarizeRun(tmtx, diagnostics, shotId, settings, screenNote)
                profilesPath = os.path.join(runDir, f'{runId}_profiles.nc')
                saveRunProfiles(tmtx, comparison, profilesPath, {'runId': runId, 'shotId': shotId, 'batchName': batchName, 'configHash': configHash(settings)}, extras)
                row.update({'runStatus': 'ok', 'runFailStage': '', 'runError': '', 'runProfilesPath': os.path.relpath(profilesPath, originalCwd)})
                row.update(summary)
            except Exception as e:
                traceback.print_exc()
                row.update({'runStatus': 'failed', 'runFailStage': 'database', 'runError': f'{type(e).__name__}: {e}'[:300]})
        else:
            row.update({'runStatus': 'failed', 'runFailStage': failStage(error), 'runError': f'{type(error).__name__}: {error}'[:300]})
    finally:
        os.chdir(originalCwd)
        plt.close('all')
        sys.stdout.flush()
        sys.stderr.flush()
        # Workers leave output pointed at this shot's log, Fortran flushes its buffer late and it would land in the batch terminal
        if restoreConsole:
            os.dup2(savedOut, 1)
            os.dup2(savedErr, 2)
        os.close(savedOut)
        os.close(savedErr)
        consoleLog.close()

    row['runSeconds'] = (datetime.now() - runStart).total_seconds()
    row['runOutputDir'] = os.path.relpath(runDir, originalCwd)
    row['codeHash'] = codeHash()
    row['configHash'] = configHash(settings)
    row['toraxVersion'] = torax.__version__
    row['oftVersion'] = OpenFUSIONToolkit.__version__
    row.setdefault('screenedOut', bool(screenNote))
    row.setdefault('screenReasons', screenNote)
    with open(os.path.join(originalCwd, pendingDir, f'{runId}.json'), 'w') as rowFile:
        json.dump(row, rowFile, default=str)
    return row

def mergePendingRows():
    """
    Appends every pending per-run row to runs.csv and deletes the pending files once the merged file completes
    """
    try:
        pendingFiles = sorted(name for name in os.listdir(pendingDir) if name.endswith('.json'))
    except FileNotFoundError:
        return 0
    newRows = []
    for name in pendingFiles:
        with open(os.path.join(pendingDir, name)) as rowFile:
            newRows.append(json.load(rowFile))
    if newRows:
        try:
            runs = pd.concat([pd.read_csv(runsPath, dtype={'configHash': str, 'codeHash': str}), pd.DataFrame(newRows)], ignore_index=True)
        except FileNotFoundError:
            runs = pd.DataFrame(newRows)
        tempPath = runsPath + '.tmp'
        runs.to_csv(tempPath, index=False)
        # Pending rows only get deleted once merged file completes with every row in it
        if len(pd.read_csv(tempPath, dtype=str)) != len(runs):
            raise RuntimeError(f'{tempPath} did not read back with all {len(runs)} rows, pending rows kept')
        os.replace(tempPath, runsPath)
        try:
            documented = set(pd.read_csv(dictionaryPath)['column'])
            undocumented = [column for column in runs.columns if column not in documented]
            if undocumented:
                print(f'warning: columns missing from {dictionaryPath}: {undocumented}')
        except FileNotFoundError:
            print(f'warning: no data dictionary at {dictionaryPath}')
    for name in pendingFiles:
        os.remove(os.path.join(pendingDir, name))
    os.rmdir(pendingDir)
    return len(newRows)

def findResumableBatch(shotIds, settingsHash, label):
    """
    Most recent earlier batch with the same shots, settings, code and label, so a rerun picks back up
    """
    try:
        batchNames = sorted(os.listdir('pulse_outputs'), reverse=True)
    except FileNotFoundError:
        return None
    for batchName in batchNames:
        try:
            with open(os.path.join('pulse_outputs', batchName, 'batch_shots.json')) as manifestFile:
                manifest = json.load(manifestFile)
        except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError):
            continue
        if (sorted(manifest['shotIds']) == sorted(int(shotId) for shotId in shotIds) and manifest['configHash'] == settingsHash
                and manifest['codeHash'] == codeHash() and manifest['label'] == label):
            return batchName
    return None

def runBatch(shotIds, screenNotes=None, selectionName=None, label=None, workers=4, **kwargs):
    """
    Runs a batch of shots across worker processes and merges every outcome into pulse database
    """
    screenNotes = screenNotes or {}
    settings = runSettings(**kwargs)
    settingsHash = configHash(settings)

    batchName = findResumableBatch(shotIds, settingsHash, label)
    if batchName is None:
        batchName = makeBatchName(selectionName or f'shotlist_{len(shotIds)}shots', label)
        # Shots, settings, code and label saved with the batch so rerun recognizes it
        os.makedirs(os.path.join('pulse_outputs', batchName), exist_ok=True)
        with open(os.path.join('pulse_outputs', batchName, 'batch_shots.json'), 'w') as manifestFile:
            json.dump({'shotIds': [int(shotId) for shotId in shotIds], 'configHash': settingsHash, 'codeHash': codeHash(), 'label': label}, manifestFile, indent=1)
    else:
        print(f'resuming batch {batchName}')

    # Rows left behind by an interrupted batch get merged before new runs
    mergePendingRows()
    try:
        runs = pd.read_csv(runsPath, dtype={'configHash': str, 'codeHash': str})
        done = set(runs.loc[(runs['batchName'] == batchName) & (runs['configHash'] == settingsHash) & (runs['runStatus'] == 'ok'), 'shotId'])
    except FileNotFoundError:
        done = set()
    toRun = [shotId for shotId in shotIds if shotId not in done]
    print(f'batch {batchName}: {len(shotIds) - len(toRun)} shots already done with these settings, {len(toRun)} to run on {workers} workers')

    rows = []

    def report(i, row):
        """
        Keeps the row and prints one line per finished shot
        """
        rows.append(row)
        outcome = f'ok in {row["runSeconds"]:.0f} s' if row['runStatus'] == 'ok' else f'failed at {row["runFailStage"]}: {row["runError"]}'
        print(f'[{i + 1}/{len(toRun)}] shot {row["shotId"]} {outcome}')

    if workers <= 1:
        for i, shotId in enumerate(toRun):
            report(i, runShotForBatch(shotId, batchName, settings, screenNotes.get(shotId, '')))
    else:
        # Each worker makes its own OFT_env
        os.environ['MPLBACKEND'] = 'Agg'
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as pool:
            futures = {pool.submit(runShotForBatch, shotId, batchName, settings, screenNotes.get(shotId, ''), restoreConsole=False): shotId for shotId in toRun}
            for i, future in enumerate(as_completed(futures)):
                shotId = futures[future]
                try:
                    report(i, future.result())
                except Exception as e:
                    # A worker that dies never writes its own row
                    runStart = datetime.now()
                    row = {'runId': f'{shotId}_{runStart.strftime("%Y%m%d_%H%M%S")}', 'runTimestamp': runStart.isoformat(timespec='seconds'),
                           'batchName': batchName, 'shotId': shotId, 'runStatus': 'failed', 'runFailStage': 'worker',
                           'runError': f'{type(e).__name__}: {e}'[:300], 'runSeconds': np.nan, 'codeHash': codeHash(), 'configHash': settingsHash}
                    os.makedirs(pendingDir, exist_ok=True)
                    with open(os.path.join(pendingDir, f'{row["runId"]}.json'), 'w') as rowFile:
                        json.dump(row, rowFile, default=str)
                    report(i, row)

    merged = mergePendingRows()
    failed = [row for row in rows if row['runStatus'] != 'ok']
    print(f'batch {batchName} done: {len(rows) - len(failed)} ok, {len(failed)} failed, {merged} rows added to {runsPath}')
    if failed:
        print(pd.Series([row['runFailStage'] for row in failed]).value_counts().to_string())
    return rows

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a batch of MAST pulses and add the results to the pulse database')
    parser.add_argument('--label', help='optional label for the batch folder name in place of the shot selection')
    parser.add_argument('--shots', type=int, nargs='+', help='explicit shot list, screened shots still run with a warning')
    parser.add_argument('--heating', nargs='+', help="heatingResolved labels: Ohmic, 'SS Beam', 'SW Beam', '2 Beams,SS Beam,SW Beam'")
    parser.add_argument('--campaigns', nargs='+', help='campaign labels, M5 through M9')
    parser.add_argument('--min-current', type=float, help='minimum peak plasma current in kA')
    parser.add_argument('--max-current', type=float, help='maximum peak plasma current in kA')
    parser.add_argument('--sample', type=int, help='run a random subset of this many shots')
    parser.add_argument('--seed', type=int, default=0, help='seed for the random subset')
    parser.add_argument('--workers', type=int, default=4, help='parallel worker processes')
    parser.add_argument('--max-loop', type=int, help='coupling loops, runMastPulse default when left out')
    parser.add_argument('--transport', choices=['tglfnn', 'qlknn'], default='tglfnn', help='turbulent transport surrogate')
    parser.add_argument('--edge-source', choices=['measured', 'constants'], default='measured', help='edge Te and ne from Thomson at the TORAX boundary or from the calibrated constants')
    args = parser.parse_args()

    shotIds, screenNotes, selectionName = selectShots(shotIds=args.shots, heating=args.heating, campaigns=args.campaigns, minCurrent=args.min_current,
                                                      maxCurrent=args.max_current, sampleSize=args.sample, seed=args.seed)
    settings = {'maxLoop': args.max_loop} if args.max_loop is not None else {}
    settings['turbulenceModel'] = args.transport
    settings['edgeSource'] = args.edge_source
    runBatch(shotIds, screenNotes=screenNotes, selectionName=selectionName, label=args.label, workers=args.workers, **settings)
