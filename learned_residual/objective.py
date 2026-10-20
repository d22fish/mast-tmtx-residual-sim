"""
Loss of simulated profiles against the raw diagnostic points in a shot's targets file

Each point gets a normalized error e = (sim - meas)/sigmaEff with sigmaEff^2 = sigma^2 + (floor * meas)^2. The
loss of a channel is mean Huber of e over the points the loss mask keeps, the shot loss is the weighted mean of
the channel losses
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

# Double precision as in TORAX
jax.config.update('jax_enable_x64', True)

projectDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
targetsDir = os.path.join(projectDir, 'pulse_database', 'targets')

# Target prefix, measured name, time coordinate, and simulated TORAX profile of each channel
channels = {'Te': ('thomson', 'te', 'thomson_time', 'T_e'), 'ne': ('thomson', 'ne', 'thomson_time', 'n_e'), 'Ti': ('cxrs', 'ti', 'cxrs_time', 'T_i')}
defaultWeights = {'Te': 1.0, 'ne': 1.0, 'Ti': 0.5}

def huber(x, delta=3.0):
    """
    Quadratic within delta of zero and linear outside, 0.5 x^2 and delta (abs(x) - 0.5 delta)
    """
    absX = jnp.abs(x)
    return jnp.where(absX <= delta, 0.5 * x ** 2, delta * (absX - 0.5 * delta))

def effectiveSigma(sigma, measured, floor=0.05):
    """
    Measurement error with a model error floor of floor times the measured value added in quadrature
    """
    return jnp.sqrt(sigma ** 2 + (floor * measured) ** 2)

def normalizedError(simulated, measured, sigma, mask, floor=0.05):
    """
    (sim - meas)/sigmaEff at every point
    """
    error = (simulated - measured) / effectiveSigma(sigma, measured, floor)
    return jnp.where(mask, error, 0.0)

def simAtPoints(profile, simTimes, rho, pointTimes, pointRho):
    """
    Simulated profile at each point, linear in time between TORAX outputs, linear in rho at each point time
    """
    # Time interpolation column by column, rho interpolation per point time
    inTime = jax.vmap(lambda column: jnp.interp(pointTimes, simTimes, column), in_axes=1, out_axes=1)(profile)
    return jax.vmap(lambda r, row: jnp.interp(r, rho, row))(pointRho, inTime)

def channelLoss(simulated, measured, sigma, mask, floor=0.05, delta=3.0):
    """
    Mean Huber of the normalized error over the masked points
    """
    count = jnp.sum(mask)
    return jnp.sum(huber(normalizedError(simulated, measured, sigma, mask, floor), delta)) / jnp.maximum(count, 1)

def shotLoss(profiles, targets, weights=defaultWeights, floor=0.05, delta=3.0):
    """
    Weighted mean of the channel losses of one shot
    """
    total, weightSum = 0.0, 0.0
    for name, points in targets.items():
        simulated = simAtPoints(profiles[name], profiles['time'], profiles['rho'], points['times'], points['rho'])
        weight = weights[name] * (jnp.sum(points['mask']) > 0)
        total = total + weight * channelLoss(simulated, points['measured'], points['sigma'], points['mask'], floor, delta)
        weightSum = weightSum + weight
    return total / jnp.maximum(weightSum, 1e-12)

def loadTargets(shotId, tStart=None, tEnd=None, directory=targetsDir):
    """
    Reads shot's targets file
    """
    targets = {}
    with xr.open_dataset(os.path.join(directory, f'{shotId}_targets.nc')) as data:
        for name, (prefix, measName, timeName, _) in channels.items():
            try:
                times = data[timeName].values
                mask = data[f'{prefix}LossMask'].values == 1
            except KeyError:
                continue
            keep = np.ones(len(times), dtype=bool)
            if tStart is not None:
                keep &= times >= tStart
            if tEnd is not None:
                keep &= times <= tEnd
            mask = mask[keep]
            if not mask.any():
                continue
            rho, measured, sigma = (data[key].values[keep] for key in (f'{prefix}Rho', f'{measName}Meas', f'{measName}Sigma'))
            mask &= np.isfinite(rho) & np.isfinite(measured) & np.isfinite(sigma) & (measured > 0)
            targets[name] = {'times': jnp.asarray(times[keep]), 'rho': jnp.asarray(np.where(mask, rho, 0.5)), 'measured': jnp.asarray(np.where(mask, measured, 1.0)),
                             'sigma': jnp.asarray(np.where(mask, sigma, 1.0)), 'mask': jnp.asarray(mask)}
    return targets

def loadProfiles(path):
    """
    Simulated Te, ne, and Ti from baseline file
    """
    with xr.open_dataset(path) as data:
        profiles = {name: jnp.asarray(data[simName].values) for name, (_, _, _, simName) in channels.items()}
        profiles['time'] = jnp.asarray(data['time'].values)
        profiles['rho'] = jnp.asarray(data['rho_norm'].values)
    return profiles

def scoreRun(profilesPath, shotId, weights=defaultWeights, floor=0.05, delta=3.0, directory=targetsDir):
    """
    Shot loss, per channel loss, point count of one saved run over its window
    """
    profiles = loadProfiles(profilesPath)
    targets = loadTargets(shotId, float(profiles['time'][0]), float(profiles['time'][-1]), directory)
    result = {'loss': float(shotLoss(profiles, targets, weights, floor, delta))}
    for name, points in targets.items():
        simulated = simAtPoints(profiles[name], profiles['time'], profiles['rho'], points['times'], points['rho'])
        result[f'{name}Loss'] = float(channelLoss(simulated, points['measured'], points['sigma'], points['mask'], floor, delta))
        result[f'{name}Points'] = int(points['mask'].sum())
    return result