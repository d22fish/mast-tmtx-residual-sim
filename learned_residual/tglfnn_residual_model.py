"""
TGLF-NN model design with a learned local correction, TORAX is transport model tglfnn-residual

Small NN, inputs plasma features and returns multipliers electron heat diffusivity, ion heat diffusivity, 
electron particle transport. TORAX clips, patches, smooths corrected coefficients.
"""

import dataclasses
from functools import cache
from typing import Annotated, Literal, get_args

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pydantic
from jax.flatten_util import ravel_pytree
from torax._src.torax_pydantic import model_config
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import pydantic_model
from torax._src.transport_model import tglfnn_ukaea_transport_model
from torax.transport import register_transport_model

featureNames = ('aOverLTe', 'aOverLTi', 'aOverLne', 'q', 'magneticShear', 'tiOverTe', 'logCollisionality', 'logBetaE', 'localEpsilon', 'elongation', 'triangularity', 'rho')
outputNames = ('chiElectron', 'chiIon', 'particle')
defaultHidden = (32, 32)

class CorrectionNetwork(nn.Module):
    """
    Tanh hidden layers and a linear output layer with zero starting weights
    """
    hiddenSizes: tuple[int, ...] = defaultHidden

    @nn.compact
    def __call__(self, x):
        """
        Network output at every row of x
        """
        for size in self.hiddenSizes:
            x = jnp.tanh(nn.Dense(size, param_dtype=jnp.float64)(x))
        return nn.Dense(len(outputNames), param_dtype=jnp.float64, kernel_init=nn.initializers.zeros)(x)

def initParams(seed=0, hiddenSizes=defaultHidden):
    """
    Network parameter tree, hidden layers at the flax default scale
    """
    return CorrectionNetwork(tuple(hiddenSizes)).init(jax.random.PRNGKey(seed), jnp.zeros((1, len(featureNames))))

def initWeights(seed=0, hiddenSizes=defaultHidden):
    """
    Flat weight array of the network parameters with zero output layer - every multiplier starts at 1
    """
    flat, _ = ravel_pytree(initParams(seed, hiddenSizes))
    return np.array(flat)

@cache
def unravelWeights(hiddenSizes=defaultHidden):
    """
    Function from a flat weight array to the network parameter tree
    """
    _, unravel = ravel_pytree(initParams(0, hiddenSizes))
    return unravel

def weightCount(hiddenSizes=defaultHidden):
    """
    Length of the flat weight array for given hidden layer sizes
    """
    return len(initWeights(0, hiddenSizes))

def mlp(weights, features, hiddenSizes=defaultHidden):
    """
    Network output at every row of features
    """
    hiddenSizes = tuple(hiddenSizes)
    return CorrectionNetwork(hiddenSizes).apply(unravelWeights(hiddenSizes)(weights), features)

def multipliers(weights, features, maxFactor=5.0, hiddenSizes=defaultHidden):
    """
    Correction factors between 1/maxFactor and maxFactor, exp of the bounded network output
    """
    return jnp.exp(jnp.log(maxFactor) * jnp.tanh(mlp(weights, features, hiddenSizes)))

def safeLog(x, floor=1e-12):
    """
    Log with non-finite and tiny values set to floor
    """
    clean = jnp.where(jnp.isfinite(x) & (x > floor), x, floor)
    return jnp.log(clean)

def correctionFeatures(tglfInputs, geo):
    """
    Local features on the face grid from the same inputs as TGLF-NN, columns in order of featureNames
    """
    # TGLF-NN shear definition, (r/q)^2 q' reduces to s = (r/q) dq/dr
    shear = (tglfInputs.r_minor / tglfInputs.q) ** 2 * tglfInputs.q_prime
    epsilon = tglfInputs.r_minor * tglfInputs.Rmin / tglfInputs.Rmaj
    columns = [tglfInputs.lref_over_lte, tglfInputs.lref_over_lti, tglfInputs.lref_over_lne, tglfInputs.q, shear,
               tglfInputs.Ti_over_Te, safeLog(tglfInputs.nu_ee), safeLog(tglfInputs.beta_e), epsilon,
               tglfInputs.kappa, tglfInputs.delta, geo.rho_face_norm]
    features = jnp.stack(columns, axis=-1)
    # Set non-finite features to 0
    return jnp.where(jnp.isfinite(features), features, 0.0)

def standardize(features, featureMean, featureSpread, limit=5.0):
    """
    Features shifted and scaled to unit spread, then clipped to plus or minus limit
    """
    return jnp.clip((features - featureMean) / featureSpread, -limit, limit)

def finiteCoefficient(values, low, high):
    """
    Infinite values set to the bound they clip to, NaN set to the lower bound
    """
    bound = jnp.where(values > 0, high, low)
    return jnp.where(jnp.isfinite(values), values, bound)

def finiteCoefficients(coefficients, transport):
    """
    TGLF-NN coefficients with every non-finite face replaced
    """
    return dataclasses.replace(coefficients, chi_face_el=finiteCoefficient(coefficients.chi_face_el, transport.chi_min, transport.chi_max),
                               chi_face_ion=finiteCoefficient(coefficients.chi_face_ion, transport.chi_min, transport.chi_max),
                               d_face_el=finiteCoefficient(coefficients.d_face_el, transport.D_e_min, transport.D_e_max),
                               v_face_el=finiteCoefficient(coefficients.v_face_el, transport.V_e_min, transport.V_e_max))

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ResidualRuntimeParams(tglfnn_ukaea_transport_model.RuntimeParams):
    """
    TGLF-NN runtime params plus network weights, feature scaling, multiplier bound
    """
    weights: jax.Array
    featureMean: jax.Array
    featureSpread: jax.Array
    maxFactor: float

@dataclasses.dataclass(frozen=True, eq=False)
class ResidualTransportModel(tglfnn_ukaea_transport_model.TGLFNNukaeaTransportModel):
    """
    TGLF-NN coefficients times network multipliers, holdBase stops gradient through TGLF-NN and features
    """
    hiddenSizes: tuple[int, ...] = defaultHidden
    holdBase: bool = False

    def call_implementation(self, transport, runtime_params, geo, core_profiles, pedestal_model_output):
        """
        Plain TGLF-NN coefficients scaled by the multipliers at every face
        """
        base = finiteCoefficients(super().call_implementation(transport, runtime_params, geo, core_profiles, pedestal_model_output), transport)
        tglfInputs = self._prepare_tglf_inputs(transport=transport, geo=geo, core_profiles=core_profiles, poloidal_velocity_multiplier=runtime_params.neoclassical.poloidal_velocity_multiplier)
        features = correctionFeatures(tglfInputs, geo)
        if self.holdBase:
            base = jax.tree.map(jax.lax.stop_gradient, base)
            features = jax.lax.stop_gradient(features)
        features = standardize(features, transport.featureMean, transport.featureSpread)
        factors = multipliers(transport.weights, features, transport.maxFactor, self.hiddenSizes)
        # Particle factor scales D and V together
        return dataclasses.replace(base, chi_face_el=base.chi_face_el * factors[:, 0], chi_face_ion=base.chi_face_ion * factors[:, 1],
                                   d_face_el=base.d_face_el * factors[:, 2], v_face_el=base.v_face_el * factors[:, 2])

class ResidualTransportConfig(pydantic_model.TGLFNNukaeaTransportModel):
    """
    Config for tglfnn-residual, the TGLF-NN config plus the correction network fields
    """
    model_name: Annotated[Literal['tglfnn-residual'], torax_pydantic.JAX_STATIC] = 'tglfnn-residual'
    weights: torax_pydantic.NumpyArray1D = pydantic.Field(default_factory=initWeights)
    featureMean: torax_pydantic.NumpyArray1D = pydantic.Field(default_factory=lambda: np.zeros(len(featureNames)))
    featureSpread: torax_pydantic.NumpyArray1D = pydantic.Field(default_factory=lambda: np.ones(len(featureNames)))
    maxFactor: pydantic.PositiveFloat = 5.0
    hiddenSizes: Annotated[tuple[int, ...], torax_pydantic.JAX_STATIC] = defaultHidden
    holdBase: Annotated[bool, torax_pydantic.JAX_STATIC] = False

    @pydantic.model_validator(mode='after')
    def checkShapes(self):
        """
        Weight array length must match hidden sizes, feature scaling must have one value per feature
        """
        if len(self.weights) != weightCount(self.hiddenSizes):
            raise ValueError(f'weights has {len(self.weights)} values, hidden sizes {self.hiddenSizes} need {weightCount(self.hiddenSizes)}')
        if len(self.featureMean) != len(featureNames) or len(self.featureSpread) != len(featureNames):
            raise ValueError(f'featureMean and featureSpread need {len(featureNames)} values each')
        return self

    def build_transport_model(self):
        """
        Transport model with static network layout
        """
        return ResidualTransportModel(machine=self.machine, hiddenSizes=self.hiddenSizes, holdBase=self.holdBase)

    def build_runtime_params(self, t):
        """
        TGLF-NN runtime params at time t and the network fields
        """
        base = dataclasses.asdict(super().build_runtime_params(t))
        return ResidualRuntimeParams(weights=jnp.asarray(self.weights), featureMean=jnp.asarray(self.featureMean),
                                     featureSpread=jnp.asarray(self.featureSpread), maxFactor=self.maxFactor, **base)

def stateFeatures(stepFn, simState):
    """
    Unscaled features of one simulation state on the face grid, for fitting featureMean and featureSpread
    """
    model = stepFn.solver.physics_models.transport_model
    runtimeParams = stepFn.runtime_params_provider(simState.t)
    tglfInputs = model._prepare_tglf_inputs(transport=runtimeParams.transport, geo=simState.geometry, core_profiles=simState.core_profiles,
                                            poloidal_velocity_multiplier=runtimeParams.neoclassical.poloidal_velocity_multiplier)
    return np.asarray(correctionFeatures(tglfInputs, simState.geometry))

def residualTransportConfig(baseTransport, weights=None, hiddenSizes=defaultHidden, holdBase=False, featureMean=None, featureSpread=None, maxFactor=5.0):
    """
    Transport section for a TORAX config dict, the saved TGLF-NN settings kept and the model swapped to tglfnn-residual
    """
    config = dict(baseTransport, model_name='tglfnn-residual', hiddenSizes=tuple(hiddenSizes), holdBase=holdBase, maxFactor=maxFactor)
    config['weights'] = initWeights(hiddenSizes=hiddenSizes) if weights is None else np.asarray(weights, dtype=float)
    config['featureMean'] = np.zeros(len(featureNames)) if featureMean is None else np.asarray(featureMean, dtype=float)
    config['featureSpread'] = np.ones(len(featureNames)) if featureSpread is None else np.asarray(featureSpread, dtype=float)
    return config

def isRegistered(modelName='tglfnn-residual'):
    """
    Whether TORAX already has a transport config with this model name
    """
    choices = get_args(model_config.ToraxConfig.model_fields['transport'].annotation)
    return any(getattr(choice, 'model_fields', {}).get('model_name') is not None and choice.model_fields['model_name'].default == modelName for choice in choices)

# Register tglfnn-residual with TORAX
if not isRegistered():
    register_transport_model(ResidualTransportConfig)
