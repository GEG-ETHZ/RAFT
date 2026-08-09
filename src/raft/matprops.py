"""All material property correlations are stored here."""

import CoolProp.CoolProp as CP  # noqa: N817
import numpy as np
import numpy.typing as npt

# Fluid used for CoolProp-based (nonconstant) viscosity, selected when the config
# viscosity is set to 0.
COOLPROP_FLUID = "WATER"

_abstract_state: CP.AbstractState | None = None


def _get_abstract_state() -> CP.AbstractState:
    """Lazily create and cache the CoolProp AbstractState used for viscosity lookups."""
    global _abstract_state
    if _abstract_state is None:
        _abstract_state = CP.AbstractState("HEOS", COOLPROP_FLUID)
    return _abstract_state


def get_viscosity(
    T: npt.NDArray[np.float64],
    P: npt.NDArray[np.float64],
    viscosity_coefficients: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Get viscosity and its temperature derivative, at temperature T [K] and pressure P [Pa].

    `viscosity_coefficients` of 0 selects a non-constant, real-fluid viscosity looked up
    from CoolProp (`COOLPROP_FLUID`) at each (T, P); any other value is treated as a
    constant viscosity, and P is ignored.
    """
    T_arr = np.atleast_1d(np.asarray(T, dtype=float))

    if viscosity_coefficients != 0:
        mu = np.full(T_arr.shape, viscosity_coefficients)
        dmudT = np.zeros(T_arr.shape)
        return mu.reshape(np.shape(T)), dmudT.reshape(np.shape(T))

    # Cap limits, to keep (T, P) within CoolProp's valid range even for transient/
    # intermediate solver states (e.g. under-relaxed Newton iterates).
    T_arr = np.clip(T_arr, 274.0, 600.0)
    P_arr = np.broadcast_to(np.asarray(P, dtype=float), T_arr.shape)
    P_arr = np.clip(P_arr, 1.0e5, 1.0e8)
    AS = _get_abstract_state()

    # CoolProp has no closed-form viscosity derivative, so dmu/dT is a central difference.
    dT = 0.5  # K

    mu = np.empty(T_arr.shape)
    dmudT = np.empty(T_arr.shape)
    for i, (T_i, P_i) in enumerate(zip(T_arr.ravel(), P_arr.ravel())):
        AS.update(CP.PT_INPUTS, P_i, T_i)
        mu.flat[i] = AS.viscosity()

        AS.update(CP.PT_INPUTS, P_i, T_i + dT)
        mu_plus = AS.viscosity()
        AS.update(CP.PT_INPUTS, P_i, T_i - dT)
        mu_minus = AS.viscosity()
        dmudT.flat[i] = (mu_plus - mu_minus) / (2 * dT)

    return mu.reshape(np.shape(T)), dmudT.reshape(np.shape(T))


def get_density(
    T: npt.NDArray[np.float64],
    P: npt.NDArray[np.float64],  # noqa
    T0: npt.NDArray[np.float64],
    alpha: float,
    rho_0: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Get density and its temperature derivative, at temperature T [K] and pressure P [Pa]."""
    density = rho_0 * (1 - alpha * (T - T0))

    drho_dT = rho_0 * alpha * np.ones(np.shape(T))

    return density, drho_dT
