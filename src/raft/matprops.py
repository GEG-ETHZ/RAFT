"""All material property correlations are stored here."""

from typing import Any

import CoolProp.CoolProp as CP  # noqa: N817
import numpy as np
import numpy.typing as npt


def create_abstract_state(fluid_type: str) -> CP.AbstractState:
    """Create a CoolProp AbstractState for a fluid, to be reused across property lookups."""
    return CP.AbstractState("HEOS", fluid_type)


def get_viscosity(
    T: npt.NDArray[np.float64],
    P: npt.NDArray[np.float64],
    fluid: dict[str, Any],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Get viscosity and its temperature derivative, at temperature T [K] and pressure P [Pa].

    `fluid["viscosity"]` of 0 selects a non-constant, real-fluid viscosity looked up from
    CoolProp via `fluid["abstract_state"]` at each (T, P); any other value is treated as a
    constant viscosity, and P and the abstract state are ignored.
    """
    viscosity_coefficients = fluid["viscosity"]
    T_arr = np.atleast_1d(np.asarray(T, dtype=float))

    if viscosity_coefficients != 0:
        mu = np.full(T_arr.shape, viscosity_coefficients)
        dmudT = np.zeros(T_arr.shape)
        dmudP = np.zeros(T_arr.shape)
        return mu.reshape(np.shape(T)), dmudT.reshape(np.shape(T)), dmudP.reshape(np.shape(T))

    # Cap limits, to keep (T, P) within CoolProp's valid range even for transient/
    # intermediate solver states (e.g. under-relaxed Newton iterates).
    T_arr = np.clip(T_arr, 274.0, 600.0)
    P_arr = np.broadcast_to(np.asarray(P, dtype=float), T_arr.shape)
    P_arr = np.clip(P_arr, 1.0e5, 1.0e8)
    AS = fluid["abstract_state"]
    assert AS is not None

    # CoolProp has no closed-form viscosity derivative, so dmu/dT is a central difference.
    dT = 0.5  # K
    dP = 1  # Pa

    mu = np.empty(T_arr.shape)
    dmudT = np.empty(T_arr.shape)
    dmudP = np.empty(T_arr.shape)
    for i, (T_i, P_i) in enumerate(zip(T_arr.ravel(), P_arr.ravel())):
        AS.update(CP.PT_INPUTS, P_i, T_i)
        mu.flat[i] = AS.viscosity()

        AS.update(CP.PT_INPUTS, P_i, T_i + dT)
        mu_plus = AS.viscosity()
        AS.update(CP.PT_INPUTS, P_i, T_i - dT)
        mu_minus = AS.viscosity()
        dmudT.flat[i] = (mu_plus - mu_minus) / (2 * dT)

        AS.update(CP.PT_INPUTS, P_i + dP, T_i)
        mu_plus = AS.viscosity()
        AS.update(CP.PT_INPUTS, P_i - dP, T_i)
        mu_minus = AS.viscosity()
        dmudP.flat[i] = (mu_plus - mu_minus) / (2 * dP)

    return mu.reshape(np.shape(T)), dmudT.reshape(np.shape(T)), dmudP.reshape(np.shape(T))


def get_density(
    T: npt.NDArray[np.float64],
    P: npt.NDArray[np.float64],  # noqa
    fluid: dict[str, Any],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Get density and its temperature derivative, at temperature T [K] and pressure P [Pa]."""
    T0 = fluid["reference_temperature"]
    alpha = fluid["therm_expansion"]
    rho_0 = fluid["density"]
    density = rho_0 * (1 - alpha * (T - T0))

    drho_dT = rho_0 * alpha * np.ones(np.shape(T))
    drho_dp = 0 * np.ones(np.shape(T))

    return density, drho_dT, drho_dp
