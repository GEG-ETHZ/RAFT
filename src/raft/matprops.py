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


def get_porosity(
    P: npt.NDArray[np.float64],
    P_0: npt.NDArray[np.float64],
    phi_0: float,
    K_d: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Get the current porosity based on the undeformed reference state.

    Assume alpha_m = alpha_s and temperature therefore does not affect porosity.
    """
    phi = 1 - (1 - phi_0) * np.exp(-(P - P_0) / K_d)
    dphi_dT = 0
    dphi_dp = -(phi - 1) / K_d

    return phi, dphi_dT, dphi_dp


def get_volumetricstrain(
    P: npt.NDArray[np.float64],
    P_0: npt.NDArray[np.float64],
    T: npt.NDArray[np.float64],
    T_0: npt.NDArray[np.float64],
    K_d: float,
    biot: float,
    therm_expansion,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Get the volumetric strain of the solid skeleton relative to the undeformed reference state.
    Non-volumetric sources are neglected.
    """
    strain = biot / K_d * (P - P_0) + therm_expansion * (T - T_0)
    dstrain_dT = therm_expansion
    dstrain_dp = biot / K_d

    return strain, dstrain_dT, dstrain_dp


def get_porosity_lagrangian(
    P: npt.NDArray[np.float64],
    P_0: npt.NDArray[np.float64],
    T: npt.NDArray[np.float64],
    T_0: npt.NDArray[np.float64],
    phi_0: float,
    K_d: float,
    biot: float,
    therm_expansion,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Get the Lagrangian porosity which includes the volumetric strain in the deformed state."""
    phi, dphi_dT, dphi_dp = get_porosity(P, P_0, phi_0, K_d)
    strain, dstrain_dT, dstrain_dp = get_volumetricstrain(P, P_0, T, T_0, K_d, biot, therm_expansion)

    phi_L = phi * (1 + strain)
    dphiL_dT = dphi_dT * (1 + strain) + phi * dstrain_dT
    dphiL_dp = dphi_dp * (1 + strain) + phi * dstrain_dp

    return phi_L, dphiL_dT, dphiL_dp


def get_enthalpy(
    p: npt.NDArray[np.float64],
    T: npt.NDArray[np.float64],
    fluid: dict[str, Any],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Get the fluid enthalpy at a specified pressure and temperature."""
    if fluid["density"] != 0:
        # Linearized internal energy
        rho_f = fluid["density"]
        T0 = fluid["reference_temperature"]
        p0 = fluid["reference_pressure"]
        alpha_f = fluid["therm_expansion"]
        cp_f = fluid["specific_heat"]

        h_f = cp_f * (T - T0) + (1 - T0 * alpha_f) / rho_f * (p - p0)
        dhf_dT = cp_f
        dhf_dp = (1 - T0 * alpha_f) / rho_f
    else:
        # Nonlinear internal energy
        h_f = 0
        dhf_dT = 0
        dhf_dp = 0
    return h_f, dhf_dT, dhf_dp


def get_internal_energy(
    T: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],  # noqa
    fluid: dict[str, Any],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Get the fluid internal energy at a specified pressure and temperature."""
    if fluid["density"] != 0:
        # Linearized internal energy
        rho_f = fluid["density"]  # noqa
        T0 = fluid["reference_temperature"]
        p0 = fluid["reference_pressure"]  # noqa
        alpha_f = fluid["therm_expansion"]  # noqa
        cp_f = fluid["specific_heat"]

        # u_f = (cp_f + p0 * alpha_f / rho_f) * (T - T0) + (p0 * cp_f - T0 * alpha_f) / rho_f * (p - p0)
        # duf_dT = (cp_f + p0 * alpha_f / rho_f)
        # duf_dp = (p0 * cp_f - T0 * alpha_f) / rho_f
        u_f = cp_f * (T - T0)
        duf_dT = cp_f
        duf_dp = 0
    else:
        # Nonlinear internal energy
        u_f = 0
        duf_dT = 0
        duf_dp = 0
    return u_f, duf_dT, duf_dp


def get_effective_permeability(
    kf: npt.NDArray[np.float64],
    T: npt.NDArray[np.float64],  # noqa
    p: npt.NDArray[np.float64],  # noqa
    fluid: dict[str, Any],  # noqa
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Get the effective permeability. Apply Forchheimer flow and other modifiers."""
    kf_eff = kf
    dkf_dT = 0
    dkf_dp = 0

    return kf_eff, dkf_dT, dkf_dp
