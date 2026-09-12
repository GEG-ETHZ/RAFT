"""All material property correlations are stored here."""

from dataclasses import dataclass
from typing import Any

import CoolProp.CoolProp as CP  # noqa: N817
import numpy as np
import numpy.typing as npt

Number = npt.NDArray[np.float64] | float


@dataclass(frozen=True, slots=True)
class PropertyResult:
    """A material property value together with its partial derivatives w.r.t. p and T.

    Always array-valued, even where the underlying formula is a constant, so that callers
    can slice/index freely without checking for a bare scalar.

    Plain data container — no arithmetic operators. Callers that need to combine two
    `PropertyResult`s (e.g. via the product rule) do so explicitly and build a new one.
    """

    value: npt.NDArray[np.float64]  # Value of the property
    dp: npt.NDArray[np.float64]  # Derivative wrt pressure
    dT: npt.NDArray[np.float64]  # Derivative wrt temperature # noqa


@dataclass(frozen=True, slots=True)
class EnthalpyResult:
    """Fluid enthalpy, its partial derivatives, and the constant-pressure specific heat.

    Kept separate from `PropertyResult` since `cp` is not a partial derivative of `value` —
    it's an independent quantity (see `get_enthalpy` docstring).
    """

    value: npt.NDArray[np.float64]  # Value of the property
    dp: npt.NDArray[np.float64]  # Derivative wrt pressure
    dT: npt.NDArray[np.float64]  # Derivative wrt temperature # noqa
    cp: npt.NDArray[np.float64]  # specific heat for SUPG calcs


def create_abstract_state(fluid_type: str) -> CP.AbstractState:
    """Create a CoolProp AbstractState for a fluid, to be reused across property lookups."""
    return CP.AbstractState("HEOS", fluid_type)


def get_viscosity(
    T: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    fluid: dict[str, Any],
    T_flow_lagged: bool = False,
) -> PropertyResult:
    """Get viscosity and its temperature derivative, at temperature T [K] and pressure p [Pa].

    `fluid["viscosity"]` of 0 selects a non-constant, real-fluid viscosity looked up from
    CoolProp via `fluid["abstract_state"]` at each (T, p); any other value is treated as a
    constant viscosity, and p and the abstract state are ignored.

    `T_flow_lagged` says whether this T is independent of the primary unknown T being solved
    for (True, e.g. a frozen/extrapolated T_flow) or literally is it (False); when True,
    dmu_dT is reported as zero since that coupling doesn't exist for the caller's Jacobian.
    """
    viscosity_coefficients = fluid["viscosity"]
    T_arr = np.atleast_1d(np.asarray(T, dtype=float))

    if viscosity_coefficients != 0:
        mu = np.full(T_arr.shape, viscosity_coefficients)
        dmu_dT = np.zeros(T_arr.shape)
        dmu_dp = np.zeros(T_arr.shape)
        return PropertyResult(
            mu.reshape(np.shape(T)), dmu_dp.reshape(np.shape(T)), dmu_dT.reshape(np.shape(T))
        )

    # Cap limits, to keep (T, p) within CoolProp's valid range even for transient/
    # intermediate solver states (e.g. under-relaxed Newton iterates).
    T_arr = np.clip(T_arr, 274.0, 600.0)
    p_arr = np.broadcast_to(np.asarray(p, dtype=float), T_arr.shape)
    p_arr = np.clip(p_arr, 1.0e5, 1.0e8)
    AS = fluid["abstract_state"]
    assert AS is not None

    # CoolProp has no closed-form viscosity derivative, so dmu/dT is a central difference.
    dT = 0.5  # K
    dp = 1  # Pa

    mu = np.empty(T_arr.shape)
    dmu_dT = np.empty(T_arr.shape)
    dmu_dp = np.empty(T_arr.shape)
    for i, (T_i, p_i) in enumerate(zip(T_arr.ravel(), p_arr.ravel())):
        AS.update(CP.PT_INPUTS, p_i, T_i)
        mu.flat[i] = AS.viscosity()

        AS.update(CP.PT_INPUTS, p_i, T_i + dT)
        mu_plus = AS.viscosity()
        AS.update(CP.PT_INPUTS, p_i, T_i - dT)
        mu_minus = AS.viscosity()
        dmu_dT.flat[i] = (mu_plus - mu_minus) / (2 * dT)

        AS.update(CP.PT_INPUTS, p_i + dp, T_i)
        mu_plus = AS.viscosity()
        AS.update(CP.PT_INPUTS, p_i - dp, T_i)
        mu_minus = AS.viscosity()
        dmu_dp.flat[i] = (mu_plus - mu_minus) / (2 * dp)

    if T_flow_lagged:
        dmu_dT = np.zeros(T_arr.shape)

    return PropertyResult(mu.reshape(np.shape(T)), dmu_dp.reshape(np.shape(T)), dmu_dT.reshape(np.shape(T)))


def get_density(
    T: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],  # noqa
    fluid: dict[str, Any],
    T_flow_lagged: bool = False,
) -> PropertyResult:
    """Get density and its temperature derivative, at temperature T [K] and pressure p [Pa].

    `T_flow_lagged` says whether this T is independent of the primary unknown T being solved
    for (True, e.g. a frozen/extrapolated T_flow) or literally is it (False); when True,
    drho_dT is reported as zero since that coupling doesn't exist for the caller's Jacobian.
    """
    T0 = fluid["reference_temperature"]
    p0 = fluid["reference_pressure"]
    alpha = fluid["therm_expansion"]
    rho_0 = fluid["density"]
    Kf = fluid["bulk_mod"]
    density = rho_0 * (1 - alpha * (T - T0)) * (1 + 1 / Kf * (p - p0))

    drho_dT = np.zeros(np.shape(T)) if T_flow_lagged else -rho_0 * alpha * np.ones(np.shape(T))
    drho_dp = rho_0 * 1 / Kf * np.ones(np.shape(p))

    return PropertyResult(density, drho_dp, drho_dT)


def get_porosity(
    p: npt.NDArray[np.float64],
    p_0: npt.NDArray[np.float64],
    phi_0: float,
    K_d: float,
) -> PropertyResult:
    """Get the current porosity based on the undeformed reference state.

    Assume alpha_m = alpha_s and temperature therefore does not affect porosity.
    """
    exponent = np.clip(-(p - p_0) / K_d, -700, 700)  # exp(709) ~ float64 max
    phi = 1 - (1 - phi_0) * np.exp(exponent)
    dphi_dT = np.zeros(np.shape(phi))
    dphi_dp = -(phi - 1) / K_d

    return PropertyResult(phi, dphi_dp, dphi_dT)


def get_volumetricstrain(
    p: npt.NDArray[np.float64],
    p_0: npt.NDArray[np.float64],
    T: npt.NDArray[np.float64],
    T_0: npt.NDArray[np.float64],
    K_d: float,
    biot: float,
    therm_expansion: float,
    T_flow_lagged: bool = False,
    volumetric_strain_on: bool = True,
) -> PropertyResult:
    """Get the volumetric strain of the solid skeleton relative to the undeformed reference state.

    Non-volumetric sources are neglected.

    `T_flow_lagged` says whether this T is independent of the primary unknown T being solved
    for (True, e.g. a frozen/extrapolated T_flow) or literally is it (False); when True,
    dstrain_dT is reported as zero since that coupling doesn't exist for the caller's Jacobian.

    `volumetric_strain_on` turns strain off entirely (e.g. while establishing initial
    conditions, which are by definition unstrained relative to themselves) - zero strain
    regardless of T/T_0/p/p_0, avoiding any dependence on the reference state.
    """
    if not volumetric_strain_on:
        zeros = np.zeros(np.shape(p))
        return PropertyResult(zeros, zeros, zeros)

    strain = biot / K_d * (p - p_0) + therm_expansion * (T - T_0)
    dstrain_dT = np.full(np.shape(strain), 0.0 if T_flow_lagged else therm_expansion)
    dstrain_dp = np.full(np.shape(strain), biot / K_d)

    return PropertyResult(strain, dstrain_dp, dstrain_dT)


def get_porosity_lagrangian(
    p: npt.NDArray[np.float64],
    p_0: npt.NDArray[np.float64],
    T: npt.NDArray[np.float64],
    T_0: npt.NDArray[np.float64],
    phi_0: float,
    K_d: float,
    biot: float,
    therm_expansion: float,
    T_flow_lagged: bool = False,
    volumetric_strain_on: bool = True,
) -> PropertyResult:
    """Get the Lagrangian porosity which includes the volumetric strain in the deformed state."""
    phi = get_porosity(p, p_0, phi_0, K_d)
    strain = get_volumetricstrain(
        p, p_0, T, T_0, K_d, biot, therm_expansion, T_flow_lagged, volumetric_strain_on
    )

    # phi_L = phi * (1 + strain), by the product rule:
    phi_L = phi.value * (1 + strain.value)
    dphi_L_dT = phi.dT * (1 + strain.value) + phi.value * strain.dT
    dphi_L_dp = phi.dp * (1 + strain.value) + phi.value * strain.dp

    return PropertyResult(phi_L, dphi_L_dp, dphi_L_dT)


def get_enthalpy(
    p: npt.NDArray[np.float64],
    T: npt.NDArray[np.float64],
    T_flow: npt.NDArray[np.float64],
    fluid: dict[str, Any],
    T_flow_lagged: bool = False,
) -> EnthalpyResult:
    """Get the fluid enthalpy at a specified pressure and temperature.

    Built as hf = u_f + p / rho_f, using the same closed-form u_f as
    get_internal_energy and the same density formula as get_density (with its constant,
    reference-density derivatives, rather than get_density's exact ones), so the two stay
    exactly consistent without either function calling the other at runtime.

    `rho_f` is evaluated at `T_flow` rather than `T`. `T_flow_lagged` says whether T_flow is
    independent of the unknown T (True, e.g. the linearized solver's frozen/extrapolated
    T_flow) or literally tracks it (False, e.g. the fully-coupled solver's T_flow=T), which
    determines whether rho_f's T-dependence propagates into dhf_dT.

    Also returns `cp`, the constant-pressure specific heat (∂h/∂T)_p with the pressure-work
    contribution stripped out, so callers needing a pure thermal-capacity scale (e.g. SUPG's
    Peclet number) don't have to re-derive it from `dhf_dT`. For the nonlinear (CoolProp)
    branch, this is where that will come from once implemented, rather than each caller
    fetching `fluid["specific_heat"]` independently.
    """
    if fluid["density"] != 0:
        rho_0 = fluid["density"]
        T0 = fluid["reference_temperature"]
        p0 = fluid["reference_pressure"]
        alpha_f = fluid["therm_expansion"]
        cp_f = fluid["specific_heat"]
        Kf = fluid["bulk_mod"]

        rho_f = rho_0 * (1 - alpha_f * (T_flow - T0)) * (1 + (p - p0) / Kf)
        drho_dT = np.zeros(np.shape(T)) if T_flow_lagged else -rho_0 * alpha_f * np.ones(np.shape(T))
        drho_dp = rho_0 / Kf * np.ones(np.shape(p))

        u_f = cp_f * T - alpha_f * T * p / rho_0 + p**2 / (2 * Kf * rho_0)
        duf_dT = cp_f - alpha_f * p / rho_0
        duf_dp = p / (Kf * rho_0) - alpha_f * T / rho_0

        hf = u_f + p / rho_f
        dhf_dT = duf_dT - p / rho_f**2 * drho_dT
        dhf_dp = duf_dp + 1 / rho_f - p / rho_f**2 * drho_dp
        cp = cp_f * np.ones(np.shape(T))
    else:
        # Nonlinear internal energy
        hf = 0 * np.ones(np.shape(T))
        dhf_dT = 0 * np.ones(np.shape(T))
        dhf_dp = 0 * np.ones(np.shape(T))
        cp = 0 * np.ones(np.shape(T))
    return EnthalpyResult(hf, dhf_dp, dhf_dT, cp)


def get_internal_energy(
    T: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],  # noqa
    fluid: dict[str, Any],
) -> PropertyResult:
    """Get the fluid internal energy at a specified pressure and temperature.

    Integrated from du_f = (cp_f - alpha_f/rho_0 * p) dT + (p/(K_f*rho_0) - alpha_f/rho_0 * T) dp
    (the standard du = cp*dT + [T*(dv/dT)_p - p]*(-dv) identity, linearized at the reference
    density rho_0, matching get_density's own level of linearization). This is an exact
    potential given that linearization, so both partials below are its literal derivatives.
    """
    if fluid["density"] != 0:
        rho_0 = fluid["density"]
        alpha_f = fluid["therm_expansion"]
        cp_f = fluid["specific_heat"]
        Kf = fluid["bulk_mod"]

        u_f = cp_f * T - alpha_f * T * p / rho_0 + p**2 / (2 * Kf * rho_0)
        duf_dT = cp_f - alpha_f * p[..., 0:1] / rho_0
        duf_dp = p[..., 0:1] / (Kf * rho_0) - alpha_f * T[..., 0:1] / rho_0
    else:
        # Nonlinear internal energy
        u_f = 0 * np.ones(np.shape(T))
        duf_dT = 0 * np.ones(np.shape(T))
        duf_dp = 0 * np.ones(np.shape(T))
    return PropertyResult(u_f, duf_dp, duf_dT)


def get_effective_permeability(
    kf: npt.NDArray[np.float64] | float,
    T: npt.NDArray[np.float64],  # noqa
    p: npt.NDArray[np.float64],  # noqa
    fluid: dict[str, Any],  # noqa
    T_flow_lagged: bool = False,  # noqa
) -> PropertyResult:
    """Get the effective permeability. Apply Forchheimer flow and other modifiers."""
    kf_eff = np.asarray(kf, dtype=np.float64)
    dkf_dT = np.zeros(np.shape(T))
    dkf_dp = np.zeros(np.shape(T))

    return PropertyResult(kf_eff, dkf_dp, dkf_dT)
