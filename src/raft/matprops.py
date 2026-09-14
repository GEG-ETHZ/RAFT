"""All material property correlations are stored here."""

import functools
from dataclasses import dataclass
from typing import Any

import CoolProp.CoolProp as CP  # noqa: N817
import numpy as np
import numpy.typing as npt
from numba import njit  # type: ignore[import-untyped]

# Set clipping max and min to prevent errors during convergence
_P_MAX = 7e8
_P_MIN = 1e5
_T_MAX = 1.9e3
_T_MIN = 274


@functools.lru_cache(maxsize=64)
def _const_array(value: float, shape: tuple[int, ...]) -> npt.NDArray[np.float64]:
    """Cached constant array, reused across calls with the same (value, shape) pair.

    Constant array for 0 derivatives in functions. Avoids overhead when called millions
    of times in the Newton loop. Read-only: never mutate the returned array in place.
    """
    arr = np.full(shape, value)
    arr.flags.writeable = False
    return arr


@dataclass(frozen=True, slots=True)
class PropertyResult:
    """A material property value with its partial derivatives w.r.t. p and T.

    Always array-valued, even for constant-formula properties, so callers can slice/index
    freely without checking for a bare scalar. Plain data container, no arithmetic operators.
    """

    value: npt.NDArray[np.float64]  # Value of the property
    dp: npt.NDArray[np.float64]  # Derivative wrt pressure
    dT: npt.NDArray[np.float64]  # Derivative wrt temperature # noqa


@dataclass(frozen=True, slots=True)
class DensityResult(PropertyResult):
    """Density with an extra, never-lagged/zeroed temperature derivative for `cp`.

    `dT` follows the usual `T_flow_lagged` convention (zeroed when T isn't the primary
    unknown, for Jacobian consistency). `dT_true` is always the genuine dρ/dT at the
    point evaluated, needed by `get_enthalpy`'s `cp` regardless of that convention.
    """

    dT_true: npt.NDArray[np.float64]  # noqa: N815


@dataclass(frozen=True, slots=True)
class EnthalpyResult:
    """Fluid enthalpy and its partial derivatives, plus `cp`.

    cp treated separately as enthalpy derivative minus pressure contribution.
    cp needed for SUPG.
    """

    value: npt.NDArray[np.float64]  # Value of the property
    dp: npt.NDArray[np.float64]  # Derivative wrt pressure
    dT: npt.NDArray[np.float64]  # Derivative wrt temperature # noqa
    cp: npt.NDArray[np.float64]  # specific heat for SUPG calcs


@njit  # type: ignore[untyped-decorator]
def _clip_T(T_in: float) -> float:
    """Cap limits of T to prevent errors in unconverged intermediate states."""
    return max(min(T_in, _T_MAX), _T_MIN)


@njit  # type: ignore[untyped-decorator]
def _clip_p(p_in: float) -> float:
    """Cap limits of p to prevent errors in unconverged intermediate states."""
    return max(min(p_in, _P_MAX), _P_MIN)


def create_abstract_state(fluid_type: str) -> CP.AbstractState:
    """Create a CoolProp AbstractState for a fluid, to be reused across property lookups."""
    return CP.AbstractState("BICUBIC&HEOS", fluid_type)


def get_viscosity(
    T: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    fluid: dict[str, Any],
    T_flow_lagged: bool = False,
    want_dp: bool = False,
    want_dT: bool = False,
) -> PropertyResult:
    """Get viscosity and its T/p derivatives, at temperature T [K] and pressure p [Pa].

    `fluid["viscosity"]` of 0 looks it up from CoolProp instead of treating it as a constant.
    `T_flow_lagged` zeroes dmu_dT when T is not the primary unknown being solved for.
    """
    viscosity_coefficients = fluid["viscosity"]

    if viscosity_coefficients != 0:
        mu = np.full(T.shape, viscosity_coefficients)
        zeros = _const_array(0.0, T.shape)
        return PropertyResult(mu, zeros, zeros)

    AS = fluid["abstract_state"]

    # CoolProp has no closed-form viscosity derivative, so dmu/dT is a central difference.
    dT = 0.5  # K
    dp = 1  # Pa
    want_dT_eff = want_dT and not T_flow_lagged

    mu = np.empty(T.shape)
    dmu_dp = np.empty(T.shape) if want_dp else _const_array(0.0, T.shape)
    dmu_dT = np.empty(T.shape) if want_dT_eff else _const_array(0.0, T.shape)

    for i in range(T.size):
        T_i = _clip_T(T.flat[i])
        p_i = _clip_p(p.flat[i])
        AS.update(CP.PT_INPUTS, p_i, T_i)
        mu.flat[i] = AS.viscosity()

        if want_dp:
            AS.update(CP.PT_INPUTS, p_i + dp, T_i)
            mu_plus = AS.viscosity()
            AS.update(CP.PT_INPUTS, p_i - dp, T_i)
            mu_minus = AS.viscosity()
            dmu_dp.flat[i] = (mu_plus - mu_minus) / (2 * dp)

        if want_dT_eff:
            AS.update(CP.PT_INPUTS, p_i, T_i + dT)
            mu_plus = AS.viscosity()
            AS.update(CP.PT_INPUTS, p_i, T_i - dT)
            mu_minus = AS.viscosity()
            dmu_dT.flat[i] = (mu_plus - mu_minus) / (2 * dT)

    return PropertyResult(mu, dmu_dp, dmu_dT)


def get_density(
    T: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],  # noqa
    fluid: dict[str, Any],
    T_flow_lagged: bool = False,
    want_dp: bool = False,
    want_dT: bool = False,
) -> DensityResult:
    """Get density and its T/p derivatives, at temperature T [K] and pressure p [Pa].

    `fluid["density"]` of 0 looks it up from CoolProp instead of treating it as a constant
    reference density. `T_flow_lagged` zeroes `dT` when T is not the primary unknown; `dT_true`
    is always the real derivative at the point evaluated (needed by `get_enthalpy`'s `cp`).
    """
    if fluid["density"] != 0:
        T0 = fluid["reference_temperature"]
        p0 = fluid["reference_pressure"]
        alpha = fluid["therm_expansion"]
        rho_0 = fluid["density"]
        Kf = fluid["bulk_mod"]
        density = rho_0 * (1 - alpha * (T - T0)) * (1 + 1 / Kf * (p - p0))

        dT_true = _const_array(-rho_0 * alpha, T.shape)
        drho_dT = _const_array(0.0, T.shape) if T_flow_lagged else dT_true
        drho_dp = _const_array(rho_0 / Kf, p.shape)

        return DensityResult(density, drho_dp, drho_dT, dT_true)

    AS = fluid["abstract_state"]

    rho = np.empty(T.shape)
    drho_dp = np.empty(T.shape) if want_dp else _const_array(0.0, T.shape)
    dT_true = np.empty(T.shape) if want_dT else _const_array(0.0, T.shape)

    for i in range(T.size):
        T_i = _clip_T(T.flat[i])
        p_i = _clip_p(p.flat[i])
        AS.update(CP.PT_INPUTS, p_i, T_i)
        rho.flat[i] = AS.rhomass()
        if want_dp:
            drho_dp.flat[i] = AS.first_partial_deriv(CP.iDmass, CP.iP, CP.iT)
        if want_dT:
            dT_true.flat[i] = AS.first_partial_deriv(CP.iDmass, CP.iT, CP.iP)

    drho_dT = _const_array(0.0, T.shape) if T_flow_lagged else dT_true
    return DensityResult(rho, drho_dp, drho_dT, dT_true)


def get_porosity(
    p: npt.NDArray[np.float64],
    p_0: npt.NDArray[np.float64],
    phi_0: float,
    K_d: float,
) -> PropertyResult:
    """Get the current porosity based on the undeformed reference state.

    Assume alpha_m = alpha_s and temperature therefore does not affect porosity.
    """
    phi = 1 - (1 - phi_0) * np.exp(-(p - p_0) / K_d)
    dphi_dT = _const_array(0.0, phi.shape)
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

    Non-volumetric sources are neglected. `T_flow_lagged` zeroes dstrain_dT when T is not the
    primary unknown. `volumetric_strain_on=False` forces zero strain, e.g. for initial conditions.
    """
    if not volumetric_strain_on:
        zeros = _const_array(0.0, np.shape(p))
        return PropertyResult(zeros, zeros, zeros)

    strain = biot / K_d * (p - p_0) + therm_expansion * (T - T_0)
    dstrain_dT = _const_array(0.0 if T_flow_lagged else therm_expansion, strain.shape)
    dstrain_dp = _const_array(biot / K_d, strain.shape)

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
    rho_f: DensityResult,
    u_f: PropertyResult,
    p: npt.NDArray[np.float64],
    *,
    want_dp: bool = False,
    want_dT: bool = False,
    want_cp: bool = False,
) -> EnthalpyResult:
    """Get fluid enthalpy from already-evaluated density and internal energy.

    hf = u_f + p / rho_f. Takes `rho_f`/`u_f` as inputs (rather than evaluating them from
    T/p itself) so callers can reuse a single density/internal-energy evaluation across
    enthalpy, storage, and flux terms instead of re-deriving them per term.

    `rho_f.dT` may follow a `T_flow_lagged` convention (zeroed when T isn't the primary
    unknown) for `dhf_dT`'s sake; `cp`, (∂h/∂T)_p with the pressure-work term stripped out
    (e.g. for SUPG's Peclet number), instead always uses `rho_f.dT_true`, since `cp` is a
    physical quantity independent of that Jacobian convention. Set `want_cp=True` only when
    the caller actually uses `.cp` (it requires `u_f.dT`/`rho_f.dT_true` to have been computed).
    """
    hf = u_f.value + p / rho_f.value
    dhf_dT = np.zeros(hf.shape)
    dhf_dp = np.zeros(hf.shape)
    if want_dT:
        dhf_dT = u_f.dT - p / rho_f.value**2 * rho_f.dT
    if want_dp:
        dhf_dp = u_f.dp + 1 / rho_f.value - p / rho_f.value**2 * rho_f.dp

    cp = np.zeros(hf.shape)
    if want_cp:
        cp = u_f.dT - p / rho_f.value**2 * rho_f.dT_true

    return EnthalpyResult(hf, dhf_dp, dhf_dT, cp)


def get_internal_energy(
    T: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],  # noqa
    fluid: dict[str, Any],
    want_dp: bool = False,
    want_dT: bool = False,
) -> PropertyResult:
    """Get the fluid internal energy at a specified pressure and temperature.

    Integrated from du_f = (cp_f - alpha_f/rho_0 * p) dT + (p/(K_f*rho_0) - alpha_f/rho_0 * T) dp,
    linearized at the reference density rho_0 to match get_density.
    Density is used as the sentinel evaluate u_f directly from CoolProp at (T, p).
    """
    if fluid["density"] != 0:
        rho_0 = fluid["density"]
        alpha_f = fluid["therm_expansion"]
        cp_f = fluid["specific_heat"]
        Kf = fluid["bulk_mod"]

        u_f = cp_f * T - alpha_f * T * p / rho_0 + p**2 / (2 * Kf * rho_0)
        duf_dT = cp_f - alpha_f * p[..., 0:1] / rho_0
        duf_dp = p[..., 0:1] / (Kf * rho_0) - alpha_f * T[..., 0:1] / rho_0
        return PropertyResult(u_f, duf_dp, duf_dT)

    AS = fluid["abstract_state"]

    u_f = np.empty(T.shape)
    duf_dT = np.empty(T.shape) if want_dT else _const_array(0.0, T.shape)
    duf_dp = np.empty(T.shape) if want_dp else _const_array(0.0, T.shape)

    for i in range(T.size):
        T_i = _clip_T(T.flat[i])
        p_i = _clip_p(p.flat[i])
        AS.update(CP.PT_INPUTS, p_i, T_i)
        u_f.flat[i] = AS.umass()
        if want_dT:
            duf_dT.flat[i] = AS.first_partial_deriv(CP.iUmass, CP.iT, CP.iP)
        if want_dp:
            duf_dp.flat[i] = AS.first_partial_deriv(CP.iUmass, CP.iP, CP.iT)

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
    dkf_dT = _const_array(0.0, T.shape)
    dkf_dp = _const_array(0.0, T.shape)

    return PropertyResult(kf_eff, dkf_dp, dkf_dT)
