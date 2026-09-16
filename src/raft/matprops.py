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

# Sentinel for an optional input not needed unless explicitly requested (e.g. `get_enthalpy`'s `cp`).
_EMPTY = np.empty((0, 0))


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
class EnthalpyResult:
    """Fluid enthalpy and its partial derivatives, plus `cp`.

    cp treated separately as enthalpy derivative minus pressure contribution.
    cp needed for SUPG.
    """

    value: npt.NDArray[np.float64]  # Value of the property
    dp: npt.NDArray[np.float64]  # Derivative wrt pressure
    dT: npt.NDArray[np.float64]  # Derivative wrt temperature # noqa
    cp: npt.NDArray[np.float64]  # specific heat for SUPG calcs


@njit(cache=True)  # type: ignore[untyped-decorator]
def _clip_T(T_in: float) -> float:
    """Cap limits of T to prevent errors in unconverged intermediate states."""
    return max(min(T_in, _T_MAX), _T_MIN)


@njit(cache=True)  # type: ignore[untyped-decorator]
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

    `fluid["viscosity"]` = 0 looks it up from CoolProp instead of treating it as a constant.
    `T_flow_lagged` zeroes dmu_dT when T is not the primary unknown being solved for.

    `fluid["viscosity"]` = -1 uses a 4 coefficient temperature-only approximation.
    """
    mu0 = fluid["viscosity"]

    if mu0 > 0:
        # Constant Viscosity
        mu = np.full(T.shape, mu0)
        zeros = _const_array(0.0, T.shape)
        return PropertyResult(mu, zeros, zeros)

    if mu0 == -1:
        # Get visocsity using a 4 coefficient approximation.
        A = 1.856e-14  # Pa-s
        B = 4209  # K
        C = 0.04527  # K^-1
        D = -3.376e-5  # K^-2

        # Cap limits on temperature
        low_limit = 200.0
        up_limit = 1.0e3

        T = np.maximum(T, _T_MIN)
        T = np.minimum(T, _T_MAX)

        mu = A * np.exp(B / T + C * T + D * T**2)

        out_of_limits = (low_limit >= T) | (up_limit <= T)
        dmu_dT = np.where(out_of_limits, 0.0, mu * (-B / T**2 + C + 2 * D * T))
        zeros = _const_array(0.0, T.shape)

        return PropertyResult(mu, zeros, dmu_dT)

    # Otherwise use CoolProp viscosity

    AS = fluid["abstract_state"]

    # CoolProp has no closed-form viscosity derivative, so dmu/dT is a forward difference.
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
        mu_i = AS.viscosity()
        mu.flat[i] = mu_i

        if want_dp:
            AS.update(CP.PT_INPUTS, p_i + dp, T_i)
            mu_plus = AS.viscosity()
            dmu_dp.flat[i] = (mu_plus - mu_i) / dp

        if want_dT_eff:
            AS.update(CP.PT_INPUTS, p_i, T_i + dT)
            mu_plus = AS.viscosity()
            dmu_dT.flat[i] = (mu_plus - mu_i) / dT

    return PropertyResult(mu, dmu_dp, dmu_dT)


def _eos_linear_values(
    T: npt.NDArray[np.float64],
    T_flow: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    fluid: dict[str, Any],
    T_flow_lagged: bool,
    need_density: bool,
    need_u: bool,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Closed-form (non-CoolProp) `get_EOS` branch for a linear density fluid model."""
    rho_0 = fluid["density"]
    T0 = fluid["reference_temperature"]
    p0 = fluid["reference_pressure"]
    alpha = fluid["therm_expansion"]
    Kf = fluid["bulk_mod"]
    cp_f = fluid["specific_heat"]

    rho_value = drho_dp = drho_dT = drho_dT_true = rho_true_value = _EMPTY
    if need_density:
        rho_value = rho_0 * (1 - alpha * (T_flow - T0) + 1 / Kf * (p - p0))
        drho_dT_true = _const_array(-rho_0 * alpha, T_flow.shape)
        drho_dT = _const_array(0.0, T_flow.shape) if T_flow_lagged else drho_dT_true
        drho_dp = _const_array(rho_0 / Kf, p.shape)
        rho_true_value = rho_value

    u_value = duf_dp = duf_dT = _EMPTY
    if need_u:
        u_value = cp_f * T - alpha * T * p / rho_0 + p**2 / (2 * Kf * rho_0)
        duf_dT = cp_f - alpha * p[..., 0:1] / rho_0
        duf_dp = p[..., 0:1] / (Kf * rho_0) - alpha * T[..., 0:1] / rho_0

    if need_density and need_u:
        # Density at the same point internal energy uses (T, not T_flow), for cp's exact fidelity.
        # Cheap either way (closed-form), so computed here whenever both are wanted, regardless
        # of whether `T` and `T_flow` actually differ (`T_flow_lagged`).
        rho_true_value = rho_0 * (1 - alpha * (T - T0)) * (1 + 1 / Kf * (p - p0))
        drho_dT_true = _const_array(-rho_0 * alpha, T.shape)

    return rho_value, drho_dp, drho_dT, drho_dT_true, u_value, duf_dp, duf_dT, rho_true_value


def _eos_coolprop_values(  # noqa
    T: npt.NDArray[np.float64],
    T_flow: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    fluid: dict[str, Any],
    T_flow_lagged: bool,
    need_density: bool,
    need_u: bool,
    want_dp: bool,
    want_dT: bool,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """CoolProp `get_EOS` branch: optimized for `AbstractState` update only when actually needed.

    Density is read at `T_flow`.
    Internal energy at `T`.
    In the linearized solver, T != T_flow. In the fully_coupled solver, T = T_flow.
    """
    AS = fluid["abstract_state"]
    same_point = not T_flow_lagged

    # Initialize containers
    rho_value = np.empty(T_flow.shape) if need_density else _EMPTY
    drho_dp = np.empty(T_flow.shape) if need_density and want_dp else _const_array(0.0, T_flow.shape)
    drho_dT_true = np.empty(T_flow.shape) if need_density and want_dT else _const_array(0.0, T_flow.shape)

    u_value = np.empty(T.shape) if need_u else _EMPTY
    duf_dT = np.empty(T.shape) if need_u and want_dT else _const_array(0.0, T.shape)
    duf_dp = np.empty(T.shape) if need_u and want_dp else _const_array(0.0, T.shape)

    # Determine if drho_dT at T is required for evaluating cp
    need_rho_true_at_u_point = need_density and need_u and not same_point
    rho_true_value = np.empty(T.shape) if need_rho_true_at_u_point else rho_value
    drho_true_dT = np.empty(T.shape) if need_rho_true_at_u_point and want_dT else _const_array(0.0, T.shape)

    for i in range(T.size):
        if need_density:
            T_flow_i = _clip_T(T_flow.flat[i])
            p_i = _clip_p(p.flat[i])
            AS.update(CP.PT_INPUTS, p_i, T_flow_i)
            rho_value.flat[i] = AS.rhomass()
            if want_dp:
                drho_dp.flat[i] = AS.first_partial_deriv(CP.iDmass, CP.iP, CP.iT)
            if want_dT:
                drho_dT_true.flat[i] = AS.first_partial_deriv(CP.iDmass, CP.iT, CP.iP)

        if need_u:
            if not (need_density and same_point):
                T_i = _clip_T(T.flat[i])
                p_i = _clip_p(p.flat[i])
                AS.update(CP.PT_INPUTS, p_i, T_i)
            # else: `AS` is already at (p_i, T_flow_i) == (p_i, T_i) from the density read above.
            u_value.flat[i] = AS.umass()
            if want_dT:
                duf_dT.flat[i] = AS.first_partial_deriv(CP.iUmass, CP.iT, CP.iP)
            if want_dp:
                duf_dp.flat[i] = AS.first_partial_deriv(CP.iUmass, CP.iP, CP.iT)

            if need_rho_true_at_u_point:
                rho_true_value.flat[i] = AS.rhomass()
                if want_dT:
                    drho_true_dT.flat[i] = AS.first_partial_deriv(CP.iDmass, CP.iT, CP.iP)

    if need_rho_true_at_u_point:
        drho_dT_true = drho_true_dT
    drho_dT = _const_array(0.0, T_flow.shape) if T_flow_lagged else drho_dT_true

    return rho_value, drho_dp, drho_dT, drho_dT_true, u_value, duf_dp, duf_dT, rho_true_value


def get_EOS(  # noqa: N802
    T: npt.NDArray[np.float64],
    T_flow: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    fluid: dict[str, Any],
    *,
    T_flow_lagged: bool = False,
    want_density: bool = False,
    want_internal_energy: bool = False,
    want_enthalpy: bool = False,
    want_dp: bool = False,
    want_dT: bool = False,
) -> tuple[
    PropertyResult,  # rho, at T_flow; `_EMPTY`s unless `want_density`
    PropertyResult,  # u, at T; `_EMPTY`s unless `want_internal_energy`
    EnthalpyResult,  # hf, with `cp`; `_EMPTY`s unless `want_enthalpy`
]:
    """Get density, internal energy, and/or enthalpy and their derivatives.

    Density is evaluated at `T_flow`. Enternal energy is evaluated at `T`.
    Enthalpy is reconstructructed from these base values as hf = u + p / rho,
    resulting in a mixed formulation for enthalpy.

    The enthalpy return also returns cp stripped of its pressure contribution
    for use in the SUPG formulation.
    """
    need_density = want_density or want_enthalpy
    need_u = want_internal_energy or want_enthalpy

    if fluid["density"] != 0:  # Density = 0 is the sentinel.
        # Use linearized EOS
        rho_value, drho_dp, drho_dT, drho_dT_true, u_value, duf_dp, duf_dT, rho_true_value = (
            _eos_linear_values(T, T_flow, p, fluid, T_flow_lagged, need_density, need_u)
        )
    else:
        # Use real EOS from CoolProp
        rho_value, drho_dp, drho_dT, drho_dT_true, u_value, duf_dp, duf_dT, rho_true_value = (
            _eos_coolprop_values(T, T_flow, p, fluid, T_flow_lagged, need_density, need_u, want_dp, want_dT)
        )

    # Distribute results to storage dataclasses.
    rho = PropertyResult(_EMPTY, _EMPTY, _EMPTY)
    if want_density:
        rho = PropertyResult(rho_value, drho_dp, drho_dT)

    u = PropertyResult(_EMPTY, _EMPTY, _EMPTY)
    if want_internal_energy:
        u = PropertyResult(u_value, duf_dp, duf_dT)

    hf = EnthalpyResult(_EMPTY, _EMPTY, _EMPTY, _EMPTY)
    if want_enthalpy:
        rho_for_hf = PropertyResult(rho_value, drho_dp, drho_dT)
        u_for_hf = PropertyResult(u_value, duf_dp, duf_dT)
        hf = get_enthalpy(
            rho_for_hf,
            u_for_hf,
            p,
            rho_true_value=rho_true_value,
            rho_dT_true=drho_dT_true,
            want_dp=want_dp,
            want_dT=want_dT,
            want_cp=True,
        )

    return rho, u, hf


@njit(cache=True)  # type: ignore[untyped-decorator]
def _porosity_values(
    p: npt.NDArray[np.float64],
    p_0: npt.NDArray[np.float64],
    phi_0: float,
    K_d: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Porosity evaluation kernel.

    Assume alpha_m = alpha_s and temperature therefore does not affect porosity.
    """
    phi = 1.0 - (1.0 - phi_0) * np.exp(-(p - p_0) / K_d)
    dphi_dp = -(phi - 1.0) / K_d
    return phi, dphi_dp


def get_porosity(
    p: npt.NDArray[np.float64],
    p_0: npt.NDArray[np.float64],
    phi_0: float,
    K_d: float,
) -> PropertyResult:
    """Get the current porosity based on the undeformed reference state."""
    phi, dphi_dp = _porosity_values(p, p_0, phi_0, K_d)
    dphi_dT = _const_array(0.0, phi.shape)

    return PropertyResult(phi, dphi_dp, dphi_dT)


@njit(cache=True)  # type: ignore[untyped-decorator]
def _volumetric_strain_values(
    p: npt.NDArray[np.float64],
    p_0: npt.NDArray[np.float64],
    T: npt.NDArray[np.float64],
    T_0: npt.NDArray[np.float64],
    K_d: float,
    biot: float,
    therm_expansion: float,
    T_flow_lagged: bool,
    volumetric_strain_on: bool,
) -> tuple[npt.NDArray[np.float64], float, float]:
    """Volumetric strain kernel."""
    if volumetric_strain_on:
        strain = biot / K_d * (p - p_0) + therm_expansion * (T - T_0)
        dstrain_dp = biot / K_d
        dstrain_dT = 0.0 if T_flow_lagged else therm_expansion
    else:
        strain = np.zeros_like(p)
        dstrain_dp = 0.0
        dstrain_dT = 0.0
    return strain, dstrain_dp, dstrain_dT


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
    strain, dstrain_dp, dstrain_dT = _volumetric_strain_values(
        p, p_0, T, T_0, K_d, biot, therm_expansion, T_flow_lagged, volumetric_strain_on
    )
    return PropertyResult(
        strain, _const_array(dstrain_dp, strain.shape), _const_array(dstrain_dT, strain.shape)
    )


@njit(cache=True)  # type: ignore[untyped-decorator]
def _porosity_lagrangian_values(
    p: npt.NDArray[np.float64],
    p_0: npt.NDArray[np.float64],
    T: npt.NDArray[np.float64],
    T_0: npt.NDArray[np.float64],
    phi_0: float,
    K_d: float,
    biot: float,
    therm_expansion: float,
    T_flow_lagged: bool,
    volumetric_strain_on: bool,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Kernel for lagrangian porosity: phi_L = phi * (1 + volumetric strain)."""
    phi, dphi_dp = _porosity_values(p, p_0, phi_0, K_d)
    strain, dstrain_dp, dstrain_dT = _volumetric_strain_values(
        p, p_0, T, T_0, K_d, biot, therm_expansion, T_flow_lagged, volumetric_strain_on
    )

    one_plus_strain = 1.0 + strain
    phi_L = phi * one_plus_strain
    dphi_L_dp = dphi_dp * one_plus_strain + phi * dstrain_dp
    dphi_L_dT = phi * dstrain_dT  # d_phi_dT = 0 assuming alpha_m = alpha_s

    return phi_L, dphi_L_dp, dphi_L_dT


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
    """Get the Lagrangian porosity which includes the volumetric strain in the deformed state.

    Non-volumetric strain sources are neglected. `T_flow_lagged` zeroes dphi_L/dT when T is
    not the primary unknown. `volumetric_strain_on=False` forces zero strain.
    """
    phi_L, dphi_L_dp, dphi_L_dT = _porosity_lagrangian_values(
        p, p_0, T, T_0, phi_0, K_d, biot, therm_expansion, T_flow_lagged, volumetric_strain_on
    )
    return PropertyResult(phi_L, dphi_L_dp, dphi_L_dT)


@njit(cache=True)  # type: ignore[untyped-decorator]
def _enthalpy_values(
    rho_value: npt.NDArray[np.float64],
    rho_dp: npt.NDArray[np.float64],
    rho_dT: npt.NDArray[np.float64],
    u_value: npt.NDArray[np.float64],
    u_dp: npt.NDArray[np.float64],
    u_dT: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    want_dp: bool,
    want_dT: bool,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Numba kernel for evaluating enthalpy: hf = u_f + p / rho_f."""
    hf = u_value + p / rho_value
    dhf_dT = np.zeros_like(hf)
    dhf_dp = np.zeros_like(hf)
    if want_dT:
        dhf_dT = u_dT - p / rho_value**2 * rho_dT
    if want_dp:
        dhf_dp = u_dp + 1 / rho_value - p / rho_value**2 * rho_dp
    return hf, dhf_dp, dhf_dT


@njit(cache=True)  # type: ignore[untyped-decorator]
def _enthalpy_cp_values(
    rho_true_value: npt.NDArray[np.float64],
    u_dT: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    rho_dT_true: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """cp, (dh/dT)_p with the pressure-work term stripped out for SUPG's Peclet number."""
    return u_dT - p / rho_true_value**2 * rho_dT_true


def get_enthalpy(
    rho_f: PropertyResult,
    u_f: PropertyResult,
    p: npt.NDArray[np.float64],
    *,
    rho_true_value: npt.NDArray[np.float64] = _EMPTY,
    rho_dT_true: npt.NDArray[np.float64] = _EMPTY,
    want_dp: bool = False,
    want_dT: bool = False,
    want_cp: bool = False,
) -> EnthalpyResult:
    """Get fluid enthalpy from already-evaluated density and internal energy.

    hf = u_f + p / rho_f.
    """
    # Evaluated in numba kernel.
    hf, dhf_dp, dhf_dT = _enthalpy_values(
        rho_f.value, rho_f.dp, rho_f.dT, u_f.value, u_f.dp, u_f.dT, p, want_dp, want_dT
    )

    cp = np.zeros(hf.shape)
    if want_cp:
        cp = _enthalpy_cp_values(rho_true_value, u_f.dT, p, rho_dT_true)

    return EnthalpyResult(hf, dhf_dp, dhf_dT, cp)


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
