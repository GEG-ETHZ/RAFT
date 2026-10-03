# Copyright 2026 The RAFT Community
#
# This file is part of RAFT.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
_RE_MIN = 1.0e-3  # Reynolds number floor for the friction factor; phi = f Re is 64 below it
_RE_LAMINAR = 2000.0  # upper bound of laminar heat transfer
_RE_TURBULENT = 2300.0  # lower bound of turbulent heat transfer (Gnielinski)
_NU_LAM_PIPE = 0.5 * (3.66 + 4.36)  # laminar pipe Nu, average of constant wall temperature and heat flux
_NU_LAM_PLATES = 0.5 * (7.54 + 8.23)  # laminar parallel plate Nu (D_h = 2w), same average

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

    # Determine if dT is wanted or not based on lagging.
    want_dT_eff = want_dT and not T_flow_lagged

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
        dmu_dT = np.empty(T.shape) if want_dT_eff else _const_array(0.0, T.shape)
        if want_dT_eff:
            dmu_dT = np.where(out_of_limits, 0.0, mu * (-B / T**2 + C + 2 * D * T))
        zeros = _const_array(0.0, T.shape)

        return PropertyResult(mu, zeros, dmu_dT)

    # Otherwise use CoolProp viscosity

    AS = fluid["abstract_state"]

    # CoolProp has no closed-form viscosity derivative, so dmu/dT is a forward difference.
    dT = 0.5  # K
    dp = 1  # Pa

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


@njit(cache=True)  # type: ignore[untyped-decorator]
def _eos_linear_kernel_values(
    T: npt.NDArray[np.float64],
    T_flow: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    rho_0: float,
    T0: float,
    p0: float,
    alpha: float,
    Kf: float,
    cp_f: float,
    T_flow_lagged: bool,
    need_density: bool,
    need_u: bool,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Numba kernel for the closed-form linear-density EOS branch of `get_EOS`.

    Returns two stacked bundles rather than 8 separate arrays: numba boxes a returned tuple
    one element at a time on *every* call regardless of caching, so an 8-way return here (this
    function is called millions of times) was measurably more expensive than the handful of
    microseconds its own arithmetic takes - bundling same-shaped outputs into one array each
    cuts that boxing cost ~4x. `_eos_linear_values` below unpacks the bundles and restores the
    `_EMPTY` convention for fields `need_density`/`need_u` didn't ask for.

    `main` stacks (rho_value, drho_dp, drho_dT, drho_dT_true, rho_true_value, u_value), all
    T's shape. `duf` stacks (duf_dT, duf_dp), `T[..., 0:1]`'s shape.
    """
    rho_value = np.zeros_like(T_flow)
    drho_dp = np.zeros_like(p)
    drho_dT = np.zeros_like(T_flow)
    drho_dT_true = np.zeros_like(T_flow)
    rho_true_value = np.zeros_like(T_flow)
    if need_density:
        rho_value = rho_0 * (1 - alpha * (T_flow - T0) + 1 / Kf * (p - p0))
        drho_dT_true = np.full(T_flow.shape, -rho_0 * alpha)
        drho_dT = np.zeros(T_flow.shape) if T_flow_lagged else drho_dT_true
        drho_dp = np.full(p.shape, rho_0 / Kf)
        rho_true_value = rho_value

    u_value = np.zeros_like(T)
    duf_dT = np.zeros_like(T[..., 0:1])
    duf_dp = np.zeros_like(T[..., 0:1])
    if need_u:
        u_value = cp_f * T - alpha * T * p / rho_0 + p**2 / (2 * Kf * rho_0)
        duf_dT = cp_f - alpha * p[..., 0:1] / rho_0
        duf_dp = p[..., 0:1] / (Kf * rho_0) - alpha * T[..., 0:1] / rho_0

    if need_density and need_u:
        # Density at the same point internal energy uses (T, not T_flow), for cp's exact fidelity.
        # Cheap either way (closed-form), so computed here whenever both are wanted, regardless
        # of whether `T` and `T_flow` actually differ (`T_flow_lagged`).
        rho_true_value = rho_0 * (1 - alpha * (T - T0)) * (1 + 1 / Kf * (p - p0))
        drho_dT_true = np.full(T.shape, -rho_0 * alpha)

    main = np.empty((6,) + T.shape)
    main[0] = rho_value
    main[1] = drho_dp
    main[2] = drho_dT
    main[3] = drho_dT_true
    main[4] = rho_true_value
    main[5] = u_value

    duf = np.empty((2,) + duf_dT.shape)
    duf[0] = duf_dT
    duf[1] = duf_dp

    return main, duf


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
    # `float(...)`: config values may parse as Python int (e.g. a YAML "4200" with no decimal
    # point) - numba treats int and float scalar args as different types, so an unnormalized
    # mix costs an extra compiled specialization the first time it's hit.
    rho_0 = float(fluid["density"])
    T0 = float(fluid["reference_temperature"])
    p0 = float(fluid["reference_pressure"])
    alpha = float(fluid["therm_expansion"])
    Kf = float(fluid["bulk_mod"])
    cp_f = float(fluid["specific_heat"])

    # Evaluated in numba kernel.
    main, duf = _eos_linear_kernel_values(
        T, T_flow, p, rho_0, T0, p0, alpha, Kf, cp_f, T_flow_lagged, need_density, need_u
    )
    rho_value, drho_dp, drho_dT, drho_dT_true, rho_true_value, u_value = main
    duf_dT, duf_dp = duf

    if not need_density:
        rho_value = drho_dp = drho_dT = drho_dT_true = rho_true_value = _EMPTY
    if not need_u:
        u_value = duf_dp = duf_dT = _EMPTY

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


def get_friction_factor(
    Re: npt.NDArray[np.float64], rel_roughness: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Churchill (1977) Darcy friction factor as phi = f Re and dphi/dRe.

    Smooth across laminar, transitional, and turbulent flow; phi -> 64 as Re -> 0.
    """
    Re = np.maximum(Re, _RE_MIN)
    r7 = (7.0 / Re) ** 0.9
    t = r7 + 0.27 * rel_roughness
    L = -np.log(t)
    A = (2.457 * L) ** 16  # turbulent term
    B = (37530.0 / Re) ** 16  # transition term
    P = Re**12 * (A + B) ** -1.5
    phi = 8.0 * (8.0**12 + P) ** (1.0 / 12.0)

    # Chain rule through L, A, B, and P
    dL = 0.9 * r7 / (Re * t)
    dA = 16.0 * 2.457 * (2.457 * L) ** 15 * dL
    dB = -16.0 * B / Re
    dP = P * (12.0 / Re - 1.5 * (dA + dB) / (A + B))
    dphi = phi / (12.0 * (8.0**12 + P)) * dP
    return phi, dphi


def _gnielinski_values(
    Re: npt.NDArray[np.float64], Pr: npt.NDArray[np.float64], rel_roughness: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Gnielinski turbulent Nu, with Churchill Darcy f, and its derivatives wrt Re and Pr."""
    phi, dphi = get_friction_factor(Re, rel_roughness)
    f = phi / Re
    df_dRe = (dphi - f) / Re

    # Nu = k (Re - 1000) Pr / D, with k = f/8 and D = 1 + 12.7 sqrt(k) (Pr^(2/3) - 1)
    k = f / 8.0
    s = np.sqrt(k)
    pr_term = Pr ** (2.0 / 3.0) - 1.0
    D = 1.0 + 12.7 * s * pr_term
    Nu = k * (Re - 1000.0) * Pr / D

    # Quotient rule through k(Re) and Pr
    dk_dRe = df_dRe / 8.0
    dD_dRe = 12.7 * pr_term * dk_dRe / (2.0 * s)
    dNu_dRe = ((dk_dRe * (Re - 1000.0) + k) * Pr - Nu * dD_dRe) / D
    dD_dPr = 12.7 * s * (2.0 / 3.0) * Pr ** (-1.0 / 3.0)
    dNu_dPr = (k * (Re - 1000.0) - Nu * dD_dPr) / D
    return Nu, dNu_dRe, dNu_dPr


def get_nusselt_number(
    Re: npt.NDArray[np.float64],
    Pr: npt.NDArray[np.float64],
    rel_roughness: float,
    *,
    parallel_plates: bool,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Nusselt number and its derivatives wrt Re and Pr, for pipes or parallel plates.

    Constant laminar Nu below Re = 2000, Gnielinski above Re = 2300, and log-interpolated between.
    """
    Re = np.asarray(Re, dtype=np.float64)
    Pr = np.asarray(Pr, dtype=np.float64)
    Nu_lam = _NU_LAM_PLATES if parallel_plates else _NU_LAM_PIPE

    # Turbulent: Gnielinski, evaluated with Re clipped into its range of validity
    Nu_turb, dturb_dRe, dturb_dPr = _gnielinski_values(np.maximum(Re, _RE_TURBULENT), Pr, rel_roughness)

    # Transition: log-interpolate from Nu_lam at Re = 2000 to Gnielinski at Re = 2300
    Nu_2300, _, d2300_dPr = _gnielinski_values(np.full_like(Re, _RE_TURBULENT), Pr, rel_roughness)
    log_span = np.log(_RE_TURBULENT / _RE_LAMINAR)
    w = np.log(np.clip(Re, _RE_LAMINAR, _RE_TURBULENT) / _RE_LAMINAR) / log_span
    Nu_trans = Nu_lam ** (1.0 - w) * Nu_2300**w
    dtrans_dRe = Nu_trans * np.log(Nu_2300 / Nu_lam) / (np.maximum(Re, _RE_MIN) * log_span)
    dtrans_dPr = Nu_trans * w * d2300_dPr / Nu_2300

    # Select the regime
    laminar = Re <= _RE_LAMINAR
    turbulent = Re >= _RE_TURBULENT
    Nu = np.where(laminar, Nu_lam, np.where(turbulent, Nu_turb, Nu_trans))
    dNu_dRe = np.where(laminar, 0.0, np.where(turbulent, dturb_dRe, dtrans_dRe))
    dNu_dPr = np.where(laminar, 0.0, np.where(turbulent, dturb_dPr, dtrans_dPr))
    return Nu, dNu_dRe, dNu_dPr
