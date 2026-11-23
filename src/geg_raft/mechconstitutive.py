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

"""Constitutive laws of the solid: matrix elasticity, fracture contact, fracture cohesion, and fracture
aperture, returning values with their derivatives where the solve needs a tangent.
"""

import numpy as np
import numpy.typing as npt
from scipy.special import expit

from .mechanicsdata import CohesiveParameters

COHESIVE_LAWS = ("linear", "bilinear", "cubic")
# unloading leaves the envelope below this fraction of the maximum effective displacement
HISTORY_MARGIN = 0.95


def elasticity_matrix(
    E: float, nu: float, nsd: int, plane: str, axisymmetric: bool
) -> npt.NDArray[np.float64]:
    """Isotropic elasticity matrix D (Voigt form of the elasticity tensor C)."""
    if nsd == 3:
        # 3D matrix
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        G = E / (2.0 * (1.0 + nu))
        D = np.zeros((6, 6))
        D[:3, :3] = lam
        D[[0, 1, 2], [0, 1, 2]] = lam + 2.0 * G
        D[[3, 4, 5], [3, 4, 5]] = G
        return D
    if axisymmetric:
        # axisymmetric matrix with 4 stress components
        c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return c * np.array(
            [
                [1.0 - nu, nu, 0.0, nu],
                [nu, 1.0 - nu, 0.0, nu],
                [0.0, 0.0, 0.5 - nu, 0.0],
                [nu, nu, 0.0, 1.0 - nu],
            ]
        )
    if plane == "stress":
        # plane stress
        c = E / (1.0 - nu**2)
        return c * np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]])
    # plane strain
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return c * np.array([[1.0 - nu, nu, 0.0], [nu, 1.0 - nu, 0.0], [0.0, 0.0, 0.5 - nu]])


def smooth_bracket(
    x: npt.NDArray[np.float64], ell_g: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Positive part <x> and its slope: ell_g ln(1 + exp(x / ell_g)) where the smoothing length ell_g > 0,
    and exact, with slope 1 at x = 0, where ell_g = 0.
    """
    smooth = ell_g > 0.0
    ell = np.where(smooth, ell_g, 1.0)
    value = np.where(smooth, ell * np.logaddexp(0.0, x / ell), np.maximum(x, 0.0))
    slope = np.where(smooth, expit(x / ell), (x >= 0.0).astype(np.float64))
    return value, slope


def penalty_contact(
    g_n: npt.NDArray[np.float64], k_N: npt.NDArray[np.float64], ell_g: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Normal contact pressure sigma_c = k_N <-g_n> (compression positive) and d(sigma_c)/d(g_n), with
    the bracket smoothed over the length ell_g where it is positive. Closed at g_n = 0.
    """
    closure, slope = smooth_bracket(-g_n, ell_g)
    return k_N * closure, -k_N * slope


def cohesive_softening_slope(
    law: str, f_u: npt.NDArray[np.float64], w_c: npt.NDArray[np.float64], delta_p: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Magnitude of the steepest envelope slope past the peak, |k_c|_max; zero for the linear law."""
    if law == "cubic":
        return 2.25 * f_u / w_c  # at delta = 2 w_c / 3
    if law == "bilinear":
        return f_u / (w_c - delta_p)
    return np.zeros_like(f_u)


def cohesive_scales(
    law: str, f_u: npt.NDArray[np.float64], G_c: npt.NDArray[np.float64], weakening_ratio: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Critical effective displacement w_c and peak displacement delta_p of an envelope dissipating G_c."""
    if law == "cubic":
        w_c = 16.0 * G_c / (9.0 * f_u)
        return w_c, w_c / 3.0
    w_c = 2.0 * G_c / f_u
    if law == "bilinear":
        return w_c, weakening_ratio * w_c
    return w_c, w_c.copy()


def cohesive_envelope(
    delta: npt.NDArray[np.float64], params: CohesiveParameters
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Envelope traction t_e(delta), its slope t_e', and the secant t_e / delta (K_0 at delta = 0).

    linear: t_e = f_u delta / w_c, then 0 from w_c
    bilinear: t_e = f_u delta / w_w up to w_w = a w_c, then f_u (w_c - delta) / (w_c - w_w)
    cubic: t_e = 27/4 f_u x (1 - x)^2, x = delta / w_c
    """
    f_u, w_c = params.f_u, params.w_c
    bonded = delta < w_c
    if params.law == "cubic":
        # Cubic cohesive law - Tvegaard 1990
        x = np.minimum(delta / w_c, 1.0)
        t_e = 27.0 / 4.0 * f_u * x * (1.0 - x) ** 2
        slope = np.where(bonded, 27.0 / 4.0 * f_u / w_c * (1.0 - x) * (1.0 - 3.0 * x), 0.0)
        # secant from the law directly, so delta = 0 gives K_0 rather than 0 / 0
        secant = 27.0 / 4.0 * f_u / w_c * (1.0 - x) ** 2
        return t_e, slope, secant

    if params.law == "bilinear":
        # linear rise to the peak at w_w, then softening to w_c
        w_w = params.weakening_ratio * w_c
        rising = delta < w_w
        k_rise = f_u / w_w
        k_soft = f_u / (w_c - w_w)
        t_e = np.where(rising, k_rise * delta, np.where(bonded, k_soft * (w_c - delta), 0.0))
        slope = np.where(rising, k_rise, np.where(bonded, -k_soft, 0.0))
        # softening secant; delta >= w_w > 0 wherever it is used
        delta_soft = np.where(rising, w_w, delta)
        secant_soft = np.where(bonded, k_soft * (w_c - delta_soft) / delta_soft, 0.0)
        secant = np.where(rising, k_rise, secant_soft)
        return t_e, slope, secant

    # linear rise to f_u at w_c, then debonded
    secant = np.where(bonded, f_u / w_c, 0.0)
    t_e = secant * delta
    slope = secant
    return t_e, slope, secant


def cohesive_damage(
    delta_max: npt.NDArray[np.float64], params: CohesiveParameters
) -> npt.NDArray[np.float64]:
    """Damage D = 1 - (t_e / delta)(h) / (t_e / delta)(delta_p) of the unloading point h = HISTORY_MARGIN
    delta_max: 0 before the peak, 1 from w_c.
    """
    h = HISTORY_MARGIN * delta_max
    _, _, secant = cohesive_envelope(h, params)
    _, _, secant_p = cohesive_envelope(params.delta_p, params)
    damage = np.where(h > params.delta_p, 1.0 - secant / np.where(secant_p > 0.0, secant_p, 1.0), 0.0)
    return np.where(h >= params.w_c, 1.0, damage)


def cohesive_initial_history(
    D_0: npt.NDArray[np.float64], params: CohesiveParameters
) -> npt.NDArray[np.float64]:
    """Maximum effective displacement delta_max of initial damage D_0, the inverse of `cohesive_damage`.

    D_0 = 0 gives delta_max = 0, the intact envelope.
    """
    w_c = params.w_c
    if params.law == "cubic":
        delta_max = w_c * (1.0 - 2.0 / 3.0 * np.sqrt(1.0 - D_0))

    elif params.law == "bilinear":
        w_w = params.weakening_ratio * w_c
        delta_max = w_w * w_c / (w_w + (1.0 - D_0) * (w_c - w_w))

    else:
        # linear: intact or debonded
        delta_max = D_0 * w_c
    return np.where(D_0 > 0.0, delta_max / HISTORY_MARGIN, 0.0)


def _outer(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Outer product a_q b_q^T at each point q: (nq, n), (nq, m) -> (nq, n, m)."""
    nq = a.shape[0]
    product: npt.NDArray[np.float64] = a.reshape(nq, -1, 1) * b.reshape(nq, 1, -1)
    return product


def _damage_driver(
    delta: npt.NDArray[np.float64], h: npt.NDArray[np.float64], w_eps: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Damage driver kappa = max(h, delta), smoothed in a band of half-width w_eps about the unloading
    point h, and d(kappa)/d(delta).
    """
    dx = delta - h  # offset from the unloading point
    unloading = dx <= -w_eps
    in_band = np.abs(dx) < w_eps

    # loading: kappa follows delta
    kappa = delta.copy()
    dkappa = np.ones_like(delta)

    # unloading: kappa held at h
    kappa[unloading] = h[unloading]
    dkappa[unloading] = 0.0

    # band: quadratic blend from h to delta
    s = dx[in_band] + w_eps[in_band]
    kappa[in_band] = h[in_band] + s**2 / (4.0 * w_eps[in_band])
    dkappa[in_band] = s / (2.0 * w_eps[in_band])
    return kappa, dkappa


def cohesive_traction(
    jump: npt.NDArray[np.float64],
    normal: npt.NDArray[np.float64],
    delta_max_c: npt.NDArray[np.float64],
    params: CohesiveParameters,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Cohesive traction t_coh (nq, nsd), its tangent d(t_coh)/d([[u]]) (nq, nsd, nsd), and the effective
    displacement delta (nq,) at the displacement jumps (nq, nsd).

    t_coh = t / delta (<g_n> n + gamma g_t), delta = sqrt(<g_n>^2 + g_t . g_t), from the envelope before
    the peak and from the secant t_e(kappa) / kappa delta after it, with kappa = max(h, delta) smoothed in a
    band of half-width w_eps about the unloading point h = HISTORY_MARGIN delta_max_c.
    """
    nq, nsd = normal.shape
    gamma = params.gamma

    # Effective displacement delta = sqrt(<g_n>^2 + g_t . g_t)
    g_n = np.sum(jump * normal, axis=1)  # (nq,)
    g_t = jump - g_n.reshape(nq, 1) * normal  # (nq, nsd)
    g_n_open = np.maximum(g_n, 0.0)
    delta = np.sqrt(g_n_open**2 + np.sum(g_t * g_t, axis=1))  # (nq,)

    # Regime of each point from its unloading point h
    h = HISTORY_MARGIN * delta_max_c
    damaged = (h > params.delta_p) | (h >= params.w_c)  # h >= w_c: linear law, delta_p = w_c
    intact = ~damaged

    # Secant stiffness k_s = t / delta, finite at delta = 0, and dt/d(delta)
    k_s = np.empty(nq)
    dt_ddelta = np.empty(nq)

    # intact: reversible envelope, t = t_e(delta)
    _, slope, secant = cohesive_envelope(delta[intact], params.at(np.flatnonzero(intact)))
    k_s[intact] = secant
    dt_ddelta[intact] = slope

    # damaged: secant from the damage driver, t = t_e(kappa) / kappa * delta, with kappa >= h > 0
    delta_d = delta[damaged]
    kappa, dkappa = _damage_driver(delta_d, h[damaged], params.w_eps[damaged])
    _, slope_k, secant_k = cohesive_envelope(kappa, params.at(np.flatnonzero(damaged)))
    k_s[damaged] = secant_k
    dt_ddelta[damaged] = secant_k + delta_d * (slope_k - secant_k) / kappa * dkappa

    # Cohesive traction t_coh = t / delta (<g_n> n + gamma g_t)
    direction = g_n_open.reshape(nq, 1) * normal + gamma.reshape(nq, 1) * g_t  # (nq, nsd)
    t_coh = k_s.reshape(nq, 1) * direction

    # Tangent d(t_coh)/d([[u]]) = k_s (H n n^T + gamma P_t) + (dt/d(delta) - k_s) / delta^2 direction b^T,
    # with H = 1 where open and b = delta d(delta)/d([[u]]) = <g_n> n + g_t
    nn_T = _outer(normal, normal)  # (nq, nsd, nsd)
    P_t = np.eye(nsd) - nn_T
    H = (g_n > 0.0).astype(np.float64).reshape(nq, 1, 1)
    tangent = k_s.reshape(nq, 1, 1) * (H * nn_T + gamma.reshape(nq, 1, 1) * P_t)

    # second term, zero where delta = 0 (closed with no shear)
    b = g_n_open.reshape(nq, 1) * normal + g_t  # (nq, nsd)
    has_delta = delta > 0.0
    correction = np.zeros(nq)
    correction[has_delta] = (dt_ddelta[has_delta] - k_s[has_delta]) / delta[has_delta] ** 2
    tangent = tangent + correction.reshape(nq, 1, 1) * _outer(direction, b)
    return t_coh, tangent, delta


def bandis_barton_aperture(
    g_n: npt.NDArray[np.float64],
    sigma_c: npt.NDArray[np.float64],
    dsigma_c_dg: npt.NDArray[np.float64],
    w_m: npt.NDArray[np.float64],
    K_w: npt.NDArray[np.float64],
    ell_g: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Hydraulic aperture w and dw/dg_n from the normal jump g_n and contact pressure sigma_c.

    In contact: w = w_m K_w / (K_w + sigma_c) == w_m (1 - (sigma_c/K_w) / (1 + (sigma_c/K_w)) )
    Open: w = w_m + g_n
    Smoothed (ell_g > 0): w = w_m K_w / (K_w + sigma_c) + <g_n>, with the smoothed bracket, which
    tends to each branch away from g_n = 0.
    """
    # Bandis-Barton
    w_closed = w_m * K_w / (K_w + sigma_c)
    dw_closed = -w_closed / (K_w + sigma_c) * dsigma_c_dg
    closed = sigma_c > 0.0

    # Smoothed opening
    opening, dopening = smooth_bracket(g_n, ell_g)
    
    # Smoothed aperture across contact
    w = np.where(ell_g > 0.0, w_closed + opening, np.where(closed, w_closed, w_m + g_n))
    dw_dg = np.where(ell_g > 0.0, dw_closed + dopening, np.where(closed, dw_closed, 1.0))
    return w, dw_dg
