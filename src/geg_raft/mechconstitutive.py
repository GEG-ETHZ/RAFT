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

from .mechanicsdata import CohesiveParameters

COHESIVE_LAWS = ("linear", "bilinear", "cubic")


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


def penalty_contact(
    g_n: npt.NDArray[np.float64], penalty: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Normal contact pressure sigma_c = eps_N <-g_n> (compression positive) and d(sigma_c)/d(g_n).

    Closed at g_n = 0.
    """
    closed = g_n <= 0.0
    sigma_c = np.where(closed, -penalty * g_n, 0.0)
    dsigma_c_dg = np.where(closed, -penalty, 0.0)
    return sigma_c, dsigma_c_dg


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
        x = np.minimum(delta / w_c, 1.0)
        secant = 6.75 * f_u / w_c * (1.0 - x) ** 2  # 6.75 = 27 / 4
        slope = 6.75 * f_u / w_c * (1.0 - x) * (1.0 - 3.0 * x)
        return secant * delta, np.where(bonded, slope, 0.0), secant
    if params.law == "bilinear":
        w_w = params.weakening_ratio * w_c
        rising = delta < w_w
        k_soft = f_u / (w_c - w_w)
        t_e = np.where(rising, f_u * delta / w_w, np.where(bonded, k_soft * (w_c - delta), 0.0))
        slope = np.where(rising, f_u / w_w, np.where(bonded, -k_soft, 0.0))
        # secant on the softening branch, delta >= w_w > 0
        secant = np.where(rising, f_u / w_w, t_e / np.where(rising, 1.0, delta))
        return t_e, slope, secant
    # linear
    secant = np.where(bonded, f_u / w_c, 0.0)
    return secant * delta, secant, secant


def cohesive_damage(
    delta_max: npt.NDArray[np.float64], params: CohesiveParameters
) -> npt.NDArray[np.float64]:
    """Damage D = 1 - (t_e / delta)(delta_max) / (t_e / delta)(delta_p): 0 before the peak, 1 from w_c."""
    _, _, secant = cohesive_envelope(delta_max, params)
    _, _, secant_p = cohesive_envelope(params.delta_p, params)
    damage = np.where(delta_max > params.delta_p, 1.0 - secant / np.where(secant_p > 0.0, secant_p, 1.0), 0.0)
    return np.where(delta_max >= params.w_c, 1.0, damage)


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
    return np.where(D_0 > 0.0, delta_max, 0.0)


def cohesive_traction(
    jump: npt.NDArray[np.float64],
    normal: npt.NDArray[np.float64],
    delta_max_c: npt.NDArray[np.float64],
    params: CohesiveParameters,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Cohesive traction t_coh (nq, nsd), its tangent d(t_coh)/d([[u]]) (nq, nsd, nsd), and the effective
    displacement delta (nq,) at the displacement jumps (nq, nsd).

    t_coh = t / delta (<g_n> n + gamma g_t), delta = sqrt(<g_n>^2 + g_t . g_t), from the envelope before
    the peak and from the secant t_e(kappa) / kappa delta after it, with kappa = max(delta_max_c, delta)
    smoothed in a band of half-width eps.
    """
    nsd = normal.shape[1]
    gamma = params.gamma
    # normal and tangential jumps
    g_n = np.sum(jump * normal, axis=1)
    g_t = jump - g_n[:, None] * normal
    open_n = np.maximum(g_n, 0.0)
    delta = np.sqrt(open_n**2 + np.sum(g_t * g_t, axis=1))

    # envelope branch: intact before the peak
    _, slope_env, secant_env = cohesive_envelope(delta, params)
    # secant branch: damaged, or debonded at w_c
    x = delta - delta_max_c
    eps = params.eps
    in_band = np.abs(x) < eps
    band = delta_max_c + (x + eps) ** 2 / (4.0 * eps)
    kappa = np.where(x <= -eps, delta_max_c, np.where(x >= eps, delta, band))
    dkappa = np.where(x <= -eps, 0.0, np.where(in_band, (x + eps) / (2.0 * eps), 1.0))
    _, slope_k, secant_k = cohesive_envelope(kappa, params)
    # kappa >= delta_max_c > 0 wherever the secant branch is used
    dsecant_k = (slope_k - secant_k) / np.where(kappa > 0.0, kappa, 1.0)
    damaged = (delta_max_c > params.delta_p) | (delta_max_c >= params.w_c)
    t_over_delta = np.where(damaged, secant_k, secant_env)
    dt_ddelta = np.where(damaged, secant_k + delta * dsecant_k * dkappa, slope_env)

    # traction t / delta (<g_n> n + gamma g_t)
    a = open_n[:, None] * normal + gamma[:, None] * g_t  # (nq, nsd)
    b = open_n[:, None] * normal + g_t  # (nq, nsd), delta * d(delta)/d([[u]])
    traction = t_over_delta[:, None] * a

    # tangent t / delta (H n n^T + gamma P_t) + (dt/ddelta - t / delta) / delta^2 a b^T
    nn_T = normal[:, :, None] * normal[:, None, :]  # (nq, nsd, nsd)
    P_t = np.eye(nsd) - nn_T
    H = (g_n > 0.0).astype(np.float64)
    tangent = t_over_delta[:, None, None] * (H[:, None, None] * nn_T + gamma[:, None, None] * P_t)
    safe = np.where(delta > 0.0, delta, 1.0)
    coeff = np.where(delta > 0.0, (dt_ddelta - t_over_delta) / safe**2, 0.0)
    tangent = tangent + coeff[:, None, None] * a[:, :, None] * b[:, None, :]
    return traction, tangent, delta


def bandis_barton_aperture(
    g_n: npt.NDArray[np.float64],
    sigma_c: npt.NDArray[np.float64],
    dsigma_c_dg: npt.NDArray[np.float64],
    w_m: npt.NDArray[np.float64],
    K_w: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Hydraulic aperture w and dw/dg_n from the normal jump g_n and contact pressure sigma_c.

    In contact: w = w_m K_w / (K_w + sigma_c) == w_m (1 - (sigma_c/K_w) / (1 + (sigma_c/K_w)) )
    Open: w = w_m + g_n
    """
    closed = sigma_c > 0.0
    w_closed = w_m * K_w / (K_w + sigma_c)
    w = np.where(closed, w_closed, w_m + g_n)
    dw_dg = np.where(closed, -w_closed / (K_w + sigma_c) * dsigma_c_dg, 1.0)
    return w, dw_dg
