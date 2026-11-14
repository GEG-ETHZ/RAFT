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

"""Constitutive laws of the solid: matrix elasticity and fracture contact, returning values with their
derivatives where the solve needs a tangent.
"""

import numpy as np
import numpy.typing as npt


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

    Open at g_n = 0.
    """
    closed = g_n < 0.0
    sigma_c = np.where(closed, -penalty * g_n, 0.0)
    dsigma_c_dg = np.where(closed, -penalty, 0.0)
    return sigma_c, dsigma_c_dg
