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

"""Constant matrices of the linear force balance, the fracture face operators, and the fracture aperture."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.sparse import csc_matrix


@dataclass(frozen=True, slots=True)
class CohesiveParameters:
    """Cohesive law and its parameters at the fracture Gauss points."""

    law: str  # "linear", "bilinear" or "cubic"; "" without cohesion
    weakening_ratio: float  # a = w_w / w_c, bilinear only
    f_u: npt.NDArray[np.float64]  # (nq,) tensile strength
    gamma: npt.NDArray[np.float64]  # (nq,) shear to tensile strength ratio s_u / f_u
    w_c: npt.NDArray[np.float64]  # (nq,) critical effective displacement
    delta_p: npt.NDArray[np.float64]  # (nq,) effective displacement at the peak
    eps: npt.NDArray[np.float64]  # (nq,) half-width of the smoothed-maximum band

    @classmethod
    def empty(cls) -> CohesiveParameters:
        """Placeholder without cohesion."""
        return cls("", 0.0, np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0))


@dataclass(frozen=True, slots=True)
class FractureInterface:
    """Fracture face operators at the fracture Gauss points, integrated once per run.

    The displacement jump [[u]] = u+ - u- at Gauss point k is jump[nsd*k : nsd*(k+1)] @ u, and its
    normal component g_n (opening positive) is jump_normal[k] @ u.
    """

    jump: csc_matrix  # (nsd*nq, nsd*nn)
    jump_normal: csc_matrix  # (nq, nsd*nn)
    weights: npt.NDArray[np.float64]  # (nq,) Gauss weight * face measure (* r axisymmetric)
    penalty: npt.NDArray[np.float64]  # (nq,) contact penalty stiffness eps_N
    element: npt.NDArray[np.int64]  # (nq,) fracture element of each Gauss point
    w_m: npt.NDArray[np.float64]  # (nq,) unstressed aperture of the point's element
    K_w: npt.NDArray[np.float64]  # (nq,) Bandis-Barton stiffness of the point's fracture
    points: npt.NDArray[np.int64]  # (ne_f, nq_e) Gauss points of each fracture element
    projection: npt.NDArray[np.float64]  # (ne_f, nne_f, nq_e) lumped projection to the element nodes
    normal: npt.NDArray[np.float64]  # (nq, nsd) unit normal of the point's fracture
    cohesive: CohesiveParameters

    @classmethod
    def empty(cls) -> FractureInterface:
        """Placeholder without fractures."""
        return cls(
            csc_matrix((0, 0)),
            csc_matrix((0, 0)),
            np.empty(0),
            np.empty(0),
            np.empty(0, dtype=np.int64),
            np.empty(0),
            np.empty(0),
            np.empty((0, 0), dtype=np.int64),
            np.empty((0, 0, 0)),
            np.empty((0, 0)),
            CohesiveParameters.empty(),
        )


@dataclass(frozen=True, slots=True)
class FractureAperture:
    """Bandis-Barton aperture at every time level of the displacement, with its displacement tangent.

    Columns follow the displacement history [k, n, n-1, n-2]. Element arrays are indexed by fracture
    element; D_w = dw/du at the Gauss points for the current iterate.
    """

    gauss: npt.NDArray[np.float64]  # (ne_f, nq_e, n_levels) aperture at the Gauss points
    element_nodal: npt.NDArray[np.float64]  # (ne_f, nne_f, n_levels) lumped projection to the nodes
    D_w: csc_matrix  # (nq, nsd*nn), Gauss points in interface order
    points: npt.NDArray[np.int64]  # (ne_f, nq_e) interface order of each element's Gauss points

    @classmethod
    def empty(cls) -> FractureAperture:
        """Placeholder when the aperture is the constant input."""
        return cls(
            np.empty((0, 0, 0)), np.empty((0, 0, 0)), csc_matrix((0, 0)), np.empty((0, 0), dtype=np.int64)
        )


_NO_APERTURE = FractureAperture.empty()


@dataclass(frozen=True, slots=True)
class MechanicsMatrices:
    """Constant force-balance matrices, nodal strain operators and fracture face operators.

    R_u = M_uu sum_j c_a,j u_(n-j) + K_uu u + K_up (p - p_0) + K_uT (T - T_0) - F_fracture_0 + R_interface(u)
    - F_ext, where the fracture pressure columns of K_up carry the total fracture face pressure (p_0 taken as
    zero there). M_uu is empty without inertia.
    """

    K_uu: csc_matrix  # (nsd*nn, nsd*nn)
    K_up: csc_matrix  # (nsd*nn, nn_total)
    K_uT: csc_matrix  # (nsd*nn, nn_total)
    nodal_strain: tuple[csc_matrix, ...]  # per region, (nn, nsd*nn): eps_v at nodes = nodal_strain[r] @ u
    F_fracture_0: npt.NDArray[np.float64]  # (nsd*nn, 1) face load of the in-situ stress
    interface: FractureInterface
    M_uu: csc_matrix  # (nsd*nn, nsd*nn) consistent mass; (0, 0) without inertia

    @classmethod
    def empty(cls) -> MechanicsMatrices:
        """Placeholder when mechanics is off."""
        return cls(
            csc_matrix((0, 0)),
            csc_matrix((0, 0)),
            csc_matrix((0, 0)),
            (),
            np.empty((0, 1)),
            FractureInterface.empty(),
            csc_matrix((0, 0)),
        )


_NO_MECHANICS = MechanicsMatrices.empty()
