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

"""Constant matrices of the linear force balance and the fracture face operators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.sparse import csc_matrix


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

    @classmethod
    def empty(cls) -> FractureInterface:
        """Placeholder without fractures."""
        return cls(
            csc_matrix((0, 0)),
            csc_matrix((0, 0)),
            np.empty(0),
            np.empty(0),
            np.empty(0, dtype=np.int64),
        )


@dataclass(frozen=True, slots=True)
class MechanicsMatrices:
    """Constant force-balance matrices, nodal strain operators and fracture face operators.

    R_u = K_uu u + K_up (p - p_0) + K_uT (T - T_0) - F_fracture_0 + R_contact(u) - F_ext, where the
    fracture pressure columns of K_up carry the total fracture face pressure (p_0 taken as zero there).
    """

    K_uu: csc_matrix  # (nsd*nn, nsd*nn)
    K_up: csc_matrix  # (nsd*nn, nn_total)
    K_uT: csc_matrix  # (nsd*nn, nn_total)
    nodal_strain: tuple[csc_matrix, ...]  # per region, (nn, nsd*nn): eps_v at nodes = nodal_strain[r] @ u
    F_fracture_0: npt.NDArray[np.float64]  # (nsd*nn, 1) face load of the in-situ stress
    interface: FractureInterface

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
        )


_NO_MECHANICS = MechanicsMatrices.empty()
