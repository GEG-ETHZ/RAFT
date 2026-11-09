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

"""Constant matrices of the linear force balance."""

from __future__ import annotations

from dataclasses import dataclass

from scipy.sparse import csc_matrix


@dataclass(frozen=True, slots=True)
class MechanicsMatrices:
    """Constant force-balance matrices and nodal strain operators, integrated once per run.

    R_u = K_uu u + K_up (p - p_0) + K_uT (T - T_0) - F_ext.
    """

    K_uu: csc_matrix  # (nsd*nn, nsd*nn)
    K_up: csc_matrix  # (nsd*nn, nn_total)
    K_uT: csc_matrix  # (nsd*nn, nn_total)
    nodal_strain: tuple[csc_matrix, ...]  # per region, (nn, nsd*nn): eps_v at nodes = nodal_strain[r] @ u

    @classmethod
    def empty(cls) -> MechanicsMatrices:
        """Placeholder when mechanics is off."""
        return cls(csc_matrix((0, 0)), csc_matrix((0, 0)), csc_matrix((0, 0)), ())


_NO_MECHANICS = MechanicsMatrices.empty()
