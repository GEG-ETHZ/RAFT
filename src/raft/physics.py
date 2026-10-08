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

"""Per-domain physics switches resolved from the config `physics` block."""

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import numpy.typing as npt

FIELD_MODES = ("off", "steady", "transient")
DOMAINS = ("matrix", "fractures", "wells")


@dataclass(frozen=True, slots=True)
class DomainPhysics:
    """Physics switches for one domain (matrix, fractures, or wells)."""

    flow: str = "transient"  # one of FIELD_MODES
    heat: str = "transient"
    advection: bool = False

    @property
    def steady_flow(self) -> bool:
        """Pressure storage term dropped."""
        return self.flow == "steady"

    @property
    def steady_heat(self) -> bool:
        """Energy storage term dropped."""
        return self.heat == "steady"

    def as_steady(self) -> "DomainPhysics":
        """Transient fields made steady; `off` fields stay off."""
        return replace(
            self,
            flow="steady" if self.flow == "transient" else self.flow,
            heat="steady" if self.heat == "transient" else self.heat,
        )


@dataclass(frozen=True, slots=True)
class Physics:
    """Physics switches for every domain."""

    matrix: DomainPhysics
    fractures: DomainPhysics
    wells: DomainPhysics

    def domain(self, *, is_fracture: bool = False, is_well: bool = False) -> DomainPhysics:
        """Physics of the assembly pass's domain."""
        if is_well:
            return self.wells
        if is_fracture:
            return self.fractures
        return self.matrix

    def as_steady(self) -> "Physics":
        """All domains made steady."""
        return Physics(self.matrix.as_steady(), self.fractures.as_steady(), self.wells.as_steady())

    def advection_node_mask(self, mesh: dict[str, Any]) -> npt.NDArray[np.bool_]:
        """Global nodes lying in a domain with advection on (nodes ordered matrix, fractures, wells)."""
        mask = np.full(mesh["nn_total"], self.matrix.advection)
        nn_f = mesh["fracture_mesh"]["nn"] if "fracture_mesh" in mesh else 0
        mask[mesh["nn"] : mesh["nn"] + nn_f] = self.fractures.advection
        mask[mesh["nn"] + nn_f :] = self.wells.advection
        return mask
