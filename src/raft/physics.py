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

from collections.abc import Callable
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
    def solve_flow(self) -> bool:
        """Mass balance assembled; `off` holds pressure at its current values."""
        return self.flow != "off"

    @property
    def solve_heat(self) -> bool:
        """Energy balance assembled; `off` holds temperature at its current values."""
        return self.heat != "off"

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

    def with_off(self, field: str) -> "Physics":
        """`field` ("flow" or "heat") switched off in every domain."""
        domains = (self.matrix, self.fractures, self.wells)
        if field == "flow":
            return Physics(*(replace(d, flow="off") for d in domains))
        return Physics(*(replace(d, heat="off") for d in domains))

    def with_domain_off(self, domain: str) -> "Physics":
        """Both fields of `domain` ("matrix", "fractures" or "wells") switched off."""
        off = replace(getattr(self, domain), flow="off", heat="off")
        return replace(self, **{domain: off})

    def solves(self, field: str) -> bool:
        """Whether `field` ("flow" or "heat") is on in any domain."""
        return any(getattr(d, field) != "off" for d in (self.matrix, self.fractures, self.wells))

    def node_mask(self, mesh: dict[str, Any], test: Callable[[DomainPhysics], bool]) -> npt.NDArray[np.bool_]:
        """Global nodes whose domain passes `test` (nodes ordered matrix, fractures, wells)."""
        mask = np.full(mesh["nn_total"], test(self.matrix))
        nn_f = mesh["fracture_mesh"]["nn"] if "fracture_mesh" in mesh else 0
        mask[mesh["nn"] : mesh["nn"] + nn_f] = test(self.fractures)
        mask[mesh["nn"] + nn_f :] = test(self.wells)
        return mask

    def advection_node_mask(self, mesh: dict[str, Any]) -> npt.NDArray[np.bool_]:
        """Global nodes lying in a domain with advection on."""
        return self.node_mask(mesh, lambda d: d.advection)

    def off_nodes(self, mesh: dict[str, Any], field: str) -> npt.NDArray[np.intp]:
        """Global nodes whose `field` ("flow" or "heat") is off."""
        return np.flatnonzero(self.node_mask(mesh, lambda d: getattr(d, field) == "off"))
