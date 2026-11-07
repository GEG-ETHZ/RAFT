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

"""Degree-of-freedom layouts of the primary fields."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

COMPONENTS = ("x", "y", "z")


@dataclass(frozen=True, slots=True)
class FieldLayout:
    """DoF layout of one primary field; vector DoFs are node-interleaved."""

    name: str
    ndof: int  # DoFs per node
    nn: int  # nodes carrying the field

    @property
    def size(self) -> int:
        """Number of DoFs of the field."""
        return self.nn * self.ndof


def field_layouts(mesh: dict[str, Any]) -> dict[str, FieldLayout]:
    """Layouts of p, T (all nodes) and u (matrix nodes only)."""
    return {
        "p": FieldLayout("p", 1, mesh["nn_total"]),
        "T": FieldLayout("T", 1, mesh["nn_total"]),
        "u": FieldLayout("u", mesh["nsd"], mesh["nn"]),
    }


def get_scatter(nodes: Sequence[int] | npt.NDArray[np.integer], ndof: int) -> npt.NDArray[np.int64]:
    """Global DoFs of `nodes`, node-interleaved.

    Node n: x = ndof*n, y = ndof*n + 1, z = ndof*n + 2.
    """
    nodes_arr = np.asarray(nodes, dtype=np.int64).ravel()
    # each node's block of ndof consecutive DoFs, components in x, y, z order
    return (nodes_arr[:, None] * ndof + np.arange(ndof)).ravel()


def component_dofs(
    nodes: Sequence[int] | npt.NDArray[np.integer], ndof: int, components: Sequence[str]
) -> npt.NDArray[np.int64]:
    """Global DoFs of the named vector `components` ("x", "y", "z") at `nodes`, node-major."""
    comps = [COMPONENTS.index(c) for c in components]
    if any(c >= ndof for c in comps):
        raise ValueError(f"Components {list(components)} exceed the {ndof} DoFs per node")
    nodes_arr = np.asarray(nodes, dtype=np.int64).ravel()
    return (nodes_arr[:, None] * ndof + np.asarray(comps, dtype=np.int64)).ravel()
