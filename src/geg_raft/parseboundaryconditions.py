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

"""Parse the input boundary conditions and translate them into FEM input.

Each group in `config["boundary_conditions"]` describes one geometry feature and
carries up to two field specs, "hydraulics" and "thermal".
"""

from typing import Any

import numpy as np
import numpy.typing as npt


def evaluateBCfunctions(incoming: dict[str, Any], t: float, nodes: npt.NDArray[np.float64]) -> dict[str, Any]:
    """Evaluate functions defining the boundary condition values."""
    match incoming["function"]:
        case "constant":
            incoming["BCvalues"] = [incoming["values"][0]] * len(incoming["nodes"])
        case "external":
            # Standalone runs only; the coupling API replaces this with a table each solve
            if not incoming.get("values"):
                raise ValueError(
                    "External boundary condition has no values; set them via the API or in the config."
                )
            incoming["BCvalues"] = [incoming["values"][0]] * len(incoming["nodes"])
        case "linear":
            incoming["BCvalues"] = [t * incoming["values"][0] + incoming["values"][1]] * len(
                incoming["nodes"]
            )
        case "gradient":
            s = incoming["values"][0]
            s = s / np.linalg.norm(s)  # unit vector in gradient direction
            y0 = incoming["values"][1]  # reference gradient value at min point
            slope = incoming["values"][2]  # gradient slope

            indx_nodes = incoming["nodes"]

            x = nodes[indx_nodes, :] @ s
            x_min = min(x)

            incoming["BCvalues"] = y0 + slope * (x - x_min)
        case "ramp":
            y_0 = incoming["values"][0][0]
            y_1 = incoming["values"][0][1]
            t_ramp = incoming["values"][1]
            y = y_1 if t >= t_ramp else y_0 + t / t_ramp * (y_1 - y_0)
            incoming["BCvalues"] = [y] * len(incoming["nodes"])
        case "table":
            tn = incoming["values"][0]
            yn = incoming["values"][1]
            y = np.interp(t, tn, yn) if t >= tn[0] else 0
            incoming["BCvalues"] = [y] * len(incoming["nodes"])
    return incoming


def _apply_field(
    mesh: dict[str, Any], t: float, group: dict[str, Any], field: str, nodes: list[int]
) -> tuple[npt.NDArray[np.float64], bool] | None:
    """Evaluate one field spec ("hydraulics" or "thermal") on a group into [node, value] pairs."""
    spec = group.get(field)
    if spec is None or len(nodes) == 0:
        return None

    spec["nodes"] = nodes
    if "function" not in spec:
        spec["function"] = "constant"  # assume constant value if no function given
    spec = evaluateBCfunctions(spec, t, mesh["nodes"])
    values = np.asarray(spec["BCvalues"], dtype=np.float64)

    if spec["type"] == "flux" and "flux_weight_matrix" in group:
        values = group["flux_weight_matrix"] @ values

    is_dirichlet = spec["type"] in ("pressure", "temperature")
    pairs = np.column_stack((np.asarray(nodes, dtype=np.float64), values))
    return pairs, is_dirichlet


def _stack_unique(rows: list[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    """Stack collected [node, value] row blocks, keeping one entry per node."""
    if not rows:
        return np.empty((0, 2))
    arr = np.vstack(rows)
    _, idx = np.unique(arr[:, 0], return_index=True)
    return arr[idx]


def parsebcs(
    BC_config: list[dict[str, Any]],
    mesh: dict[str, Any],
    primary_variables: dict[str, Any],  # noqa
    t: float,
    controls: dict[str, Any],  # noqa
) -> dict[str, Any]:
    """Parse input boundary condition groups and return a dictionary containing the following.

    T_dirichlet:      n x 2 vector containing a list of fixed temperature nodes and values
                        [ nodes, temperatures]
    T_neumann:        n x 2 vector containing a list of fixed temperature flux nodes and values
                        [ nodes, flux]
    p_dirichlet:      n x 2 vector containing a list of fixed pressure nodes and values
                        [ nodes, pressures]
    p_neumann:        n x 2 vector containing a list of fixed fluid flux nodes and values
                        [ nodes, flux]
    """
    p_dirichlet_rows: list[npt.NDArray[np.float64]] = []
    p_neumann_rows: list[npt.NDArray[np.float64]] = []
    T_dirichlet_rows: list[npt.NDArray[np.float64]] = []
    T_neumann_rows: list[npt.NDArray[np.float64]] = []

    for group in BC_config:
        nodes = group["nodes"]
        if len(nodes) == 0:
            continue

        hydraulics = _apply_field(mesh, t, group, "hydraulics", nodes)
        if hydraulics is not None:
            pairs, is_dirichlet = hydraulics
            (p_dirichlet_rows if is_dirichlet else p_neumann_rows).append(pairs)

        thermal = _apply_field(mesh, t, group, "thermal", nodes)
        if thermal is not None:
            pairs, is_dirichlet = thermal
            (T_dirichlet_rows if is_dirichlet else T_neumann_rows).append(pairs)

    return {
        "T_dirichlet": _stack_unique(T_dirichlet_rows),
        "T_neumann": _stack_unique(T_neumann_rows),
        "p_dirichlet": _stack_unique(p_dirichlet_rows),
        "p_neumann": _stack_unique(p_neumann_rows),
    }
