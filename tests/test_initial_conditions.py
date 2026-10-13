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

"""Tests for initial condition assignment."""

from typing import Any

import numpy as np

from raft.matprops import get_EOS
from raft.subsurface import _assign_well_ic_from_matrix, _no_wells_problem


def _mesh_with_wells() -> dict[str, Any]:
    """Three non-well nodes followed by two well nodes."""
    return {
        "nn": 3,
        "nn_total": 5,
        "nodes_total": np.arange(15, dtype=float).reshape(5, 3),
        "well_mesh": {"nn": 2},
    }


def _primary_variables() -> dict[str, Any]:
    return {
        "p_dofs": np.arange(5),
        "T_dofs": np.arange(5),
        "p": np.arange(20, dtype=float).reshape(5, 4),
        "T": np.arange(20, dtype=float).reshape(5, 4) + 100,
        "q": np.arange(5, dtype=float).reshape(5, 1),
    }


def test_no_wells_problem_removes_well_dofs() -> None:
    """The no-wells mesh and primary variables drop the well nodes, which are numbered last."""
    mesh = _mesh_with_wells()
    primary_variables = _primary_variables()
    mesh_no_wells, pv_no_wells, _ = _no_wells_problem(mesh, primary_variables, [])

    assert mesh_no_wells["nn_total"] == 3
    assert "well_mesh" not in mesh_no_wells
    np.testing.assert_array_equal(mesh_no_wells["nodes_total"], mesh["nodes_total"][:3])
    np.testing.assert_array_equal(pv_no_wells["p_dofs"], np.arange(3))
    np.testing.assert_array_equal(pv_no_wells["p"], primary_variables["p"][:3])
    np.testing.assert_array_equal(pv_no_wells["q"], primary_variables["q"][:3])
    # Inputs are untouched
    assert mesh["nn_total"] == 5
    assert "well_mesh" in mesh
    pv_no_wells["T"][:] = 0.0
    assert primary_variables["T"][0, 0] == 100.0


def test_no_wells_problem_filters_BC_groups() -> None:
    """External and well-only groups are dropped; mixed groups lose their well nodes."""
    no_wells = {"nodes": [0, 1], "thermal": {"type": "temperature", "values": [300.0]}}
    spanning = {"nodes": np.arange(5), "hydraulics": {"type": "pressure", "values": [1.0e6]}}
    well_only = {"nodes": [4], "hydraulics": {"type": "flux", "values": [1.0]}}
    external = {"nodes": [2], "label": "x", "hydraulics": {"type": "pressure", "external": True}}
    _, _, BC_no_wells = _no_wells_problem(
        _mesh_with_wells(), _primary_variables(), [no_wells, spanning, well_only, external]
    )

    assert len(BC_no_wells) == 2
    assert BC_no_wells[0] is no_wells
    np.testing.assert_array_equal(BC_no_wells[1]["nodes"], [0, 1, 2])
    # The trimmed copy does not share field specs with the original group
    assert BC_no_wells[1]["hydraulics"] is not spanning["hydraulics"]
    np.testing.assert_array_equal(spanning["nodes"], np.arange(5))


FLUID = {
    "density": 1000.0,
    "reference_temperature": 300.0,
    "reference_pressure": 1.0e5,
    "therm_expansion": 2.0e-4,
    "bulk_mod": 2.0e9,
    "specific_heat": 4200.0,
}


def _vertical_well_case() -> tuple[dict[str, Any], dict[str, Any]]:
    """Two matrix nodes; a vertical well with head node 3 at z = 30 above the domain and nodes 0-1
    inside (coincident with the matrix nodes), plus a second well (nodes 4-5) entirely outside."""
    matrix_nodes = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
    well_nodes = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 10.0], [0.0, 0.0, 20.0], [0.0, 0.0, 30.0], [5, 5, 60], [5, 5, 50]]
    )
    mesh = {
        "nsd": 3,
        "nn": 2,
        "nn_total": 8,
        "nodes_total": np.vstack([matrix_nodes, well_nodes]),
        "well_mesh": {
            "nn": 6,
            "nodes": well_nodes,
            "conn": [[0, 1], [1, 2], [2, 3], [4, 5]],
            "matrix_node_conn": [[0, 1], [-1, -1], [-1, -1], [-1, -1]],
            "head_nodes": [3, 4],
        },
    }
    p = np.full((8, 4), 7.0)
    T = np.full((8, 4), 280.0)
    p[0], p[1] = 2.0e6, 1.9e6
    T[0], T[1] = 360.0, 350.0
    return mesh, {"p": p, "T": T}


P_HEAD = 1.0e5


def _assign(mesh: dict[str, Any], primary_variables: dict[str, Any]) -> None:
    _assign_well_ic_from_matrix(mesh, {"fluid": FLUID}, {"gravity": -9.81}, primary_variables, P_HEAD)


def _rho(p: float, T: float) -> float:
    rho = get_EOS(np.array([[T]]), np.array([[T]]), np.array([[p]]), FLUID, want_density=True)[0]
    return float(rho.value[0, 0])


def test_well_ic_temperature_from_matrix_and_carried_outside() -> None:
    """In-domain well nodes copy matrix T; nodes outside take T from the nearest in-domain node."""
    mesh, primary_variables = _vertical_well_case()
    _assign(mesh, primary_variables)
    T = primary_variables["T"]

    np.testing.assert_array_equal(T[2], T[0])
    np.testing.assert_array_equal(T[3], T[1])
    np.testing.assert_array_equal(T[4], 350.0)
    np.testing.assert_array_equal(T[5], 350.0)
    np.testing.assert_array_equal(T[6:], 280.0)  # never inside the domain: unchanged


def test_well_ic_pressure_hydrostatic_from_head() -> None:
    """p is p_head at the well head and follows rho g dz down the whole well, inside the domain too."""
    mesh, primary_variables = _vertical_well_case()
    _assign(mesh, primary_variables)
    p, T = primary_variables["p"], primary_variables["T"]

    # Global indices of well nodes from the head (z = 30) down to z = 0
    path = [5, 4, 3, 2]
    np.testing.assert_array_equal(p[5], P_HEAD)
    for k, j in zip(path[:-1], path[1:]):
        expected = p[k, 0] + _rho(p[k, 0], T[k, 0]) * -9.81 * -10.0
        np.testing.assert_allclose(p[j], expected, rtol=1e-12)
    assert p[2, 0] > p[3, 0] > p[4, 0] > p[5, 0]


def test_well_ic_pressure_anchors_each_well_network() -> None:
    """A separate well network gets its own head anchor, even outside the domain."""
    mesh, primary_variables = _vertical_well_case()
    _assign(mesh, primary_variables)
    p = primary_variables["p"]

    np.testing.assert_array_equal(p[6], P_HEAD)
    np.testing.assert_allclose(p[7], P_HEAD + _rho(P_HEAD, 280.0) * -9.81 * -10.0, rtol=1e-12)
