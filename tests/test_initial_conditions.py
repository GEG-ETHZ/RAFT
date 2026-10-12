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

from raft.subsurface import _no_wells_problem


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
    external = {"nodes": [2], "tag": "x", "hydraulics": {"type": "pressure", "external": True}}
    _, _, BC_no_wells = _no_wells_problem(
        _mesh_with_wells(), _primary_variables(), [no_wells, spanning, well_only, external]
    )

    assert len(BC_no_wells) == 2
    assert BC_no_wells[0] is no_wells
    np.testing.assert_array_equal(BC_no_wells[1]["nodes"], [0, 1, 2])
    # The trimmed copy does not share field specs with the original group
    assert BC_no_wells[1]["hydraulics"] is not spanning["hydraulics"]
    np.testing.assert_array_equal(spanning["nodes"], np.arange(5))
