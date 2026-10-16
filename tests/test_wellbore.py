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

"""Unit tests for wellbore boundary conditions (constraint swapping) and well initial conditions."""

from typing import Any

import numpy as np
import pytest

from raft.matprops import get_EOS
from raft.subsurface import (
    _assign_well_ic_from_matrix,
    _check_BC_constraints,
    _initialize_WB_checks,
    load_state,
    snapshot_state,
)


def _base_bc_config() -> list[dict[str, Any]]:
    return [
        {
            "nodes": [0, 1],
            "label": "well1",
            "hydraulics": {
                "values": [0.01],
                "function": "constant",
                "type": "flux",
                "constraints": {"maximum_pressure": 1.0e7, "minimum_pressure": 1.0e4},
            },
        }
    ]


@pytest.mark.parametrize(
    ("pressures", "mode", "values"),
    [
        ([1.5e7, 9.0e6], "pressure", [1.0e7]),  # above the maximum: held at the maximum
        ([5.0e3, 9.0e6], "pressure", [1.0e4]),  # below the minimum: held at the minimum
        ([5.0e6, 6.0e6], "flux", [0.01]),  # within the bounds: flux unchanged
    ],
)
def test_check_BC_constraints_from_flux_mode(pressures: list[float], mode: str, values: list[float]) -> None:
    """A flux-controlled wellbore switches to the violated pressure limit, and only then."""
    bc_config = _initialize_WB_checks(_base_bc_config())
    primary_variables = {"p": np.array(pressures).reshape(-1, 1), "q": np.zeros((2, 1))}

    satisfied, bc_config = _check_BC_constraints(bc_config, primary_variables, t=0.0, WB_on=1, verbose=False)

    hydraulics = bc_config[0]["hydraulics"]
    assert satisfied is (mode == "flux")
    assert hydraulics["curr_BC"] == mode
    assert hydraulics["type"] == mode
    assert hydraulics["values"] == values


def test_check_BC_constraints_only_one_swap_per_timestep() -> None:
    """A group that already swapped this timestep (curr_BC != prev_BC) must not swap again."""
    bc_config = _initialize_WB_checks(_base_bc_config())
    hydraulics = bc_config[0]["hydraulics"]
    hydraulics["curr_BC"] = "pressure"
    hydraulics["prev_BC"] = "flux"  # already swapped this timestep

    # flux condition would otherwise trigger a swap back to flux
    primary_variables = {"p": np.zeros((2, 1)), "q": np.array([[0.02], [0.02]])}

    satisfied, bc_config = _check_BC_constraints(bc_config, primary_variables, t=0.0, WB_on=1, verbose=False)

    assert satisfied is True
    assert bc_config[0]["hydraulics"]["curr_BC"] == "pressure"


def test_load_state_restores_wellbore_flags() -> None:
    """Constraint switching during a trial is undone by loading the snapshot taken before it."""
    bc_config = _initialize_WB_checks(_base_bc_config())
    primary_variables = {"p": np.ones((2, 4)), "T": np.ones((2, 4)), "q": np.zeros((2, 1))}
    state = snapshot_state(0.0, 0, 0, [1.0, 1.0, 1.0, 1.0], primary_variables, bc_config, {})

    hydraulics = bc_config[0]["hydraulics"]
    hydraulics["curr_BC"] = "pressure"
    hydraulics["type"], hydraulics["function"], hydraulics["values"] = "pressure", "constant", [1.0e7]
    hydraulics["values_flux"][0] = 0.5
    load_state(state, bc_config)

    assert hydraulics["curr_BC"] == "flux"
    assert hydraulics["type"] == "flux"
    assert hydraulics["values"] == [0.01]
    assert hydraulics["values_flux"] == [0.01]


def test_well_initial_condition_from_matrix_and_hydrostatic() -> None:
    """Well T is copied from the coincident matrix nodes and carried above the domain; well p is
    hydrostatic from p_head at the well head, inside the domain too."""
    fluid = {
        "density": 1000.0,
        "reference_temperature": 300.0,
        "reference_pressure": 1.0e5,
        "therm_expansion": 2.0e-4,
        "bulk_mod": 2.0e9,
        "specific_heat": 4200.0,
    }
    # Two matrix nodes at z = 0, 10; a vertical well (global nodes 2-5) up to its head at z = 30
    well_nodes = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0], [0.0, 0.0, 20.0], [0.0, 0.0, 30.0]])
    mesh = {
        "nsd": 3,
        "nn_total": 6,
        "well_mesh": {
            "nn": 4,
            "nodes": well_nodes,
            "conn": [[0, 1], [1, 2], [2, 3]],
            "matrix_node_conn": [[0, 1], [-1, -1], [-1, -1]],
            "head_nodes": [3],
        },
    }
    p = np.full((6, 4), 7.0)
    T = np.full((6, 4), 280.0)
    T[0], T[1] = 360.0, 350.0
    primary_variables = {"p": p, "T": T}
    p_head = 1.0e5
    _assign_well_ic_from_matrix(mesh, {"fluid": fluid}, {"gravity": -9.81}, primary_variables, p_head)

    np.testing.assert_array_equal(T[2:, 0], [360.0, 350.0, 350.0, 350.0])
    np.testing.assert_array_equal(p[5], p_head)
    for k, j in [(5, 4), (4, 3), (3, 2)]:  # down the well from the head
        T_k, p_k = np.array([[T[k, 0]]]), np.array([[p[k, 0]]])
        rho = get_EOS(T_k, T_k, p_k, fluid, want_density=True)[0].value[0, 0]
        np.testing.assert_allclose(p[j], p[k, 0] + rho * 9.81 * 10.0, rtol=1e-12)
