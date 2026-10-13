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

"""Unit tests for simulation state snapshot and load."""

from typing import Any

import numpy as np

from raft.subsurface import _initialize_WB_checks, load_state, snapshot_state


def _bc_config() -> list[dict[str, Any]]:
    wellbore = {
        "nodes": [0, 1],
        "tag": "well1",
        "flux_weight_matrix": np.eye(2),
        "hydraulics": {
            "values": [0.01],
            "function": "constant",
            "type": "flux",
            "constraints": {"maximum_pressure": 1.0e7, "minimum_pressure": 1.0e4},
        },
    }
    plain = {"nodes": [2], "thermal": {"type": "temperature", "function": "constant", "values": [300.0]}}
    return _initialize_WB_checks([wellbore, plain])


def _primary_variables() -> dict[str, Any]:
    return {
        "p_dofs": np.arange(3),
        "T_dofs": np.arange(3),
        "p": np.ones((3, 4)),
        "T": 300 * np.ones((3, 4)),
        "q": np.zeros((3, 1)),
    }


def test_load_undoes_changes_since_snapshot() -> None:
    """Changes made after a snapshot are undone by restoring it."""
    BC_config = _bc_config()
    primary_variables = _primary_variables()
    state = snapshot_state(5.0, 3, 1, [4.0, 2.0, 1.0, 1.0], primary_variables, BC_config, {})

    # Mutate everything a time step can touch
    primary_variables["p"][:, 0] = 7.0
    primary_variables["q"][0] = 1.0
    hydraulics = BC_config[0]["hydraulics"]
    hydraulics["curr_BC"] = "pressure"
    hydraulics["type"], hydraulics["function"], hydraulics["values"] = "pressure", "constant", [1.0e7]
    hydraulics["values_flux"][0] = 0.5

    loaded = load_state(state, BC_config)

    np.testing.assert_array_equal(loaded["p"], np.ones((3, 4)))
    np.testing.assert_array_equal(loaded["q"], np.zeros((3, 1)))
    assert hydraulics["curr_BC"] == "flux"
    assert hydraulics["type"] == "flux"
    assert hydraulics["values"] == [0.01]
    assert hydraulics["values_flux"] == [0.01]


def test_load_twice_gives_independent_copies() -> None:
    """A loaded working copy can be changed without altering the stored state."""
    BC_config = _bc_config()
    state = snapshot_state(0.0, 0, 0, [1.0, 1.0, 1.0, 1.0], _primary_variables(), BC_config, {})

    first = load_state(state, BC_config)
    first["T"][:, 0] = 999.0
    second = load_state(state, BC_config)

    np.testing.assert_array_equal(second["T"], 300 * np.ones((3, 4)))


def test_snapshot_skips_static_BC_data() -> None:
    """Only wellbore groups are captured, and static data such as flux_weight_matrix is not copied."""
    BC_config = _bc_config()
    weights = BC_config[0]["flux_weight_matrix"]
    state = snapshot_state(0.0, 0, 0, [1.0, 1.0, 1.0, 1.0], _primary_variables(), BC_config, {})
    load_state(state, BC_config)

    assert list(state.wb_flags) == [0]
    assert "flux_weight_matrix" not in state.wb_flags[0]
    assert BC_config[0]["flux_weight_matrix"] is weights
