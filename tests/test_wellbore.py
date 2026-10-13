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

"""Unit tests for the wellbore boundary condition (flux/pressure constraint swapping)."""

import copy
from typing import Any

import numpy as np

from raft.parseboundaryconditions import parsebcs
from raft.subsurface import _check_BC_constraints, _initialize_WB_checks


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


def test_initialize_WB_checks_sets_default_flux_state() -> None:
    """A wellbore group should start in flux mode, constrained to the max pressure limit."""
    bc_config = _initialize_WB_checks(_base_bc_config())
    hydraulics = bc_config[0]["hydraulics"]

    assert hydraulics["curr_BC"] == "flux"
    assert hydraulics["prev_BC"] == "flux"
    assert hydraulics["pres_constraint"] == "max"
    assert hydraulics["values_flux"] == [0.01]
    assert hydraulics["function_flux"] == "constant"


def test_initialize_WB_checks_defaults_missing_function_to_constant() -> None:
    """A wellbore group with no `function` given should default to constant, not raise."""
    bc_config = _base_bc_config()
    del bc_config[0]["hydraulics"]["function"]

    bc_config = _initialize_WB_checks(bc_config)
    hydraulics = bc_config[0]["hydraulics"]

    assert hydraulics["function"] == "constant"
    assert hydraulics["function_flux"] == "constant"


def test_parsebcs_wellbore_applies_flux_when_curr_BC_is_flux() -> None:
    """While unconstrained, a wellbore group should apply its flux as a Neumann condition."""
    bc_config = _initialize_WB_checks(_base_bc_config())
    mesh = {"nodes": np.zeros((2, 2))}

    result = parsebcs(bc_config, mesh, {}, t=0.0, controls={})

    assert result["p_dirichlet"].size == 0
    np.testing.assert_array_equal(result["p_neumann"], np.array([[0, 0.01], [1, 0.01]]))


def test_parsebcs_wellbore_applies_pressure_when_constrained() -> None:
    """Once a pressure limit is hit, the wellbore group should apply that limit as a Dirichlet condition."""
    bc_config = _initialize_WB_checks(_base_bc_config())
    hydraulics = bc_config[0]["hydraulics"]
    hydraulics["curr_BC"] = "pressure"
    hydraulics["pres_constraint"] = "max"
    hydraulics["type"] = "pressure"
    hydraulics["function"] = "constant"
    hydraulics["values"] = [1.0e7]
    mesh = {"nodes": np.zeros((2, 2))}

    result = parsebcs(bc_config, mesh, {}, t=0.0, controls={})

    assert result["p_neumann"].size == 0
    np.testing.assert_array_equal(result["p_dirichlet"], np.array([[0, 1.0e7], [1, 1.0e7]]))


def test_check_BC_constraints_switches_to_pressure_on_max() -> None:
    """Exceeding the maximum pressure while in flux mode should switch the group to pressure mode."""
    bc_config = _initialize_WB_checks(_base_bc_config())
    primary_variables = {"p": np.array([[1.5e7], [9.0e6]]), "q": np.zeros((2, 1))}

    satisfied, bc_config = _check_BC_constraints(bc_config, primary_variables, t=0.0, WB_on=1, verbose=False)

    assert satisfied is False
    hydraulics = bc_config[0]["hydraulics"]
    assert hydraulics["curr_BC"] == "pressure"
    assert hydraulics["pres_constraint"] == "max"
    assert hydraulics["type"] == "pressure"
    assert hydraulics["values"] == [1.0e7]


def test_check_BC_constraints_switches_to_pressure_on_min() -> None:
    """Dropping below the minimum pressure while in flux mode should switch the group to pressure mode."""
    bc_config = _initialize_WB_checks(_base_bc_config())
    primary_variables = {"p": np.array([[5.0e3], [9.0e6]]), "q": np.zeros((2, 1))}

    satisfied, bc_config = _check_BC_constraints(bc_config, primary_variables, t=0.0, WB_on=1, verbose=False)

    assert satisfied is False
    hydraulics = bc_config[0]["hydraulics"]
    assert hydraulics["curr_BC"] == "pressure"
    assert hydraulics["pres_constraint"] == "min"
    assert hydraulics["values"] == [1.0e4]


def test_check_BC_constraints_stays_flux_within_pressure_bounds() -> None:
    """Pressures within the min/max range should leave the group unconstrained."""
    bc_config = _initialize_WB_checks(_base_bc_config())
    primary_variables = {"p": np.array([[5.0e6], [6.0e6]]), "q": np.zeros((2, 1))}

    satisfied, bc_config = _check_BC_constraints(bc_config, primary_variables, t=0.0, WB_on=1, verbose=False)

    assert satisfied is True
    assert bc_config[0]["hydraulics"]["curr_BC"] == "flux"


def test_check_BC_constraints_switches_back_to_flux_when_flux_target_exceeded() -> None:
    """In pressure mode, exceeding the original flux target should switch the group back to flux mode."""
    bc_config = _initialize_WB_checks(_base_bc_config())
    hydraulics = bc_config[0]["hydraulics"]
    hydraulics["curr_BC"] = "pressure"
    hydraulics["prev_BC"] = "pressure"

    # required flux is 0.01; actual flux exceeds it
    primary_variables = {"p": np.zeros((2, 1)), "q": np.array([[0.02], [0.02]])}

    satisfied, bc_config = _check_BC_constraints(bc_config, primary_variables, t=0.0, WB_on=1, verbose=False)

    assert satisfied is False
    hydraulics = bc_config[0]["hydraulics"]
    assert hydraulics["curr_BC"] == "flux"
    assert hydraulics["type"] == "flux"
    assert hydraulics["values"] == [0.01]


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


def test_check_BC_constraints_noop_when_WB_off() -> None:
    """When no wellbore groups are configured, constraint checking should be a no-op."""
    bc_config = _base_bc_config()
    original = copy.deepcopy(bc_config)
    primary_variables = {"p": np.zeros((2, 1)), "q": np.zeros((2, 1))}

    satisfied, bc_config = _check_BC_constraints(bc_config, primary_variables, t=0.0, WB_on=0, verbose=False)

    assert satisfied is True
    assert bc_config == original
