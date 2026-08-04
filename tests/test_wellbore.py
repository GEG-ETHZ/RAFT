"""Unit tests for the wellbore boundary condition (flux/pressure constraint swapping)."""

import copy
from typing import Any

import numpy as np

from raft.parseboundaryconditions import parsebcs
from raft.subsurface import _check_BC_constraints, _initialize_WB_checks


def _base_bc_config() -> dict[str, Any]:
    return {
        "wellbore": [
            {
                "nodes": [0, 1],
                "values": [0.01],
                "function": "constant",
                "tag": "well1",
                "wellbore": {"maximum_pressure": 1.0e7, "minimum_pressure": 1.0e4},
            }
        ],
        "pressure_fixed": [],
        "temperature_fixed": [],
        "fluid_flux": [],
        "temperature_flux": [],
    }


def test_initialize_WB_checks_sets_default_flux_state() -> None:
    """A wellbore group should start in flux mode, constrained to the max pressure limit."""
    bc_config = _initialize_WB_checks(_base_bc_config())
    group = bc_config["wellbore"][0]

    assert group["curr_BC"] == "flux"
    assert group["prev_BC"] == "flux"
    assert group["pres_constraint"] == "max"
    assert group["values_flux"] == [0.01]
    assert group["function_flux"] == "constant"


def test_parsebcs_wellbore_applies_flux_when_curr_BC_is_flux() -> None:
    """While unconstrained, a wellbore group should apply its flux as a Neumann condition."""
    bc_config = _initialize_WB_checks(_base_bc_config())
    mesh = {"nodes": np.zeros((2, 2))}

    result = parsebcs(bc_config, mesh, {}, t=0.0, controls={})

    assert result["p_dirichlet"].size == 0
    np.testing.assert_array_equal(result["p_neumann"], np.array([[0, 0.01], [1, 0.01]]))

    # bc_config should be restored to its original (empty) lengths after parsing
    assert bc_config["fluid_flux"] == []
    assert bc_config["pressure_fixed"] == []


def test_parsebcs_wellbore_applies_pressure_when_constrained() -> None:
    """Once a pressure limit is hit, the wellbore group should apply that limit as a Dirichlet condition."""
    bc_config = _initialize_WB_checks(_base_bc_config())
    bc_config["wellbore"][0]["curr_BC"] = "pressure"
    bc_config["wellbore"][0]["pres_constraint"] = "max"
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
    group = bc_config["wellbore"][0]
    assert group["curr_BC"] == "pressure"
    assert group["pres_constraint"] == "max"


def test_check_BC_constraints_switches_to_pressure_on_min() -> None:
    """Dropping below the minimum pressure while in flux mode should switch the group to pressure mode."""
    bc_config = _initialize_WB_checks(_base_bc_config())
    primary_variables = {"p": np.array([[5.0e3], [9.0e6]]), "q": np.zeros((2, 1))}

    satisfied, bc_config = _check_BC_constraints(bc_config, primary_variables, t=0.0, WB_on=1, verbose=False)

    assert satisfied is False
    group = bc_config["wellbore"][0]
    assert group["curr_BC"] == "pressure"
    assert group["pres_constraint"] == "min"


def test_check_BC_constraints_stays_flux_within_pressure_bounds() -> None:
    """Pressures within the min/max range should leave the group unconstrained."""
    bc_config = _initialize_WB_checks(_base_bc_config())
    primary_variables = {"p": np.array([[5.0e6], [6.0e6]]), "q": np.zeros((2, 1))}

    satisfied, bc_config = _check_BC_constraints(bc_config, primary_variables, t=0.0, WB_on=1, verbose=False)

    assert satisfied is True
    assert bc_config["wellbore"][0]["curr_BC"] == "flux"


def test_check_BC_constraints_switches_back_to_flux_when_flux_target_exceeded() -> None:
    """In pressure mode, exceeding the original flux target should switch the group back to flux mode."""
    bc_config = _initialize_WB_checks(_base_bc_config())
    group = bc_config["wellbore"][0]
    group["curr_BC"] = "pressure"
    group["prev_BC"] = "pressure"

    # required flux is 0.01; actual flux exceeds it
    primary_variables = {"p": np.zeros((2, 1)), "q": np.array([[0.02], [0.02]])}

    satisfied, bc_config = _check_BC_constraints(bc_config, primary_variables, t=0.0, WB_on=1, verbose=False)

    assert satisfied is False
    assert bc_config["wellbore"][0]["curr_BC"] == "flux"


def test_check_BC_constraints_only_one_swap_per_timestep() -> None:
    """A group that already swapped this timestep (curr_BC != prev_BC) must not swap again."""
    bc_config = _initialize_WB_checks(_base_bc_config())
    group = bc_config["wellbore"][0]
    group["curr_BC"] = "pressure"
    group["prev_BC"] = "flux"  # already swapped this timestep

    # flux condition would otherwise trigger a swap back to flux
    primary_variables = {"p": np.zeros((2, 1)), "q": np.array([[0.02], [0.02]])}

    satisfied, bc_config = _check_BC_constraints(bc_config, primary_variables, t=0.0, WB_on=1, verbose=False)

    assert satisfied is True
    assert bc_config["wellbore"][0]["curr_BC"] == "pressure"


def test_check_BC_constraints_noop_when_WB_off() -> None:
    """When no wellbore groups are configured, constraint checking should be a no-op."""
    bc_config = _base_bc_config()
    original = copy.deepcopy(bc_config)
    primary_variables = {"p": np.zeros((2, 1)), "q": np.zeros((2, 1))}

    satisfied, bc_config = _check_BC_constraints(bc_config, primary_variables, t=0.0, WB_on=0, verbose=False)

    assert satisfied is True
    assert bc_config == original
