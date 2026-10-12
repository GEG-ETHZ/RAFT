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

"""Test the config `physics` block: parsing and per-domain `off` fields."""

import contextlib
import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from raft.physics import DomainPhysics, Physics
from raft.preprocessing import DEFAULTS, _apply_defaults, _parse_physics
from raft.subsurface import _build_steady_state_controls, main

CASE_DIR = Path(__file__).parent / "test_configs"


def _parse(physics_yaml: str) -> Physics:
    config: dict[str, Any] = {"physics": yaml.safe_load(physics_yaml)}
    _apply_defaults(config, {"physics": DEFAULTS["physics"]}, verbose=False)
    return _parse_physics(config, verbose=False)


def test_defaults() -> None:
    physics = _parse("")
    assert physics.matrix == physics.fractures == physics.wells == DomainPhysics()


def test_inheritance_and_yaml_booleans() -> None:
    physics = _parse("matrix: {heat: steady}\nwells: {advection: on}")
    assert physics.matrix == DomainPhysics(heat="steady", advection=False)
    assert physics.fractures == physics.matrix
    assert physics.wells == DomainPhysics(heat="steady", advection=True)
    assert physics.as_steady().wells == DomainPhysics(flow="steady", heat="steady", advection=True)
    assert _parse("matrix: {flow: off}\nwells: {flow: steady}").matrix.flow == "off"


@pytest.mark.parametrize(
    ("physics_yaml", "error"),
    [
        ("matrix: {flow: implicit}", ValueError),
        ("matrix: {advection: 2}", ValueError),
        ("matrix: {pressure: steady}", ValueError),
        ("well: {advection: on}", ValueError),
    ],
)
def test_invalid(physics_yaml: str, error: type[Exception]) -> None:
    with pytest.raises(error):
        _parse(physics_yaml)


def test_transient_well_flow_warns() -> None:
    with pytest.warns(UserWarning, match="quasi-steady"):
        _parse("wells: {flow: transient}")


def test_advection_node_mask() -> None:
    physics = _parse("matrix: {advection: off}\nfractures: {advection: on}\nwells: {advection: off}")
    mesh = {"nn": 3, "nn_total": 7, "fracture_mesh": {"nn": 2}}
    assert np.array_equal(physics.advection_node_mask(mesh), [0, 0, 0, 1, 1, 0, 0])


def _run(
    tmp_path: Path, config_name: str, physics: dict[str, Any], assignment: str = "constant", **simulation: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run a test config with its physics block, IC assignment, and simulation entries replaced."""
    config = yaml.safe_load((CASE_DIR / config_name).read_text())
    config["mesh"]["file"] = str(CASE_DIR / config["mesh"]["file"])
    config["physics"] = physics
    config["initial_conditions"]["assignment"] = assignment
    config["simulation"].update(simulation)
    (tmp_path / config_name).write_text(yaml.safe_dump(config))
    with contextlib.redirect_stdout(io.StringIO()):
        return main(config_name=config_name, config_dir=tmp_path, verbose=False)


@pytest.mark.parametrize("assignment", ["constant", "BC_compatible"])
@pytest.mark.parametrize("solver_type", ["linearized", "fully_coupled"])
@pytest.mark.parametrize(("off", "solved"), [("heat", "p"), ("flow", "T")])
def test_single_physics_matches_coupled(
    tmp_path: Path, solver_type: str, off: str, solved: str, assignment: str
) -> None:
    """In the decoupled patch test, one field `off` leaves the other's solution unchanged and
    holds itself at its *_ini value apart from its Dirichlet BCs, ignoring the BC_compatible seed."""
    config_name = "config_test_patchX.yaml"
    steady = {"flow": "steady", "heat": "steady"}
    full, mesh = _run(tmp_path, config_name, {"matrix": steady}, assignment, solver_type=solver_type)
    single, _ = _run(
        tmp_path, config_name, {"matrix": {**steady, off: "off"}}, assignment, solver_type=solver_type
    )

    frozen = "T" if solved == "p" else "p"
    assert np.allclose(single[solved][:, 0], full[solved][:, 0], rtol=1e-10)
    x = mesh["nodes_total"][:, 0]
    on_bc = np.isclose(x, x.min()) | np.isclose(x, x.max())  # Dirichlet faces x_min, x_max
    assert np.all(single[frozen][on_bc] == full[frozen][on_bc])
    assert np.all(single[frozen][~on_bc] == DEFAULTS["initial_conditions"][frozen + "_ini"])


@pytest.mark.parametrize("solver_type", ["linearized", "fully_coupled"])
def test_matrix_off_matches_dirichlet_matrix(tmp_path: Path, solver_type: str) -> None:
    """Switching the matrix `off` reproduces the well solution of fixing every matrix node by
    Dirichlet BCs at the same values (the config's own approach); those BCs are then ignored."""
    config_name = "config_test_wellconvection.yaml"
    wells = {"flow": "steady", "heat": "steady", "advection": True}
    dirichlet, mesh = _run(tmp_path, config_name, {"matrix": wells}, solver_type=solver_type)
    off, _ = _run(
        tmp_path,
        config_name,
        {"matrix": {"flow": "off", "heat": "off"}, "wells": wells},
        solver_type=solver_type,
    )

    nn = mesh["nn"]
    for key in ("p", "T"):
        assert np.all(off[key][:nn] == dirichlet[key][:nn])
        assert np.allclose(off[key][nn:, 0], dirichlet[key][nn:, 0], rtol=1e-6)


def test_with_domain_off_switches_only_that_domain() -> None:
    """`with_domain_off` turns both fields of one domain off and leaves the others unchanged."""
    physics = _parse(
        "matrix: {flow: steady, heat: transient}\nwells: {flow: steady, heat: transient, advection: on}"
    )
    wells_off = physics.with_domain_off("wells")

    assert wells_off.wells.flow == "off"
    assert wells_off.wells.heat == "off"
    assert wells_off.wells.advection == physics.wells.advection
    assert wells_off.matrix == physics.matrix
    assert wells_off.fractures == physics.fractures


def test_steady_state_controls_turn_wells_off() -> None:
    """The initial-condition solve is steady with wells off, without changing the run's own physics."""
    physics = _parse("matrix: {flow: transient, heat: transient}\nwells: {flow: steady, heat: transient}")
    controls = {"physics": physics, "fully_coupled": True, "volumetric_strain_on": 1}
    controls_SS = _build_steady_state_controls(controls)

    assert controls_SS["physics"].matrix.flow == "steady"
    assert controls_SS["physics"].matrix.heat == "steady"
    assert controls_SS["physics"].wells.flow == "off"
    assert controls_SS["physics"].wells.heat == "off"
    assert controls["physics"] is physics
