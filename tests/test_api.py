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

"""Tests for the coupling API."""

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

import geg_raft.api
import geg_raft.subsurface
from geg_raft.api import RaftModel
from geg_raft.preprocessing import _validate_external_groups
from geg_raft.subsurface import TimestepError, main

CONFIG_DIR = Path(__file__).parent / "test_configs"
DT = 100.0
T_END = 500.0


def _transient_config(tmp_path: Path) -> str:
    """Well convection test with transient well heat and a fixed 100 s timestep, written to tmp_path."""
    config = yaml.safe_load((CONFIG_DIR / "config_test_wellconvection.yaml").read_text())
    shutil.copy(CONFIG_DIR / config["mesh"]["file"], tmp_path)
    config["physics"]["wells"] = {"flow": "steady", "heat": "transient", "advection": True}
    simulation = config["simulation"]
    simulation.update({"t_end": T_END, "timestep": DT, "timestep_min": DT, "timestep_max": DT})
    simulation["output_directory"] = str(tmp_path / "out")
    (tmp_path / "transient.yaml").write_text(yaml.safe_dump(config))
    return "transient.yaml"


def _assert_same_state(a: dict[str, Any], b: dict[str, Any]) -> None:
    np.testing.assert_array_equal(a["p"], b["p"])
    np.testing.assert_array_equal(a["T"], b["T"])
    np.testing.assert_array_equal(a["q"], b["q"])


def _failing_advance(*_args: Any, **_kwargs: Any) -> None:
    raise TimestepError("forced")


def _trial(model: RaftModel) -> dict[str, Any]:
    return model._trial.primary_variables


def test_fixed_coupling_steps_match_batch_run_exactly(tmp_path: Path) -> None:
    """Coupling steps equal to the fixed internal step reproduce the batch run bitwise."""
    config_name = _transient_config(tmp_path)
    batch, _ = main(config_name=config_name, config_dir=tmp_path, verbosity="quiet")

    model = RaftModel(config_name, 0.0, config_dir=tmp_path, verbosity="quiet")
    for k in range(5):
        model.solve(k * DT, (k + 1) * DT, {})
        if k < 4:
            model.get_fresh()
    _assert_same_state(_trial(model), batch)


def test_repeated_solve_is_identical(tmp_path: Path) -> None:
    config_name = _transient_config(tmp_path)
    model = RaftModel(config_name, 0.0, config_dir=tmp_path, verbosity="quiet")
    model.solve(0.0, 2 * DT, {})
    first = _trial(model)
    model.solve(0.0, 2 * DT, {})
    _assert_same_state(_trial(model), first)


def test_misuse_raises(tmp_path: Path) -> None:
    config_name = _transient_config(tmp_path)
    model = RaftModel(config_name, 0.0, config_dir=tmp_path, verbosity="quiet")
    with pytest.raises(RuntimeError, match="No trial state"):
        model.get_fresh()

    model.solve(0.0, DT, {})
    model.get_fresh()
    with pytest.raises(RuntimeError, match="No trial state"):
        model.get_fresh()
    with pytest.raises(ValueError, match="does not match the current state"):
        model.solve(0.0, 2 * DT, {})
    with pytest.raises(ValueError, match="must be after"):
        model.solve(DT, DT, {})


def test_timestep_error_keeps_current_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed solve leaves no trial, and the next solve starts from the unchanged current state."""
    config_name = _transient_config(tmp_path)
    model = RaftModel(config_name, 0.0, config_dir=tmp_path, verbosity="quiet")
    model.solve(0.0, DT, {})
    model.get_fresh()
    model.solve(DT, 2 * DT, {})
    expected = _trial(model)

    with monkeypatch.context() as patch:
        patch.setattr(geg_raft.api, "advance", _failing_advance)
        with pytest.raises(TimestepError):
            model.solve(DT, 2 * DT, {})
    assert not model._trial_pending
    with pytest.raises(RuntimeError, match="No trial state"):
        model.get_fresh()

    model.solve(DT, 2 * DT, {})
    _assert_same_state(_trial(model), expected)


# ----- External boundary conditions
INLET = {"p": 0.0, "T": 300.0, "q": 0.199}
OUTLET = {"p": 1.0e6, "T": 0.0, "q": 0.0}


def _external_config(tmp_path: Path, t_end: float = T_END, **inlet_hydraulics: Any) -> str:
    """The transient config with the inlet (flux, temperature) and outlet (pressure) set externally."""
    _transient_config(tmp_path)
    config = yaml.safe_load((tmp_path / "transient.yaml").read_text())
    config["simulation"]["t_end"] = t_end
    inlet, outlet = config["boundary_conditions"][1], config["boundary_conditions"][2]
    inlet["label"] = "inlet"
    inlet["hydraulics"].update({"function": "external", **inlet_hydraulics})
    inlet["thermal"]["function"] = "external"
    outlet["hydraulics"]["function"] = "external"
    (tmp_path / "external.yaml").write_text(yaml.safe_dump(config))
    return "external.yaml"


def test_external_stepwise_matches_constant_batch_run(tmp_path: Path) -> None:
    """External inputs equal to the constant config values reproduce the batch run bitwise."""
    batch, _ = main(config_name=_transient_config(tmp_path), config_dir=tmp_path, verbosity="quiet")

    model = RaftModel(_external_config(tmp_path), 0.0, config_dir=tmp_path, verbosity="quiet")
    for k in range(5):
        results = model.solve(k * DT, (k + 1) * DT, {"inlet": INLET, "outlet": OUTLET})
        if k < 4:
            model.get_fresh()
    _assert_same_state(_trial(model), batch)

    assert results["inlet"]["q"] == pytest.approx(0.199, rel=1e-12)
    assert results["inlet"]["T"] == 300.0
    assert results["inlet"]["mode"] == "flux"
    assert results["outlet"]["p"] == 1.0e6
    assert results["outlet"]["q"] == pytest.approx(-0.199, rel=1e-3)
    assert results["outlet"]["mode"] == "pressure"


def test_external_linear_matches_table_batch_run(tmp_path: Path) -> None:
    """Linear inputs ramp from the last accepted value, like the equivalent table BC."""
    config_name = _external_config(tmp_path, t_end=400.0)
    config = yaml.safe_load((tmp_path / config_name).read_text())
    config["boundary_conditions"][1]["thermal"]["interpolation"] = "linear"
    (tmp_path / config_name).write_text(yaml.safe_dump(config))

    table = yaml.safe_load((tmp_path / "transient.yaml").read_text())
    table["simulation"]["t_end"] = 400.0
    table["boundary_conditions"][1]["thermal"] = {
        "type": "temperature",
        "function": "table",
        "values": [[0.0, 200.0, 400.0], [310.0, 310.0, 330.0]],
    }
    (tmp_path / "table.yaml").write_text(yaml.safe_dump(table))
    batch, _ = main(config_name="table.yaml", config_dir=tmp_path, verbosity="quiet")

    # First step is stepwise at 310 K, the second ramps 310 -> 330 K over two internal steps
    model = RaftModel(config_name, 0.0, config_dir=tmp_path, verbosity="quiet")
    model.solve(0.0, 200.0, {"inlet": {**INLET, "T": 310.0}, "outlet": OUTLET})
    model.get_fresh()
    model.solve(200.0, 400.0, {"inlet": {**INLET, "T": 330.0}, "outlet": OUTLET})
    for key in ("p", "T"):
        np.testing.assert_allclose(_trial(model)[key], batch[key], rtol=1e-12)


def test_external_flow_rate_is_total_over_face_group(tmp_path: Path) -> None:
    """q in kg/s on a face group is spread as a flux, and the group's summed q returns it."""
    config_name = "config_test_enthalpy_bndry_flux.yaml"
    config = yaml.safe_load((CONFIG_DIR / config_name).read_text())
    shutil.copy(CONFIG_DIR / config["mesh"]["file"], tmp_path)
    config["simulation"]["output_directory"] = str(tmp_path / "out")
    face = config["boundary_conditions"][2]
    face["label"] = "face"
    face["hydraulics"]["function"] = "external"
    (tmp_path / config_name).write_text(yaml.safe_dump(config))

    model = RaftModel(config_name, 0.0, config_dir=tmp_path, verbosity="quiet")
    results = model.solve(0.0, 1.0, {"face": {"p": 0.0, "T": 0.0, "q": -2.0e-3}})
    assert results["face"]["q"] == pytest.approx(-2.0e-3, rel=1e-10)
    assert results["face"]["mode"] == "flux"


def test_external_pressure_constraint_switches_mode(tmp_path: Path) -> None:
    """Hitting the maximum pressure switches to pressure control; a lower flow target switches back
    using the latest input, not the config value."""
    p_max = OUTLET["p"] + 0.8  # below the 1.6 Pa needed to drive the requested flow
    config_name = _external_config(tmp_path, constraints={"maximum_pressure": p_max})
    model = RaftModel(config_name, 0.0, config_dir=tmp_path, verbosity="quiet")

    limited = model.solve(0.0, DT, {"inlet": INLET, "outlet": OUTLET})["inlet"]
    assert limited["mode"] == "pressure"
    assert limited["p"] == pytest.approx(p_max, rel=1e-12)
    assert 0.0 < limited["q"] < INLET["q"]
    model.get_fresh()

    target = 0.25 * limited["q"]
    released = model.solve(DT, 2 * DT, {"inlet": {**INLET, "q": target}, "outlet": OUTLET})["inlet"]
    assert released["mode"] == "flux"
    assert released["q"] == pytest.approx(target, rel=1e-12)


def test_external_input_errors(tmp_path: Path) -> None:
    model = RaftModel(_external_config(tmp_path), 0.0, config_dir=tmp_path, verbosity="quiet")
    with pytest.raises(ValueError, match="missing: \\['outlet'\\]"):
        model.solve(0.0, DT, {"inlet": INLET})
    with pytest.raises(ValueError, match="not external: \\['other'\\]"):
        model.solve(0.0, DT, {"inlet": INLET, "outlet": OUTLET, "other": OUTLET})
    with pytest.raises(ValueError, match="lack \\['q'\\]"):
        model.solve(0.0, DT, {"inlet": {"p": 0.0, "T": 300.0}, "outlet": OUTLET})
    with pytest.raises(ValueError, match="not finite"):
        model.solve(0.0, DT, {"inlet": {**INLET, "T": float("nan")}, "outlet": OUTLET})


@pytest.mark.parametrize(
    ("second", "message"),
    [
        # A duplicate label would apply one group's inputs to another
        (
            {"faces": ["x_max"], "label": "inlet", "hydraulics": {"type": "pressure", "values": [1.0]}},
            "not unique",
        ),
        # Constraints on a pressure group would never apply
        (
            {
                "faces": ["x_max"],
                "label": "outlet",
                "hydraulics": {
                    "type": "pressure",
                    "function": "external",
                    "constraints": {"maximum_pressure": 1.0},
                },
            },
            "require hydraulics type 'flux'",
        ),
    ],
)
def test_external_group_validation(second: dict[str, Any], message: str) -> None:
    inlet = {"well": [0, "inlet"], "label": "inlet", "hydraulics": {"type": "flux", "function": "external"}}
    with pytest.raises(ValueError, match=message):
        _validate_external_groups([inlet, second])


# ----- Diagnostics
PERFORM_TIME_INTEGRATION = geg_raft.subsurface._perform_time_integration


class _FailFirstAttempt:
    """Report a Newton-Raphson failure on the first call, then integrate normally."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args: Any) -> Any:
        self.calls += 1
        primary_variables, dt, converged, timestep_error = PERFORM_TIME_INTEGRATION(*args)
        if self.calls == 1:
            return primary_variables, dt, False, -1
        return primary_variables, dt, converged, timestep_error


def test_solve_stats_count_steps_and_cuts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cut attempt is counted, and the accepted steps exactly cover the coupling step."""
    config_name = _transient_config(tmp_path)
    config = yaml.safe_load((tmp_path / config_name).read_text())
    config["simulation"]["timestep_min"] = 10.0
    (tmp_path / config_name).write_text(yaml.safe_dump(config))
    monkeypatch.setattr(geg_raft.subsurface, "_perform_time_integration", _FailFirstAttempt())

    model = RaftModel(config_name, 0.0, config_dir=tmp_path, verbosity="quiet")
    model.solve(0.0, T_END, {})
    stats = model.solve_stats

    assert stats.cut_newton == 1
    assert stats.cut_extrapolation == 0
    assert stats.attempts == stats.accepted_steps + 1
    assert stats.dt_accepted[0] == 0.5 * DT
    assert sum(stats.dt_accepted) == pytest.approx(T_END, rel=1e-12)


def test_well_profile_follows_the_well(tmp_path: Path) -> None:
    """Nodes run from the well head, values match the solution, and steady flow carries the inlet rate."""
    model = RaftModel(_external_config(tmp_path), 0.0, config_dir=tmp_path, verbosity="quiet")
    model.solve(0.0, DT, {"inlet": INLET, "outlet": OUTLET})
    (profile,) = model.well_profiles()

    assert profile.s[0] == 0.0
    assert np.all(np.diff(profile.s) > 0)
    mesh = model._setup.mesh
    offset = mesh["nn_total"] - mesh["well_mesh"]["nn"]
    head = mesh["well_mesh"]["head_nodes"][0]
    np.testing.assert_array_equal(profile.p[0], _trial(model)["p"][offset + head, 0])
    np.testing.assert_allclose(profile.q, INLET["q"], rtol=1e-4)

    # After acceptance the profile comes from the accepted step, unchanged
    model.get_fresh()
    np.testing.assert_array_equal(model.well_profiles()[0].T, profile.T)
