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

import raft.api
from raft.api import RaftModel
from raft.subsurface import TimestepError, main

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


def test_single_solve_matches_batch_run(tmp_path: Path) -> None:
    config_name = _transient_config(tmp_path)
    batch, _ = main(config_name=config_name, config_dir=tmp_path, verbose=False)

    model = RaftModel(config_name, 0.0, config_dir=tmp_path, verbose=False)
    assert model.solve(0.0, T_END, {}) == {}
    _assert_same_state(_trial(model), batch)


def test_unaligned_coupling_steps_approximate_batch_run(tmp_path: Path) -> None:
    """250 s coupling steps run as 100 + 75 + 75 s internally, so they match the 100 s batch run
    only to within the time discretization error."""
    config_name = _transient_config(tmp_path)
    batch, _ = main(config_name=config_name, config_dir=tmp_path, verbose=False)

    model = RaftModel(config_name, 0.0, config_dir=tmp_path, verbose=False)
    model.solve(0.0, 250.0, {})
    model.get_fresh()
    model.solve(250.0, T_END, {})
    trial = _trial(model)
    assert model._trial.dt_hist[1:3] == [75.0, 75.0]
    np.testing.assert_allclose(trial["T"][:, 0], batch["T"][:, 0], rtol=1e-2)


def test_fixed_coupling_steps_match_batch_run_exactly(tmp_path: Path) -> None:
    """Coupling steps equal to the fixed internal step reproduce the batch run bitwise."""
    config_name = _transient_config(tmp_path)
    batch, _ = main(config_name=config_name, config_dir=tmp_path, verbose=False)

    model = RaftModel(config_name, 0.0, config_dir=tmp_path, verbose=False)
    for k in range(5):
        model.solve(k * DT, (k + 1) * DT, {})
        if k < 4:
            model.get_fresh()
    _assert_same_state(_trial(model), batch)


def test_repeated_solve_is_identical(tmp_path: Path) -> None:
    config_name = _transient_config(tmp_path)
    model = RaftModel(config_name, 0.0, config_dir=tmp_path, verbose=False)
    model.solve(0.0, 2 * DT, {})
    first = _trial(model)
    model.solve(0.0, 2 * DT, {})
    _assert_same_state(_trial(model), first)


def test_get_fresh_writes_one_output_row_per_coupling_step(tmp_path: Path) -> None:
    config_name = _transient_config(tmp_path)
    model = RaftModel(config_name, 0.0, config_dir=tmp_path, verbose=False)
    for k in range(3):
        model.solve(k * 1.5 * DT, (k + 1) * 1.5 * DT, {})
        model.get_fresh()

    rows = (tmp_path / "out" / "transient" / "outflow.log").read_text().splitlines()[1:]
    times = [float(row.split("\t")[0]) for row in rows]
    assert times == [0.0, 150.0, 300.0, 450.0]


def test_misuse_raises(tmp_path: Path) -> None:
    config_name = _transient_config(tmp_path)
    model = RaftModel(config_name, 0.0, config_dir=tmp_path, verbose=False)
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
    model = RaftModel(config_name, 0.0, config_dir=tmp_path, verbose=False)
    model.solve(0.0, DT, {})
    model.get_fresh()
    model.solve(DT, 2 * DT, {})
    expected = _trial(model)

    with monkeypatch.context() as patch:
        patch.setattr(raft.api, "advance", _failing_advance)
        with pytest.raises(TimestepError):
            model.solve(DT, 2 * DT, {})
    assert not model._trial_pending
    with pytest.raises(RuntimeError, match="No trial state"):
        model.get_fresh()

    model.solve(DT, 2 * DT, {})
    _assert_same_state(_trial(model), expected)
