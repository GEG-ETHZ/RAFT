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

"""Tests for the console verbosity levels."""

from pathlib import Path
from typing import Any

import pytest
import yaml

import raft.subsurface
from raft.api import RaftModel
from raft.subsurface import main
from tests.test_api import DT, _transient_config

PERFORM_TIME_INTEGRATION = raft.subsurface._perform_time_integration


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


@pytest.mark.parametrize("verbosity", ["Lite", "verbose", True, False])
def test_unknown_verbosity_is_rejected(tmp_path: Path, verbosity: Any) -> None:
    """Only "quiet", "lite" and "full" are accepted, by the batch run and the API alike."""
    config_name = _transient_config(tmp_path)
    with pytest.raises(ValueError, match="expected 'quiet', 'lite' or 'full'"):
        main(config_name=config_name, config_dir=tmp_path, verbosity=verbosity)
    with pytest.raises(ValueError, match="expected 'quiet', 'lite' or 'full'"):
        RaftModel(config_name, 0.0, config_dir=tmp_path, verbosity=verbosity)


def test_lite_prints_only_one_line_per_step(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(config_name=_transient_config(tmp_path), config_dir=tmp_path, verbosity="lite")
    lines = capsys.readouterr().out.splitlines()

    assert lines == [
        f"     Timestep {n}, t = {n * DT:.2e} with dt = {DT:.2e}, {100 * n * DT / 500.0:.0f} % complete"
        for n in range(1, 6)
    ]


def test_lite_reports_cut_and_retried_steps(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed attempt is cut and retried, and both attempts are printed."""
    config_name = _transient_config(tmp_path)
    config = yaml.safe_load((tmp_path / config_name).read_text())
    config["simulation"]["timestep_min"] = 10.0
    (tmp_path / config_name).write_text(yaml.safe_dump(config))
    monkeypatch.setattr(raft.subsurface, "_perform_time_integration", _FailFirstAttempt())

    model = RaftModel(config_name, 0.0, config_dir=tmp_path, verbosity="lite")
    model.solve(0.0, DT, {})
    lines = capsys.readouterr().out.splitlines()

    assert lines[0] == "     Timestep 1, t = 1.00e+02 with dt = 1.00e+02, 100 % complete"
    assert lines[1] == "     Timestep 1, t = 5.00e+01 with dt = 5.00e+01, 50 % complete"
    assert all(line.startswith("     Timestep ") for line in lines)


def test_quiet_prints_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(config_name=_transient_config(tmp_path), config_dir=tmp_path, verbosity="quiet")
    assert capsys.readouterr().out == ""
