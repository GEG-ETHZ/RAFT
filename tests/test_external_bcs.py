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

"""Tests for boundary condition fields with function "external"."""

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from raft.parseboundaryconditions import evaluateBCfunctions
from raft.preprocessing import _validate_external_groups
from raft.subsurface import main

CONFIG_DIR = Path(__file__).parent / "test_configs"


def _flux_group(**hydraulics: Any) -> dict[str, Any]:
    spec = {"type": "flux", "function": "external", **hydraulics}
    return {"well": [0, "inlet"], "tag": "inlet", "hydraulics": spec}


def test_validate_marks_external_fields_with_step_default() -> None:
    """External fields are marked and default to step interpolation."""
    group = _flux_group()
    group["thermal"] = {"type": "temperature", "function": "external", "interpolation": "linear"}
    _validate_external_groups([group])

    assert group["hydraulics"]["external"] is True
    assert group["hydraulics"]["interpolation"] == "step"
    assert group["thermal"]["interpolation"] == "linear"


def test_validate_ignores_non_external_groups() -> None:
    """Groups without external fields are left untouched."""
    group = {"faces": ["x_min"], "hydraulics": {"type": "flux", "values": [1.0]}}
    _validate_external_groups([group])
    assert "external" not in group["hydraulics"]


def test_validate_requires_tag() -> None:
    group = _flux_group()
    del group["tag"]
    with pytest.raises(ValueError, match="require a 'tag'"):
        _validate_external_groups([group])


def test_validate_requires_unique_tag() -> None:
    other = {"faces": ["x_max"], "tag": "inlet", "hydraulics": {"type": "pressure", "values": [1.0]}}
    with pytest.raises(ValueError, match="not unique"):
        _validate_external_groups([_flux_group(), other])


def test_validate_rejects_external_heat_flux() -> None:
    group = {"faces": ["x_min"], "tag": "wall", "thermal": {"type": "flux", "function": "external"}}
    with pytest.raises(ValueError, match="has type 'flux'"):
        _validate_external_groups([group])


def test_validate_rejects_unknown_interpolation() -> None:
    with pytest.raises(ValueError, match="interpolation 'cubic'"):
        _validate_external_groups([_flux_group(interpolation="cubic")])


def test_validate_rejects_constraints_on_pressure_group() -> None:
    group = _flux_group(constraints={"maximum_pressure": 1.0e7})
    group["hydraulics"]["type"] = "pressure"
    with pytest.raises(ValueError, match="require hydraulics type 'flux'"):
        _validate_external_groups([group])


def test_evaluate_external_acts_as_constant() -> None:
    spec = evaluateBCfunctions(
        {"function": "external", "values": [2.5], "nodes": [3, 4]}, 10.0, np.empty((0, 0))
    )
    assert spec["BCvalues"] == [2.5, 2.5]


def test_evaluate_external_without_values_raises() -> None:
    with pytest.raises(ValueError, match="has no values"):
        evaluateBCfunctions({"function": "external", "nodes": [3]}, 10.0, np.empty((0, 0)))


def test_external_with_config_values_matches_constant_twin(tmp_path: Path) -> None:
    """A standalone run with external fields and config values matches the same run with constants."""
    config_name = "config_test_wellconvection.yaml"
    config = yaml.safe_load((CONFIG_DIR / config_name).read_text())
    shutil.copy(CONFIG_DIR / config["mesh"]["file"], tmp_path)

    constant_dir = tmp_path / "constant"
    config["simulation"]["output_directory"] = str(constant_dir)
    (tmp_path / "constant.yaml").write_text(yaml.safe_dump(config))

    external_dir = tmp_path / "external"
    config["simulation"]["output_directory"] = str(external_dir)
    inlet, outlet = config["boundary_conditions"][1], config["boundary_conditions"][2]
    inlet["tag"] = "inlet"
    inlet["hydraulics"]["function"] = "external"
    inlet["thermal"]["function"] = "external"
    outlet["hydraulics"]["function"] = "external"
    (tmp_path / "external.yaml").write_text(yaml.safe_dump(config))

    pv_constant, _ = main(config_name="constant.yaml", config_dir=tmp_path, verbose=False)
    pv_external, _ = main(config_name="external.yaml", config_dir=tmp_path, verbose=False)

    np.testing.assert_array_equal(pv_external["p"], pv_constant["p"])
    np.testing.assert_array_equal(pv_external["T"], pv_constant["T"])
    np.testing.assert_array_equal(pv_external["q"], pv_constant["q"])
