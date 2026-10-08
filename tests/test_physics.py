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

"""Test parsing of the config `physics` block."""

from typing import Any

import numpy as np
import pytest
import yaml

from raft.physics import DomainPhysics, Physics
from raft.preprocessing import DEFAULTS, _apply_defaults, _parse_physics


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


@pytest.mark.parametrize(
    ("physics_yaml", "error"),
    [
        ("matrix: {flow: implicit}", ValueError),
        ("matrix: {flow: off}", NotImplementedError),
        ("matrix: {advection: 2}", ValueError),
        ("matrix: {pressure: steady}", ValueError),
        ("well: {advection: on}", ValueError),
    ],
)
def test_invalid(physics_yaml: str, error: type[Exception]) -> None:
    with pytest.raises(error):
        _parse(physics_yaml)


def test_advection_node_mask() -> None:
    physics = _parse("matrix: {advection: off}\nfractures: {advection: on}\nwells: {advection: off}")
    mesh = {"nn": 3, "nn_total": 7, "fracture_mesh": {"nn": 2}}
    assert np.array_equal(physics.advection_node_mask(mesh), [0, 0, 0, 1, 1, 0, 0])
