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

"""Unit tests for timestep control."""

from typing import Any

import numpy as np
import pytest

from raft.subsurface import TimestepError, _update_time


def _controls() -> dict[str, Any]:
    return {"timestep_max": 1.0e4, "timestep_min": 1.0, "rel_error_T": 0.05, "t_start": 0.0}


def test_update_time_raises_on_NR_failure_at_dt_min() -> None:
    """A Newton-Raphson failure at the minimum timestep raises instead of exiting."""
    dt = [1.0, 1.0, 1.0]
    with pytest.raises(TimestepError, match="Minimum timestep reached"):
        _update_time("fully_coupled", 10.0, dt, _controls(), np.array([100.0]), False, -1, False)


def test_update_time_proceeds_on_extrapolation_error_at_dt_min() -> None:
    """A linearized extrapolation error at the minimum timestep is accepted, not raised."""
    dt = [1.0, 1.0, 1.0]
    _, _, accepted = _update_time("linearized", 10.0, dt, _controls(), np.array([100.0]), False, 0.5, False)
    assert accepted
