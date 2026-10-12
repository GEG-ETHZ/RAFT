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

from raft.subsurface import TimestepError, _clamp_to_critical, _update_time


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


def test_clamp_lands_exactly_on_critical_time() -> None:
    """A step that would overshoot is shortened to end exactly on the critical time."""
    t, dt = _clamp_to_critical(0.1, [0.3, 1.0, 1.0], np.array([0.3, 5.0]))
    assert t == 0.3
    assert dt[0] == pytest.approx(0.2)


def test_clamp_splits_remainder_to_avoid_sliver() -> None:
    """Less than two steps left: the remainder is split into two equal steps."""
    t, dt = _clamp_to_critical(0.0, [6.0, 1.0, 1.0], np.array([10.0]))
    assert dt[0] == 5.0
    assert t == 5.0


def test_clamp_leaves_step_far_from_critical_time() -> None:
    """Two or more steps left: the step is unchanged."""
    t, dt = _clamp_to_critical(0.0, [4.0, 1.0, 1.0], np.array([10.0]))
    assert dt[0] == 4.0
    assert t == 4.0


def test_clamp_ignores_critical_time_at_step_start() -> None:
    """A critical time equal to the step start is already passed."""
    t, dt = _clamp_to_critical(10.0, [4.0, 1.0, 1.0], np.array([10.0]))
    assert dt[0] == 4.0
    assert t == 14.0


def test_update_time_clamp_may_go_below_dt_min() -> None:
    """Landing on a critical time takes priority over the minimum timestep."""
    dt = [2.0, 2.0, 2.0]
    t, dt, accepted = _update_time("fully_coupled", 9.5, dt, _controls(), np.array([10.0]), True, 0.0, False)
    assert accepted
    assert t == 10.0
    assert dt[0] == 0.5


def test_update_time_raises_below_dt_min_after_clamp() -> None:
    """A failed step already below dt_min (after a clamp) raises rather than growing back to dt_min."""
    dt = [0.5, 2.0, 2.0]
    with pytest.raises(TimestepError):
        _update_time("fully_coupled", 10.0, dt, _controls(), np.array([10.0]), False, -1, False)
