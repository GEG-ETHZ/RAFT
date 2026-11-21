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

"""Unit tests for timestep and Newton-Raphson control."""

from typing import Any

import numpy as np
import pytest

from geg_raft.solvers import _stagnated
from geg_raft.subsurface import TimestepError, _clamp_to_critical, _update_time


def _controls() -> dict[str, Any]:
    return {
        "time_controls": {"timestep_max": 1.0e4, "timestep_min": 1.0, "t_start": 0.0, "cut_timestep_hold": 5},
        "rel_error_T": 0.05,
    }


def test_update_time_raises_on_NR_failure_at_dt_min() -> None:
    """A Newton-Raphson failure at the minimum timestep raises instead of exiting."""
    dt = [1.0, 1.0, 1.0]
    with pytest.raises(TimestepError, match="Minimum timestep reached"):
        _update_time("monolithic", 10.0, dt, _controls(), np.array([100.0]), False, -1, False)


@pytest.mark.parametrize(
    ("t_prev", "dt0", "expected_t", "expected_dt"),
    [
        (0.0, 4.0, 4.0, 4.0),  # two or more steps left: unchanged
        (3.0, 8.0, 10.0, 7.0),  # would overshoot: lands exactly on the critical time
        (0.0, 6.0, 5.0, 5.0),  # less than two steps left: remainder split in two
        (9.5, 2.0, 10.0, 0.5),  # landing may take a step below dt_min
        (10.0, 4.0, 14.0, 4.0),  # a critical time at the step start is already passed
    ],
)
def test_clamp_to_critical(t_prev: float, dt0: float, expected_t: float, expected_dt: float) -> None:
    t, dt = _clamp_to_critical(t_prev, [dt0, 1.0, 1.0], np.array([10.0, 20.0]))
    assert t == expected_t
    assert dt[0] == expected_dt


@pytest.mark.parametrize(
    ("history", "expected"),
    [
        ([1.0, 1.0e-2, 1.0e-5] + [1.5e-5, 1.2e-5] * 5, True),  # noise floor within 10x tolerance
        ([1.0, 1.0e-2, 1.0e-4] + [1.5e-4, 1.2e-4] * 5, False),  # noise floor beyond 10x tolerance
        ([1.0, 1.0e-2] + [1.0e-4 / 2**k for k in range(10)], False),  # still decreasing
        ([1.0, 1.0e-5] + [1.5e-5, 1.2e-5] * 4, False),  # fewer iterations than the window
    ],
)
def test_stagnated(history: list[float], expected: bool) -> None:
    """A Newton solve stuck near tolerance is accepted; a converging or far-off one is not."""
    assert _stagnated(np.array(history), len(history) - 1, 2.0e-6) is expected
