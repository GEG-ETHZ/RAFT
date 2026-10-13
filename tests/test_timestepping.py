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
