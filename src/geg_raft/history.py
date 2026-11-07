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

"""Irreversible history variables with trial and committed states."""

import dataclasses
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(slots=True)
class History:
    """Named history arrays; Newton updates `trial`, accepted steps copy it to `committed`."""

    committed: dict[str, npt.NDArray[np.float64]] = dataclasses.field(default_factory=dict)
    trial: dict[str, npt.NDArray[np.float64]] = dataclasses.field(default_factory=dict)

    def add(self, name: str, initial: npt.NDArray[np.float64]) -> None:
        """Register a history variable with its initial value."""
        self.committed[name] = initial.copy()
        self.trial[name] = initial.copy()

    def begin_step(self) -> None:
        """Reset the trial state to the committed state at the start of a step attempt."""
        self.trial = {name: value.copy() for name, value in self.committed.items()}

    def commit(self) -> None:
        """Accept the trial state."""
        self.committed = {name: value.copy() for name, value in self.trial.items()}
