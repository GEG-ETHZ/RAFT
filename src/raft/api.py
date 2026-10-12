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

"""Coupling API: an external simulator drives RAFT one coupling step at a time.

Each `solve` advances a trial state from the last accepted state; `get_fresh` accepts the
trial once the external coupling iteration has converged.
"""

from pathlib import Path
from typing import Any

import numpy as np

from .logging_utils import RunLogger
from .postprocessing import postprocessing, prepare_output_directory
from .preprocessing import preprocessing
from .subsurface import (
    _is_external,
    _rollover,
    advance,
    initialize_simulation,
    load_state,
    snapshot_state,
)


class RaftModel:
    """A RAFT simulation advanced by an external solver through trial coupling steps."""

    def __init__(
        self, config_name: str, t_start: float, config_dir: Path | None = None, verbose: bool = True
    ) -> None:
        """Preprocess, assign initial conditions at t_start and write output frame 0."""
        self.verbose = verbose
        with RunLogger() as run_logger:
            if verbose:
                print("RAFT - RESERVOIR ANALYSIS FOR TANGO (coupling API) \n")
                print("Initializing simulation " + config_name + "\n")

            # Load config and mesh
            config, mesh = preprocessing(config_name, config_dir, verbose)
            if any(_is_external(group) for group in config["boundary_conditions"]):
                raise NotImplementedError("External boundary conditions are not supported yet.")
            controls = config["simulation"]
            controls["config_name"] = config["config_name"]
            controls["config_dir"] = config["config_dir"]
            controls["t_start"] = t_start  # the coupling sets the start time

            # Mirror output to run.log; later calls append to it
            self._output_log = bool(controls["output_log"])
            self._log_dir = prepare_output_directory(controls)
            if self._output_log:
                run_logger.attach_file(self._log_dir)

            # The initial condition is the current state, with no trial pending
            self._setup, self._current = initialize_simulation(config, mesh, t_start, verbose)
            self._trial = self._current
            self._trial_pending = False

    def solve(
        self, t_start: float, t_end: float, bcs: dict[str, dict[str, float]]
    ) -> dict[str, dict[str, Any]]:
        """Advance a trial state from the current state at t_start to exactly t_end."""
        with RunLogger() as run_logger:
            self._resume_log(run_logger)

            # The coupling step must start from the current state
            current = self._current
            if not np.isclose(t_start, current.t, rtol=1.0e-12, atol=0.0):
                raise ValueError(
                    f"t_start = {t_start:.9e} does not match the current state at t = {current.t:.9e}."
                )
            if t_end <= current.t:
                raise ValueError(f"t_end = {t_end:.9e} must be after t_start = {current.t:.9e}.")

            # Apply external boundary conditions
            if bcs:
                raise NotImplementedError("External boundary conditions are not supported yet.")

            # Advance a trial from the current state; a failed solve leaves no trial pending
            self._trial_pending = False
            self._trial = advance(self._setup, current, t_end, self.verbose)
            self._trial_pending = True
            return {}

    def get_fresh(self) -> None:
        """Accept the trial state as the current state and write its output."""
        if not self._trial_pending:
            raise RuntimeError("No trial state to accept: call solve before each get_fresh.")

        with RunLogger() as run_logger:
            self._resume_log(run_logger)
            setup = self._setup
            trial = self._trial

            # Write output at the end of the coupling step
            primary_variables = load_state(trial, setup.BC_config)
            n_out = trial.n_out + 1
            postprocessing(
                n_out,
                setup.mesh,
                setup.controls,
                setup.material,
                setup.BC_config,
                primary_variables,
                trial.t,
                trial.dt_hist[1:4],
                self.verbose,
            )

            # Shift the time history and accept the trial as the current state
            _rollover(primary_variables, setup.BC_config, setup.controls["wellbore_check"])
            self._current = snapshot_state(
                trial.t,
                trial.n_step,
                n_out,
                trial.dt_hist,
                primary_variables,
                setup.BC_config,
                trial.ext_values,
            )
            self._trial_pending = False

    def _resume_log(self, run_logger: RunLogger) -> None:
        """Append this call's output to run.log, when logging is on."""
        if self._output_log:
            run_logger.attach_file(self._log_dir, mode="a")
