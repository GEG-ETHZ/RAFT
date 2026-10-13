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

import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .logging_utils import RunLogger
from .postprocessing import postprocessing, prepare_output_directory
from .preprocessing import preprocessing
from .subsurface import (
    _load_wb,
    _rollover,
    _snapshot_wb,
    advance,
    initialize_simulation,
    load_state,
    snapshot_state,
)

# Coupling input used by each external BC type
_INPUT_KEY = {"pressure": "p", "flux": "q", "temperature": "T"}
_INPUT_KEYS = ("p", "T", "q")


@dataclass(frozen=True, slots=True)
class _ExternalField:
    """One BC field set through the API, and how its coupling input is converted."""

    tag: str
    group_index: int  # index into BC_config
    field: str  # "hydraulics" or "thermal"
    input_key: str  # "p", "T" or "q"
    measure: float  # group length/area (node count without flux weights), converts q in kg/s to flux


def _external_fields(BC_config: list[dict[str, Any]]) -> list[_ExternalField]:
    """Describe every external field; types are read before any wellbore constraint switching."""
    fields = []
    for i, group in enumerate(BC_config):
        if "flux_weight_matrix" in group:
            measure = float(np.sum(group["flux_weight_matrix"] @ np.ones(len(group["nodes"]))))
        else:
            measure = float(len(group["nodes"]))
        for field in ("hydraulics", "thermal"):
            spec = group.get(field, {})
            if spec.get("external", False):
                fields.append(_ExternalField(group["tag"], i, field, _INPUT_KEY[spec["type"]], measure))
    return fields


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
            self._external = _external_fields(self._setup.BC_config)
            self._tags = sorted({ext.tag for ext in self._external})
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

            # Write this step's external BC tables onto the current wellbore flags
            self._check_bcs(bcs)
            BC_config = self._setup.BC_config
            _load_wb(BC_config, current.wb_flags)
            self._apply_external_bcs(current.t, t_end, bcs, current.ext_values)
            inputs = {tag: {key: float(bcs[tag][key]) for key in _INPUT_KEYS} for tag in self._tags}
            start = replace(current, wb_flags=_snapshot_wb(BC_config), ext_values=inputs)

            # Advance a trial from the current state; a failed solve leaves no trial pending
            self._trial_pending = False
            self._trial = advance(self._setup, start, t_end, self.verbose)
            self._trial_pending = True

            # Report each external group at t_end
            return self._boundary_values(self._trial.primary_variables)

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

    def _check_bcs(self, bcs: dict[str, dict[str, float]]) -> None:
        """Every external tag, and only those, with finite p, T and q."""
        missing = sorted(set(self._tags) - set(bcs))
        unknown = sorted(set(bcs) - set(self._tags))
        if missing or unknown:
            raise ValueError(f"Boundary condition tags missing: {missing}; not external: {unknown}.")
        for tag in self._tags:
            absent = [key for key in _INPUT_KEYS if key not in bcs[tag]]
            if absent:
                raise ValueError(f"Boundary conditions for '{tag}' lack {absent}.")
            if not np.all(np.isfinite([bcs[tag][key] for key in _INPUT_KEYS])):
                raise ValueError(f"Boundary conditions for '{tag}' are not finite: {bcs[tag]}.")

    def _apply_external_bcs(
        self,
        t_start: float,
        t_end: float,
        bcs: dict[str, dict[str, float]],
        previous: dict[str, dict[str, float]],
    ) -> None:
        """Write each external field as a two-point table over [t_start, t_end]."""
        BC_config = self._setup.BC_config
        for ext in self._external:
            spec = BC_config[ext.group_index][ext.field]
            value_end = self._input_value(ext, bcs)
            # Linear interpolation starts from the last accepted input; the first step is stepwise
            if spec["interpolation"] == "linear" and previous:
                value_start = self._input_value(ext, previous)
            else:
                value_start = value_end
            table = [[t_start, t_end], [value_start, value_end]]

            if "constraints" in spec:
                # Wellbore group: keep the flux target while pressure-limited
                spec["function_flux"], spec["values_flux"] = "table", table
                if spec["curr_BC"] == "flux":
                    spec["function"], spec["values"] = "table", table
            else:
                spec["function"], spec["values"] = "table", table

    def _input_value(self, ext: _ExternalField, bcs: dict[str, dict[str, float]]) -> float:
        """Coupling input for one field; a flow rate in kg/s becomes a flux over the group."""
        value = float(bcs[ext.tag][ext.input_key])
        if ext.input_key == "q":
            return value / ext.measure
        return value

    def _boundary_values(self, primary_variables: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Mean p and T, summed q (kg/s, positive inflow) and hydraulic mode of each external group."""
        BC_config = self._setup.BC_config
        group_indices = {ext.tag: ext.group_index for ext in self._external}
        results = {}
        for tag in self._tags:
            group = BC_config[group_indices[tag]]
            nodes = np.asarray(group["nodes"])
            q = float(np.sum(primary_variables["q"][nodes, 0]))
            hydraulics = group.get("hydraulics", {})
            results[tag] = {
                "p": float(np.mean(primary_variables["p"][nodes, 0])),
                "T": float(np.mean(primary_variables["T"][nodes, 0])),
                "q": q,
                "mode": hydraulics.get("type", "none"),
            }

            # A fixed temperature only makes sense where fluid flows in
            thermal = group.get("thermal", {})
            if thermal.get("external", False) and q < 0:
                warnings.warn(
                    f"'{tag}' fixes temperature while fluid flows out (q = {q:.3e} kg/s).", stacklevel=3
                )
        return results
