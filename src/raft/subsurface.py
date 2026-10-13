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

"""Subsurface simulator - main time integration loop."""

import argparse
import copy
import datetime
import sys
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .logging_utils import RunLogger
from .matprops import get_EOS
from .parseboundaryconditions import evaluateBCfunctions, parsebcs
from .postprocessing import export_domain_properties, postprocessing, prepare_output_directory
from .preprocessing import preprocessing
from .solvers import fully_coupled_time_integration, lin_extrap_time_integration


class TimestepError(RuntimeError):
    """Raised when the solver fails to converge at the minimum timestep."""


def _propose_dt(  # noqa
    solver_type: str,
    t: float,
    dt: list[float],
    controls: dict[str, Any],
    converged: bool,
    timestep_error: float,
    verbose: bool,
) -> tuple[float, list[float], bool]:
    """Cut a failed timestep or grow the next one, ignoring critical times."""
    dt_max = controls["timestep_max"]
    dt_min = controls["timestep_min"]
    rel_tol_T = controls["rel_error_T"]

    if not converged:  # Decrease the timestep
        # <= since a step clamped to a critical time may lie below dt_min
        if dt[0] <= dt_min:
            # minimum timestep reached, proceed with simulation
            if solver_type == "linearized" and timestep_error != -1:
                if verbose:
                    print(
                        f"     Minimum timestep reached, error is {timestep_error * 100:.2f} %. Proceeding with simulation."  # noqa: E501
                    )
            else:
                # Fully coupled or Newton-Raphson failure at minmum timestep
                raise TimestepError(
                    f"Minimum timestep reached at t = {t:.6e}, unable to converge to solution."
                )
        else:
            if solver_type == "linearized" and timestep_error != -1:
                # Cut timestep in proportion with error
                cut_factor = 0.85 * 4 / (1 + 3 * timestep_error / rel_tol_T)
            else:
                # Newton-raphson failure
                cut_factor = 0.5
            t0 = controls["t_start"]
            t = t - dt[0]
            dt[0] = max(cut_factor * dt[0], dt_min)
            t = max(t + dt[0], t0)
            if verbose:
                print(
                    f"     Timestep error = {timestep_error * 100:.2f} %, reducing timestep by {cut_factor:.2f} to dt = {dt[0]:.2e}"  # noqa: E501
                )
            return t, dt, False

    # otherwise, increase the timestep
    increase_factor = min(4 / (1 + 3 * timestep_error / rel_tol_T), 2) if solver_type == "linearized" else 1.5

    dt[2] = dt[1]
    dt[1] = dt[0]
    if dt[0] < dt_max:
        # Floor at dt_min: proceeding past an error above tolerance gives a factor < 1
        dt[0] = min(max(increase_factor * dt[0], dt_min), dt_max)
    t += dt[0]
    return t, dt, True


def _clamp_to_critical(
    t_prev: float, dt: list[float], critical_times: npt.NDArray[np.float64]
) -> tuple[float, list[float]]:
    """Return the end time of the step starting at t_prev, shortened to land on the next critical time.

    If the step would leave less than one step to that time, the remainder is split in
    two equal steps instead, so no sliver step follows. Either may fall below dt_min.
    """
    upcoming = critical_times[critical_times > t_prev]
    if upcoming.size == 0:
        return t_prev + dt[0], dt

    t_next = float(upcoming.min())
    remaining = t_next - t_prev
    if dt[0] >= remaining:
        dt[0] = remaining
        return t_next, dt
    if remaining < 2 * dt[0]:
        dt[0] = remaining / 2
    return t_prev + dt[0], dt


def _update_time(  # noqa
    solver_type: str,
    t: float,
    dt: list[float],
    controls: dict[str, Any],
    critical_times: npt.NDArray[np.float64],
    converged: bool,
    timestep_error: float,
    verbose: bool,
) -> tuple[float, list[float], bool]:
    """Update the current time and timestep."""
    t_prev = t
    t, dt, accepted = _propose_dt(solver_type, t, dt, controls, converged, timestep_error, verbose)
    if not accepted:
        return t, dt, False

    t, dt = _clamp_to_critical(t_prev, dt, critical_times)
    if verbose:
        print(f"    Next timestep: {dt[0]:.2e}")
    return t, dt, True


def get_critical_times(BC_config: list[dict[str, Any]]) -> npt.NDArray[np.float64]:
    """Collect the change times of BC groups using the 'table' or 'ramp' function."""
    times: set[float] = set()
    for group in BC_config:
        for field in ("hydraulics", "thermal"):
            spec = group.get(field)
            if spec is None:
                continue
            if spec.get("function") == "table":
                times.update(spec["values"][0])
            elif spec.get("function") == "ramp":
                times.add(float(spec["values"][1]))
    return np.array(sorted(times), dtype=np.float64)


def _assign_initial_gradient(
    nodes: npt.NDArray[np.float64], y0: float, slope: float, direction: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Gradient calculation for initial conditions based on arbitrary direction."""
    s = direction / np.linalg.norm(direction)  # unit vector in gradient direction
    nsd = np.size(nodes, 1)
    s = s[0:nsd]

    x = np.dot(nodes, s)
    x_min = np.min(x)
    y = y0 + slope * (x - x_min)

    return np.asarray(y, dtype=np.float64)


def _fit_affine_from_dirichlet(
    nodes: npt.NDArray[np.float64], dirichlet: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Least-squares affine field through a set of Dirichlet (node, value) pairs.

    Used to seed BC_compatible initial conditions with a field that already varies
    smoothly toward the boundary values, rather than an arbitrary uniform guess.
    """
    node_idx = dirichlet[:, 0].astype(int)
    values = dirichlet[:, 1]
    pos = nodes[node_idx]
    A = np.hstack([np.ones((len(node_idx), 1)), pos])
    coeffs, *_ = np.linalg.lstsq(A, values, rcond=None)
    return np.asarray(np.hstack([np.ones((nodes.shape[0], 1)), nodes]) @ coeffs, dtype=np.float64)


def _broadcast_to_history(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Broadcast a per-node scalar field into the 4-column time-history layout."""
    return np.tile(values.reshape(-1, 1), (1, 4))


def _build_steady_state_controls(controls: dict[str, Any]) -> dict[str, Any]:
    """Build a simulation-control dict for the steady-state initial condition solve.

    Volumetric strain is neglected, since initial conditions are by definition
    unstrained relative to themselves. Wells are excluded from the assembly.
    """
    controls_SS = controls.copy()
    controls_SS["physics"] = controls["physics"].as_steady()
    controls_SS["wells_on"] = 0
    controls_SS["fully_coupled"] = False
    controls_SS["volumetric_strain_on"] = 0
    return controls_SS


def _assign_ic_constant(config_ini: dict[str, Any], primary_variables: dict[str, Any]) -> dict[str, Any]:
    """Apply constant uniform initial conditions."""
    primary_variables["p"][:, :] = config_ini["p_ini"]
    primary_variables["T"][:, :] = config_ini["T_ini"]
    return primary_variables


def _assign_ic_gradient(
    config_ini: dict[str, Any], mesh: dict[str, Any], primary_variables: dict[str, Any]
) -> dict[str, Any]:
    """Apply linear-gradient initial conditions."""
    T = _assign_initial_gradient(
        mesh["nodes_total"], config_ini["T_ini"], config_ini["T_grad"], config_ini["grad_direction_T"]
    )
    p = _assign_initial_gradient(
        mesh["nodes_total"], config_ini["p_ini"], config_ini["p_grad"], config_ini["grad_direction_p"]
    )

    primary_variables["p"][:, :] = _broadcast_to_history(p)
    primary_variables["T"][:, :] = _broadcast_to_history(T)
    return primary_variables


def _no_wells_problem(
    mesh: dict[str, Any], primary_variables: dict[str, Any], BC_config: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Copies of the solver inputs without wells (numbered last) and without external BC groups."""
    nn_no_wells = mesh["nn_total"] - mesh["well_mesh"]["nn"] if "well_mesh" in mesh else mesh["nn_total"]

    mesh_no_wells = dict(mesh)
    mesh_no_wells.pop("well_mesh", None)
    mesh_no_wells["nn_total"] = nn_no_wells
    mesh_no_wells["nodes_total"] = mesh["nodes_total"][:nn_no_wells]

    pv_no_wells = {
        "p_dofs": np.arange(nn_no_wells),
        "T_dofs": np.arange(nn_no_wells),
        "p": primary_variables["p"][:nn_no_wells].copy(),
        "T": primary_variables["T"][:nn_no_wells].copy(),
        "q": primary_variables["q"][:nn_no_wells].copy(),
    }

    BC_no_wells = []
    for group in BC_config:
        if _is_external(group):
            continue
        nodes = np.asarray(group["nodes"])
        in_no_wells = nodes < nn_no_wells
        if in_no_wells.all():
            BC_no_wells.append(group)
        elif in_no_wells.any():
            # Field specs are copied too, since parsebcs writes the node list into them
            trimmed = {**group, "nodes": nodes[in_no_wells]}
            for field in ("hydraulics", "thermal"):
                if field in group:
                    trimmed[field] = dict(group[field])
            BC_no_wells.append(trimmed)
    return mesh_no_wells, pv_no_wells, BC_no_wells


def _assign_ic_bc_compatible(
    config_ini: dict[str, Any],
    controls: dict[str, Any],
    mesh: dict[str, Any],
    material: dict[str, Any],
    primary_variables: dict[str, Any],
    BC_config: list[dict[str, Any]],
    t: float,
    n: int,
    verbose: bool,
) -> dict[str, Any]:
    """Steady-state initial condition without wells or external groups, the unperturbed state."""
    mesh_no_wells, pv_no_wells, BC_no_wells = _no_wells_problem(mesh, primary_variables, BC_config)
    boundary_conditions = parsebcs(BC_no_wells, mesh_no_wells, pv_no_wells, t, controls)
    pv_no_wells = _solve_steady_state(
        config_ini,
        controls,
        mesh_no_wells,
        material,
        pv_no_wells,
        boundary_conditions,
        BC_no_wells,
        n,
        verbose,
    )

    nn_no_wells = mesh_no_wells["nn_total"]
    primary_variables["p"][:nn_no_wells] = pv_no_wells["p"]
    primary_variables["T"][:nn_no_wells] = pv_no_wells["T"]
    primary_variables["q"][:nn_no_wells] = pv_no_wells["q"]
    primary_variables["T"][nn_no_wells:] = config_ini["T_ini"]  # fallback for wells never inside the domain
    if "well_mesh" in mesh:
        primary_variables = _assign_well_ic_from_matrix(
            mesh, material, controls, primary_variables, config_ini["p_ini"]
        )
    return primary_variables


def _assign_well_ic_from_matrix(
    mesh: dict[str, Any],
    material: dict[str, Any],
    controls: dict[str, Any],
    primary_variables: dict[str, Any],
    p_head: float,
) -> dict[str, Any]:
    """Set well T from the coincident matrix nodes and well p hydrostatic from p_head at the well head.

    T outside the domain is carried from the nearest in-domain node. p follows
    p_j = p_k + rho(p_k, T_k) g (z_j - z_k) along the well, once per connected well network.
    """
    wmesh = mesh["well_mesh"]
    nn_w = wmesh["nn"]
    offset = mesh["nn_total"] - nn_w  # wells are numbered last
    p = primary_variables["p"]
    T = primary_variables["T"]

    neighbours: list[list[int]] = [[] for _ in range(nn_w)]
    for a, b in wmesh["conn"]:
        neighbours[a].append(b)
        neighbours[b].append(a)

    # T: in-domain well nodes copy their coincident matrix node, then carry it outward
    T_assigned = np.zeros(nn_w, dtype=bool)
    for well_nodes, matrix_nodes in zip(wmesh["conn"], wmesh["matrix_node_conn"], strict=True):
        for w, m in zip(well_nodes, matrix_nodes, strict=True):
            if m >= 0:
                T[offset + w, :] = T[m, :]
                T_assigned[w] = True
    for k, j in _breadth_first_edges(neighbours, T_assigned, np.flatnonzero(T_assigned)):
        T[offset + j, :] = T[offset + k, 0]

    # p: hydrostatic from the head of the first well in each connected network
    z = wmesh["nodes"][:, mesh["nsd"] - 1]  # gravity acts along the last axis
    gravity = controls["gravity"]
    fluid = material["fluid"]
    p_assigned = np.zeros(nn_w, dtype=bool)
    for head in wmesh["head_nodes"]:
        if p_assigned[head]:
            continue
        p[offset + head, :] = p_head
        p_assigned[head] = True
        for k, j in _breadth_first_edges(neighbours, p_assigned, [head]):
            p_k = p[offset + k, 0:1].reshape(1, 1)
            T_k = T[offset + k, 0:1].reshape(1, 1)
            rho = get_EOS(T_k, T_k, p_k, fluid, want_density=True)[0].value[0, 0]
            p[offset + j, :] = p[offset + k, 0] + rho * gravity * (z[j] - z[k])
    return primary_variables


def _breadth_first_edges(
    neighbours: list[list[int]], assigned: npt.NDArray[np.bool_], sources: Iterable[int]
) -> Iterator[tuple[int, int]]:
    """Yield (k, j) for each unassigned node j reached from node k, breadth first; marks j assigned."""
    queue = deque(int(k) for k in sources)
    while queue:
        k = queue.popleft()
        for j in neighbours[k]:
            if assigned[j]:
                continue
            assigned[j] = True
            yield k, j
            queue.append(j)


def _solve_steady_state(  # noqa
    config_ini: dict[str, Any],
    controls: dict[str, Any],
    mesh: dict[str, Any],
    material: dict[str, Any],
    primary_variables: dict[str, Any],
    boundary_conditions: dict[str, Any],
    BC_config: list[dict[str, Any]],
    n: int,
    verbose: bool,
) -> dict[str, Any]:
    """Solve for a steady state compatible with the boundary conditions."""
    if verbose:
        print("   Solving for steady-state initial condition compatible with boundary conditions:")

    # Seed with an affine field through the Dirichlet BCs for initial guess
    T_dirichlet = boundary_conditions["T_dirichlet"]
    p_dirichlet = boundary_conditions["p_dirichlet"]
    if len(T_dirichlet) > mesh["nsd"]:
        primary_variables["T"][:, :] = _broadcast_to_history(
            _fit_affine_from_dirichlet(mesh["nodes_total"], T_dirichlet)
        )
    if len(p_dirichlet) > mesh["nsd"]:
        primary_variables["p"][:, :] = _broadcast_to_history(
            _fit_affine_from_dirichlet(mesh["nodes_total"], p_dirichlet)
        )
    # The seed is only a solver guess; `off` fields are never solved, so they hold *_ini
    controls_SS = _build_steady_state_controls(controls)
    physics = controls_SS["physics"]
    primary_variables["p"][physics.off_nodes(mesh, "flow"), :] = config_ini["p_ini"]
    primary_variables["T"][physics.off_nodes(mesh, "heat"), :] = config_ini["T_ini"]

    # Initialized reference values -> set as current values and neglect
    # strains when establishing initial conditions
    material["T_0"] = primary_variables["T"][:, 0]
    material["p_0"] = primary_variables["p"][:, 0]

    if verbose:
        print("   Using iterative sequential scheme to estimate fields...")

    # Use one-pass linear to generate approximate solution.
    primary_variables_ini = copy.deepcopy(primary_variables)
    tries = [1, 3, 5, 10]
    for num_passes in tries:
        primary_variables = copy.deepcopy(primary_variables_ini)
        for i in range(num_passes):
            primary_variables, _, _, _ = lin_extrap_time_integration(
                controls_SS,
                mesh,
                material,
                primary_variables,
                boundary_conditions,
                BC_config,
                [1, 1, 1],
                n,
                verbose,
            )
            primary_variables["T"][:, 1] = primary_variables["T"][:, 0]
        # Perform fully coupled solution
        if verbose:
            print("   Solving for fully-coupled steady-state condition...")

        controls_SS["fully_coupled"] = True
        primary_variables, _, converged, _ = fully_coupled_time_integration(
            controls_SS,
            mesh,
            material,
            primary_variables,
            boundary_conditions,
            BC_config,
            [1, 1, 1],
            n,
            verbose,
        )
        if converged:
            break
        else:
            print(
                "   Unable to find compatible boundary condition, trying again with more sequential iterations."  # noqa
            )
    if not converged:
        raise RuntimeError("Failed to converge during assignment of initial conditions.")

    p = primary_variables["p"][:, 0]
    T = primary_variables["T"][:, 0]
    primary_variables["p"][:, :] = _broadcast_to_history(p)
    primary_variables["T"][:, :] = _broadcast_to_history(T)
    return primary_variables


# ------------------ Solve for initial conditions
def _assign_initial_conditions(
    config_ini: dict[str, Any],
    controls: dict[str, Any],
    mesh: dict[str, Any],
    material: dict[str, Any],
    primary_variables: dict[str, Any],
    BC_config: list[dict[str, Any]],
    t: float,
    n: int,
    verbose: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assign initial conditions."""
    if verbose:
        print("Assigning initial conditions...")

    assign_type = config_ini["assignment"]
    supported_types = ["constant", "gradient", "BC_compatible"]

    if assign_type == "constant":
        primary_variables = _assign_ic_constant(config_ini, primary_variables)
    elif assign_type == "gradient":
        primary_variables = _assign_ic_gradient(config_ini, mesh, primary_variables)
    elif assign_type == "BC_compatible":
        primary_variables = _assign_ic_bc_compatible(
            config_ini,
            controls,
            mesh,
            material,
            primary_variables,
            BC_config,
            t,
            n,
            verbose,
        )
    else:
        raise ValueError(
            f"Unknown initial condition assignment type: '{assign_type}'\n"
            f"Supported types: {', '.join(supported_types)}\n"
        )

    material["p_0"] = primary_variables["p"][:, 0].copy()
    material["T_0"] = primary_variables["T"][:, 0].copy()
    return primary_variables, material


# ------------------------------------ Time integration decider
def _perform_time_integration(
    controls: dict[str, Any],
    mesh: dict[str, Any],
    material: dict[str, Any],
    primary_variables: dict[str, Any],
    boundary_conditions: dict[str, Any],
    BC_config: list[dict[str, Any]],
    dt: list[float],
    n: int,  # noqa
    verbose: bool,
) -> tuple[dict[str, Any], list[float], bool, float]:
    """Choose between time integration types."""
    solver_type = controls["solver_type"]
    supported_types = ["linearized", "fully_coupled"]

    if solver_type == "linearized":
        controls["fully_coupled"] = False
        primary_variables, dt, converged, timestep_error = lin_extrap_time_integration(
            controls,
            mesh,
            material,
            primary_variables,
            boundary_conditions,
            BC_config,
            dt,
            n,
            verbose,
        )
    elif solver_type == "fully_coupled":
        controls["fully_coupled"] = True
        primary_variables, dt, converged, timestep_error = fully_coupled_time_integration(
            controls,
            mesh,
            material,
            primary_variables,
            boundary_conditions,
            BC_config,
            dt,
            n,
            verbose,
        )
    else:
        raise ValueError(
            f"Unknown solver type: '{solver_type}'\nSupported types: {', '.join(supported_types)}\n"
        )

    return primary_variables, dt, converged, timestep_error


def _is_external(group: dict[str, Any]) -> bool:
    """Whether any field of a BC group is set externally through the coupling API."""
    return any(group.get(field, {}).get("external", False) for field in ("hydraulics", "thermal"))


def _wellbore_groups(BC_config: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the BC groups whose hydraulics field is pressure-constrained (wellbore-style)."""
    return [g for g in BC_config if "constraints" in g.get("hydraulics", {})]


def _initialize_WB_checks(BC_config: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Initialize current and previous flag checks for each wellbore group."""
    for group in _wellbore_groups(BC_config):
        hydraulics = group["hydraulics"]
        # Set flags
        hydraulics["curr_BC"] = "flux"
        hydraulics["prev_BC"] = "flux"
        # Initialize constraint type
        hydraulics["pres_constraint"] = "max"
        if "function" not in hydraulics:
            hydraulics["function"] = "constant"  # assume constant value if no function given
        # Copy existing constraints
        hydraulics["values_flux"] = hydraulics.get("values", [])  # external groups may have none yet
        hydraulics["function_flux"] = hydraulics["function"]
    return BC_config


def _update_WB_flags(BC_config: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Update wellbore flags from previous timestep."""
    for group in _wellbore_groups(BC_config):
        group["hydraulics"]["prev_BC"] = group["hydraulics"]["curr_BC"]

    return BC_config


def _announce_WB_change(group: dict[str, Any], constraint: str, verbose: bool) -> None:
    """Print WB constraint violation."""
    if verbose:
        label = group.get("label")
        if label is None:
            label = "(unlabeled)"

        announcement = f"  {constraint} constraint exceeded in wellbore '{label}'. Switching constraint."
        print(f"\033[96m {announcement} \033[00m")


def _check_BC_constraints(
    BC_config: list[dict[str, Any]], primary_variables: dict[str, Any], t: float, WB_on: int, verbose: bool
) -> tuple[bool, list[dict[str, Any]]]:
    """Check whether the BC constraints are satisfied, and update the flags accordingly."""
    if not WB_on:
        return True, BC_config

    constraints_satisfied = True
    p = primary_variables["p"][:, 0]
    q = primary_variables["q"][:, 0]

    for group in _wellbore_groups(BC_config):
        hydraulics = group["hydraulics"]
        curr_BC = hydraulics["curr_BC"]
        prev_BC = hydraulics["prev_BC"]
        # Only 1 swap allowed per timestep to prevent oscillations
        if curr_BC != prev_BC:
            continue

        if curr_BC == "flux":
            p_local = p[group["nodes"]]
            p_max = hydraulics["constraints"]["maximum_pressure"]
            p_min = hydraulics["constraints"]["minimum_pressure"]
            # Check max pressure
            if np.any(p_local >= p_max):
                hydraulics["curr_BC"] = "pressure"
                hydraulics["pres_constraint"] = "max"
                hydraulics["type"], hydraulics["function"], hydraulics["values"] = (
                    "pressure",
                    "constant",
                    [p_max],
                )
                constraints_satisfied = False
                _announce_WB_change(group, "Max pressure", verbose)
                break
            # Check min pressure
            if np.any(p_local <= p_min):
                hydraulics["curr_BC"] = "pressure"
                hydraulics["pres_constraint"] = "min"
                hydraulics["type"], hydraulics["function"], hydraulics["values"] = (
                    "pressure",
                    "constant",
                    [p_min],
                )
                constraints_satisfied = False
                _announce_WB_change(group, "Min pressure", verbose)
                break

        if curr_BC == "pressure":
            q_local = q[group["nodes"]]
            # Determine flux constraint
            probe = {
                "function": hydraulics["function_flux"],
                "values": hydraulics["values_flux"],
                "nodes": group["nodes"],
            }
            probe = evaluateBCfunctions(probe, t, np.empty((0, 0)))
            q_goal = probe["BCvalues"]

            if "flux_weight_matrix" in group:
                weight_matrix = group["flux_weight_matrix"]
                q_goal = np.sum(weight_matrix @ np.asanyarray(q_goal))
                q_local = np.sum(weight_matrix @ np.asanyarray(q_local))

            # Check against current fluxes. If exceeded, switch back over.
            if np.any(np.abs(q_local) >= np.abs(q_goal)):
                hydraulics["curr_BC"] = "flux"
                hydraulics["type"], hydraulics["function"], hydraulics["values"] = (
                    "flux",
                    hydraulics["function_flux"],
                    hydraulics["values_flux"],
                )
                constraints_satisfied = False
                _announce_WB_change(group, "Fluid flux", verbose)
                break

    return constraints_satisfied, BC_config


# ------------------------------------ Simulation state
# Wellbore keys changed by constraint switching; all other BC_config entries are static
_WB_STATE_KEYS = (
    "curr_BC",
    "prev_BC",
    "pres_constraint",
    "type",
    "function",
    "values",
    "function_flux",
    "values_flux",
)


@dataclass(frozen=True, slots=True)
class SimState:
    """Everything a time step changes; mesh, material and static BC data are not included."""

    t: float
    n_step: int  # internal step count, gates the solvers' early-step error check
    n_out: int  # output frame index
    dt_hist: list[float]  # [dt_next, dt_n, dt_(n-1), dt_(n-2)]
    primary_variables: dict[str, Any]
    wb_flags: dict[int, dict[str, Any]]  # BC_config index -> wellbore keys
    ext_values: dict[str, dict[str, float]]  # accepted external BC inputs, by label; empty before the first


def _snapshot_wb(BC_config: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Copy the switchable keys of each wellbore group, keyed by BC_config index."""
    flags = {}
    for i, group in enumerate(BC_config):
        hydraulics = group.get("hydraulics", {})
        if "constraints" in hydraulics:
            flags[i] = copy.deepcopy({key: hydraulics[key] for key in _WB_STATE_KEYS})
    return flags


def _load_wb(BC_config: list[dict[str, Any]], flags: dict[int, dict[str, Any]]) -> None:
    """Write a snapshot's wellbore keys into BC_config."""
    for i, group_flags in flags.items():
        BC_config[i]["hydraulics"].update(copy.deepcopy(group_flags))


def snapshot_state(
    t: float,
    n_step: int,
    n_out: int,
    dt_hist: list[float],
    primary_variables: dict[str, Any],
    BC_config: list[dict[str, Any]],
    ext_values: dict[str, dict[str, float]],
) -> SimState:
    """Capture an independent copy of the simulation state."""
    return SimState(
        t=t,
        n_step=n_step,
        n_out=n_out,
        dt_hist=list(dt_hist),
        primary_variables=copy.deepcopy(primary_variables),
        wb_flags=_snapshot_wb(BC_config),
        ext_values=copy.deepcopy(ext_values),
    )


def load_state(state: SimState, BC_config: list[dict[str, Any]]) -> dict[str, Any]:
    """Write a state's wellbore flags into BC_config and return a working copy of its primary variables."""
    _load_wb(BC_config, state.wb_flags)
    return copy.deepcopy(state.primary_variables)


def _rollover(primary_variables: dict[str, Any], BC_config: list[dict[str, Any]], WB_on: int) -> None:
    """Shift the time history after an accepted step and update the wellbore flags."""
    # column 0 gets overwritten in time integration
    primary_variables["p"][:, [3, 2, 1]] = primary_variables["p"][:, [2, 1, 0]]
    primary_variables["T"][:, [3, 2, 1]] = primary_variables["T"][:, [2, 1, 0]]
    if WB_on:
        _update_WB_flags(BC_config)


@dataclass(frozen=True, slots=True)
class SimSetup:
    """Objects that define the run and stay fixed between steps; what a step changes is in SimState."""

    mesh: dict[str, Any]
    material: dict[str, Any]
    controls: dict[str, Any]
    BC_config: list[dict[str, Any]]
    critical_times: npt.NDArray[np.float64]


def advance(  # noqa
    setup: SimSetup,
    state: SimState,
    t_target: float,
    verbose: bool,
    write_steps: bool = False,
) -> SimState:
    """Advance from state to exactly t_target; the returned state precedes the history rollover."""
    controls = setup.controls
    mesh = setup.mesh
    material = setup.material
    BC_config = setup.BC_config
    WB_on = controls["wellbore_check"]
    critical_times = np.union1d(setup.critical_times, [t_target])

    primary_variables = load_state(state, BC_config)
    n_step = state.n_step + 1
    dt = list(state.dt_hist[0:3])
    t, dt = _clamp_to_critical(state.t, dt, critical_times)

    while True:
        # Loop through timesteps
        if verbose:
            print(f"Timestep {n_step}, t = {t:.2e}, {(100 * t / t_target):.2f} % complete")

        while True:
            # Loop until BC constraints satisfied

            # Update Boundary Conditions
            boundary_conditions = parsebcs(BC_config, mesh, primary_variables, t, controls)

            # Perform time integration
            primary_variables, dt, converged, timestep_error = _perform_time_integration(
                controls,
                mesh,
                material,
                primary_variables,
                boundary_conditions,
                BC_config,
                dt,
                n_step,
                verbose,
            )

            # Check BC constraints satisfied
            BC_satisfied, BC_config = _check_BC_constraints(BC_config, primary_variables, t, WB_on, verbose)
            if BC_satisfied:
                break

        # Check error and propose the next timestep
        t_solved = t
        dt_step = list(dt)  # this step's timestep sizes, before the update, for postprocessing T_flow
        t, dt, accepted = _propose_dt(
            controls["solver_type"], t, dt, controls, converged, timestep_error, verbose
        )
        if not accepted:  # retry current timestep
            continue

        if write_steps:
            if verbose:
                print("     Postprocessing...")
            postprocessing(
                n_step, mesh, controls, material, BC_config, primary_variables, t_solved, dt_step, verbose
            )

        if t_solved >= t_target:
            dt_hist = [dt[0], *dt_step]  # dt[0] is the unclamped proposal
            return snapshot_state(
                t_solved, n_step, state.n_out, dt_hist, primary_variables, BC_config, state.ext_values
            )

        _rollover(primary_variables, BC_config, WB_on)
        n_step += 1

        t, dt = _clamp_to_critical(t_solved, dt, critical_times)
        if verbose:
            print(f"    Next timestep: {dt[0]:.2e}")


def initialize_simulation(
    config: dict[str, Any], mesh: dict[str, Any], t_start: float, verbose: bool
) -> tuple[SimSetup, SimState]:
    """Set up the time stepping objects, assign initial conditions and write output frame 0."""
    material = config["material"]
    BC_config = config["boundary_conditions"]
    controls = config["simulation"]

    # ----- Initialize simulation parameters

    ## Numerical controls
    n = 0  # Time step counter
    t = t_start
    # initialize timesteps and store 3 timesteps (necessary for higher order)
    # dt = [dt_n, dt_(n-1), dt_(n-2)]
    dt = [controls["timestep"]] * 3
    # Gather any important time points as specified in the boundary conditions
    critical_times = get_critical_times(BC_config)

    ## Initialize vectors
    primary_variables = {}  # storage structure for all primary variables and results
    p_dofs = np.arange(0, mesh["nn_total"])
    T_dofs = np.arange(0, mesh["nn_total"])
    primary_variables["p_dofs"] = p_dofs
    primary_variables["T_dofs"] = T_dofs
    # Vectors of primary DoFs - ndof x 4
    # [t_n, t_(n-1), t_(n-2), ...]
    primary_variables["p"] = 1e5 * np.ones((len(p_dofs), 4))
    primary_variables["T"] = 300 * np.ones((len(T_dofs), 4))

    ## Wellbore boundary conditions
    WB_on = controls["wellbore_check"]
    if WB_on:
        BC_config = _initialize_WB_checks(BC_config)

    #   note for future use: If a no-flow boundary is desired in the direction of buoyancy,
    #                        there should be an edge integral perform to remove its contribution
    #                        along the upper surfaces.
    if verbose:
        print("     Exporting domain material properties...")
    export_domain_properties(mesh, controls, material, BC_config)

    # --- Initial Conditions
    q = np.zeros((mesh["nn_total"], 1))
    primary_variables["q"] = q

    [primary_variables, material] = _assign_initial_conditions(
        config["initial_conditions"],
        controls,
        mesh,
        material,
        primary_variables,
        BC_config,
        t,
        n,
        verbose,
    )

    # --- Export initial condition
    if verbose:
        print("Export initial condition")
    postprocessing(n, mesh, controls, material, BC_config, primary_variables, t, dt, verbose)

    setup = SimSetup(mesh, material, controls, BC_config, critical_times)
    state = snapshot_state(t, n, n, [dt[0], *dt], primary_variables, BC_config, {})
    return setup, state


# -------------------------------- Main simulation run function -------------
def main(  # noqa
    config_name: str | None = None, config_dir: Path | None = None, verbose: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Main simulation run."""
    if config_name is None:
        parser = argparse.ArgumentParser(description="RAFT - Reservoir Analysis for TANGO")
        parser.add_argument("config", help="Name of the config file (e.g. config_example.yaml) in CONFIG_DIR")
        args = parser.parse_args()
        config_name = args.config

    with RunLogger() as run_logger:
        if verbose:
            print("RAFT - RESERVOIR ANALYSIS FOR TANGO \n")
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print("Beginning simulation " + config_name + "\n")

        # ----- Load config and mesh
        config, mesh = preprocessing(config_name, config_dir, verbose)
        controls = config["simulation"]
        # Copy config name and directory to controls
        controls["config_name"] = config["config_name"]
        controls["config_dir"] = config["config_dir"]

        if controls["output_log"]:
            run_logger.attach_file(prepare_output_directory(controls))

        setup, state = initialize_simulation(config, mesh, controls["t_start"], verbose)
        try:
            state = advance(setup, state, controls["t_end"], verbose, write_steps=True)
        except TimestepError as err:
            print(f"     {err}")
            sys.exit(1)

        if verbose:
            print("Simulation complete. Output files are stored in: \n" + controls["output_directory_full"])

        return state.primary_variables, mesh


if __name__ == "__main__":
    main()
