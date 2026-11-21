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

"""Time integration solvers for subsurface systems."""

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from .history import COHESIVE_HISTORY, VISCOUS_DISSIPATION, VISCOUS_TRACTION
from .matrixassembly import (
    integrate_enthlpy_bndry_residual,
    integrate_enthlpy_bndry_tangent,
    integrate_residuals,
    integrate_tangent_matrices,
    warn_nonmonotone_pressure_diffusion,
)
from .mechanicsdata import _NO_APERTURE, _NO_MECHANICS, MechanicsMatrices
from .physics import Physics
from .postprocessing import postprocessing, prepare_output_directory
from .solidmechanics import fracture_aperture, fracture_face_nodes, fracture_interface_force, internal_force

# Stagnation acceptance: no new minimum residual in this many iterations, within this factor of NR_tol
_NO_HISTORY = np.empty(0)  # history variable not registered
STAGNATION_WINDOW = 10
STAGNATION_FACTOR = 10.0


def rate_coefficients(
    dt: list[float], order: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Rate of change (3,), acceleration (4,) and extrapolation (3,) coefficients over the time levels."""
    # dt_nm0 = t_n - t_(n-1)
    # dt_nm1 = t_(n-1) - t_(n-2)
    # dt_nm2 = t_(n-2) - t_(n-3)
    dt_nm0 = dt[0]
    dt_nm1 = dt[1]
    dt_nm2 = dt[2]
    cv = np.ones(3)  # rate of change coefficients
    ca = np.zeros(4)  # acceleration coefficients
    cx = np.ones(3)  # extrapolation coefficients
    if order > 1:
        cv[0] = (2*dt_nm0 + dt_nm1) / dt_nm0 / (dt_nm0 + dt_nm1)
        cv[1] = -(dt_nm0 + dt_nm1) / dt_nm0 / dt_nm1
        cv[2] = dt_nm0 / dt_nm1 / (dt_nm0 + dt_nm1)

        ca[0] = 2*(3*dt_nm0 + 2*dt_nm1 + dt_nm2) / dt_nm0 / (dt_nm0 + dt_nm1) / (dt_nm0 + dt_nm1 + dt_nm2)
        ca[1] = -2*(2*dt_nm0 + 2*dt_nm1 + dt_nm2) / dt_nm0 / dt_nm1 / (dt_nm1 + dt_nm2)
        ca[2] = 2*(2*dt_nm0 + dt_nm1 + dt_nm2) / (dt_nm0 + dt_nm1) / dt_nm1 / dt_nm2
        ca[3] = -2*(2*dt_nm0 + dt_nm1) / (dt_nm0 + dt_nm1 + dt_nm2) / (dt_nm1 + dt_nm2) / dt_nm2

        cx[0] = (dt_nm0 + dt_nm1 + dt_nm2) * (dt_nm0 + dt_nm1) / (dt_nm1 + dt_nm2) / dt_nm1
        cx[1] = -dt_nm0 * (dt_nm0 + dt_nm1 + dt_nm2) / dt_nm1 / dt_nm2
        cx[2] = dt_nm0 * (dt_nm0 + dt_nm1) / (dt_nm1 + dt_nm2) / dt_nm2
    else:
        cv[0] = 1 / dt_nm0
        cv[1] = -1 / dt_nm0
        cv[2] = 0

        ca[0] = 2 / dt_nm0 / (dt_nm0 + dt_nm1)
        ca[1] = -2 / dt_nm0 / dt_nm1
        ca[2] = 2 / dt_nm1 / (dt_nm0 + dt_nm1)

        cx[0] = 1 + dt_nm0 / dt_nm1
        cx[1] = -dt_nm0 / dt_nm1
        cx[2] = 0
    return cv, ca, cx


def _check_convergence_extrap(
    T: npt.NDArray[np.float64],
    T_extrap: npt.NDArray[np.float64],
    controls: dict[str, Any],
    n: int,
) -> tuple[bool, float]:
    """Check convergence for the linear extrapolation."""
    rate_order = controls["rate_order"]
    rel_tol_T = controls["rel_error_T"]

    converged = True  # default to converged, and only break if not converged

    # First few timesteps affected by initial conditions, do not check
    if n < rate_order + 1:
        extrapolation_error = 0.0
        return converged, extrapolation_error

    # Calculate error
    extrapolation_error = max(abs(T[:, 0] - T_extrap) / T[:, 0])

    if extrapolation_error > rel_tol_T:
        converged = False

    return converged, extrapolation_error


def _apply_boundary_conditions(
    boundary_conditions: dict[str, Any],
    primary_variables: dict[str, Any],
    mesh: dict[str, Any],
    physics: Physics,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Write Dirichlet values; return fixed DoFs, Dirichlet DoFs and Neumann forces per field."""
    # DoFs held at their current values because their field is off in that domain
    off = {
        "u": np.empty(0, dtype=int),
        "p": physics.off_nodes(mesh, "flow"),
        "T": physics.off_nodes(mesh, "heat"),
    }
    fixed_dofs, dirichlet_dofs, F_ext = {}, {}, {}
    for key in ("u", "p", "T"):
        # Dirichlet DoFs and values
        fix = boundary_conditions[key + "_dirichlet"][:, 0].astype(int)
        primary_variables[key][fix, 0] = boundary_conditions[key + "_dirichlet"][:, 1]
        fixed_dofs[key] = np.union1d(fix, off[key])
        # Dirichlet DoFs with an assembled residual, for reaction fluxes and convergence scaling
        dirichlet_dofs[key] = np.setdiff1d(fix, off[key])
        # Neumann forces
        F_ext[key] = np.zeros((len(primary_variables[key + "_dofs"]), 1))
        F_ext[key][boundary_conditions[key + "_neumann"][:, 0].astype(int)] = boundary_conditions[
            key + "_neumann"
        ][:, 1:2]
    return fixed_dofs, dirichlet_dofs, F_ext


def compute_residuals(
    F_vec: dict[str, Any], F_ext: dict[str, Any], keys: list[str]
) -> dict[str, npt.NDArray[np.float64]]:
    """Assemble integrated residuals alongside external force vectors for the solved `keys`."""
    R = {}
    if "u" in keys:
        R["u"] = F_vec["F_u_internal"] - F_ext["u"]

    if "p" in keys:
        R["p"] = F_vec["F_Pstorage"] - F_vec["F_Pdiffusion"] - F_ext["p"]

    if "T" in keys:
        F_enth_bnd = F_vec.get("F_enth_bnd", 0)
        R["T"] = (
            F_vec["F_Tstorage"]
            + F_vec["F_Tdiffusion"]
            - F_vec["F_Tadvection"]
            + F_vec["F_Tstab"]
            - F_ext["T"]
            - F_enth_bnd
        )

    return R


def _partition_and_solve_matrix(
    A: sp.csc_matrix[Any],
    b: npt.NDArray[np.float64],
    dofs_all: npt.NDArray[np.int_],
    nE: npt.NDArray[np.int_],  # essential DoF indices
) -> npt.NDArray[np.float64]:
    """Partitions and solves a matrix given a set of essential DoFs. Returns full vector of DoFs."""
    nF = np.setdiff1d(dofs_all, nE)  # free DoFs

    KFF = A[np.ix_(nF, nF)]  # type: ignore[index]
    # KFE = A[np.ix_(nF, nE)]  # type: ignore[index]

    dF = spsolve(KFF, (b[nF]))

    d = np.zeros((len(dofs_all), 1))
    d[nF, 0] = dF.ravel()

    return d.ravel()


def _check_convergence_NR(
    R: dict[str, Any],
    fixed_dofs: dict[str, Any],
    dirichlet_dofs: dict[str, Any],
    NR: dict[str, Any],
    mesh: dict[str, Any],
    iteration: int,
    NR_tol: float,
) -> tuple[bool, float]:
    """Check residual norms against convergence tolerance.

    Normalize each key's free-dof residual norm against the norm of that same key's
    reaction flux at its own Dirichlet dofs - a physical flux scale for the system (how
    much is actually flowing through the domain), rather than a raw per-element residual
    that can't tell a genuinely small residual from a tiny fraction of a much larger one.
    """
    nn = mesh["nn_total"]

    for key in R:
        n_DoFs = np.arange(0, len(R[key]))  # this key's DoFs
        nF = np.setdiff1d(n_DoFs, fixed_dofs[key]).astype(int)  # free DoFs

        numerator = np.linalg.norm(R[key][nF])
        flux_scale = max(float(np.linalg.norm(R[key][dirichlet_dofs[key]])), 1.0)

        NR[key][iteration] = numerator / flux_scale / nn
        NR["all"][iteration] += NR[key][iteration] ** 2

    NR["all"][iteration] = NR["all"][iteration] ** 0.5

    converged = bool(NR["all"][iteration] < NR_tol)

    return converged, NR["all"][iteration]


def _stagnated(NR_all: npt.NDArray[np.float64], iteration: int, NR_tol: float) -> bool:
    """Residual has reached no new minimum over the last STAGNATION_WINDOW iterations and is near NR_tol.

    Accepts a solution whose residual is held above NR_tol by round-off rather than by the physics.
    """
    if iteration < STAGNATION_WINDOW:
        return False
    best_before = NR_all[: iteration + 1 - STAGNATION_WINDOW].min()
    best_recent = NR_all[iteration + 1 - STAGNATION_WINDOW : iteration + 1].min()
    return bool(best_recent >= best_before and NR_all[iteration] < STAGNATION_FACTOR * NR_tol)


def _plot_nr_residuals(
    NR: dict[str, Any],
    aitken_state: dict[str, Any],
    iteration: int,
    iter_max: int,
    residual_keys: list[str],
) -> None:
    """Plot Newton-Raphson residual history and Aitken damping coefficients."""
    labels = {"T": "Temperature", "p": "Pressure", "u": "Displacement"}
    styles = {"T": "r-", "p": "b-", "u": "g-"}

    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()

    for key in residual_keys:
        ax1.semilogy(np.arange(1, iteration + 1), NR[key][range(0, iteration)], styles[key])
    ax1.semilogy(np.arange(1, iteration + 1), NR["all"][range(0, iteration)], "ko-")
    plt.ylabel("residual")
    plt.legend([labels[key] for key in residual_keys] + ["Combined"])

    ax2.plot(
        np.arange(1, iteration + 1),
        np.array(aitken_state["aitken_coefficient"])[range(0, iteration)],
        "go-",
    )
    plt.ylabel("damping coefficient")

    plt.xlim([0, iter_max])
    plt.xlabel("iteration")


def _aitken_relax(
    iteration: int,
    sol_mp1: npt.NDArray[np.float64],
    sol_m: npt.NDArray[np.float64],
    idx_sol: npt.NDArray[np.int64] | list[int],
    res_m: npt.NDArray[np.float64],
    delta_m: float | None,
    bounds: tuple[float, float, float],
    max_init_relax: float = 1.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, npt.NDArray[np.float64]]:
    """Apply Aitken delta^2 method with a limited range of relaxation parameters.

    Aitken relaxation is a cheap relaxation method that uses information from the previous two
    iterations to improve the current guess.

    sol_mp1 : np.ndarray
        Current solution vector (at iteration k+1).
    sol_m : np.ndarray
        Previous solution vector (at iteration k).
    idx_sol : np.ndarray or list
        Indices/DoFs on which the relaxation parameters are evaluated.
    res_m : np.ndarray or None
        Difference vector from the previous iteration.
    delta_m : float or None
        Damping coefficient from the previous iteration.
    bounds : tuple
        Bounds (min_relax, max_relax, abs_min) on the relaxation parameter.
        min_relax: signed minimum
        max_relax: signed maxxmum
        abs_min: absolute minimum
    max_init_relax : float
        Control parameter flag (NaN: no relaxation, <1e-8: fixed factor, >=1e-8: Aitken).
    """
    min_relax, max_relax, abs_min = bounds

    # Residual of current iterate (defined as the difference between successive iterates)
    res_mp1 = sol_mp1 - sol_m

    if np.isnan(max_init_relax):
        delta = 1.0
    elif max_init_relax < 1e-8:
        delta = abs(max_init_relax)
    else:
        if iteration == 0:
            if delta_m is None or np.isnan(delta_m):
                aitken_para = max_init_relax
            else:
                aitken_para = min(max_init_relax, delta_m)
            delta = aitken_para
        else:
            # Safely extract specified degrees of freedom
            res_m_idx = res_m[idx_sol]
            res_mp1_idx = res_mp1[idx_sol]

            diff_res = res_mp1_idx - res_m_idx
            norm_diff = np.linalg.norm(diff_res, 2)

            if norm_diff == 0:
                aitken_para = delta_m if delta_m is not None else 1.0
            else:
                numerator = -np.dot(res_m_idx.T, diff_res)
                aitken_para = delta_m * (numerator / (norm_diff**2))

            del_sign = np.sign(aitken_para)
            if del_sign == 0:
                del_sign = 1.0
                
            delta = del_sign * max(abs_min, abs(delta))
            delta = max(min(max_relax, delta), min_relax)

    if np.isnan(delta):
        delta = 1.0

    # Apply relaxation to damp the current guess vector
    sol_mp1_damped = (1.0 - delta) * sol_m + delta * sol_mp1

    # Return elements matching the recursive structure needed for next iterations
    return sol_mp1_damped, res_mp1, delta, sol_mp1_damped


def _field_row_scale(A: sp.csc_matrix[Any], key_dofs: dict[str, Any]) -> npt.NDArray[np.float64]:
    """Row scale per field, 1 / mean |diag| of the field's diagonal block."""
    diag = np.abs(A.diagonal())
    scale = np.ones(A.shape[0])
    for dofs in key_dofs.values():
        d = diag[dofs]
        d = d[d > 0.0]
        if d.size:
            scale[dofs] = 1.0 / d.mean()
    return scale


def _relax_dofs(
    key: str, local_dofs: npt.NDArray[np.int_], mesh: dict[str, Any], mechanics: MechanicsMatrices
) -> npt.NDArray[np.int_]:
    """DoFs of `key` that set the Aitken factor: all of them for p and T; for u, the fracture face nodes
    where contact occurs (none without fractures, since the force balance is otherwise linear).
    """
    if key != "u":
        return local_dofs
    if not mechanics.interface.weights.size:
        return np.empty(0, dtype=int)
    nsd = mesh["nsd"]
    face_nodes = np.unique(fracture_face_nodes(mesh))
    dofs: npt.NDArray[np.int_] = (nsd * face_nodes[:, None] + np.arange(nsd)).ravel()
    return dofs


# Placeholder when mechanics is off

# Primary-variable keys and the physics field that solves them, in block order
_FIELD_KEYS = (("u", "mechanics"), ("p", "flow"), ("T", "heat"))


def _reaction_flow_rates(
    F_vec: dict[str, Any], F_ext: dict[str, Any], fixed_dofs: dict[str, Any], dirichlet_dofs: dict[str, Any]
) -> npt.NDArray[np.float64]:
    """Nodal flow rates: reactions at Dirichlet dofs, prescribed fluxes at free dofs."""
    R_p = compute_residuals(F_vec, F_ext, ["p"])["p"]
    q = np.zeros((len(R_p), 1))
    q[dirichlet_dofs["p"]] = R_p[dirichlet_dofs["p"]]
    nF_p = np.setdiff1d(np.arange(len(R_p)), fixed_dofs["p"])
    q[nF_p] = q[nF_p] + F_ext["p"][nF_p]
    return q


def _export_iteration(
    iteration: int,
    controls: dict[str, Any],
    material: dict[str, Any],
    mesh: dict[str, Any],
    BC_config: list[dict[str, Any]],
    primary_variables: dict[str, Any],
    mechanics: MechanicsMatrices,
) -> None:
    """Write the current Newton iterate as frame `iteration` in <output>/unconverged."""
    io_path = prepare_output_directory(controls)
    debug_controls = {
        **controls,
        "output_directory": io_path,
        "config_name": "unconverged",
        "output_frequency": 1,
        "output_vtk": 1,
    }
    postprocessing(
        iteration,
        mesh,
        debug_controls,
        material,
        BC_config,
        primary_variables,
        float(iteration),
        False,
        mechanics,
    )


def _newton_raphson_loop(  # noqa
    controls: dict[str, Any],
    material: dict[str, Any],
    mesh: dict[str, Any],
    primary_variables: dict[str, Any],
    BC_config: list[dict[str, Any]],
    T_flow: npt.NDArray[np.float64],
    cv: npt.NDArray[np.float64],  # rate coefficients
    ca: npt.NDArray[np.float64],  # acceleration coefficients
    fixed_dofs: dict[str, Any],
    dirichlet_dofs: dict[str, Any],
    F_ext: dict[str, Any],
    verbose: bool,
    debug_plot_on: bool,
    export_unconverged: bool,
    *,
    physics: Physics,
    T_flow_lagged: bool,
    dt_n: float,
    mechanics: MechanicsMatrices = _NO_MECHANICS,
) -> tuple[dict[str, Any], bool, float]:
    """Perform non-linear Newton-Raphson iterations for the fields `physics` solves.

    `T_flow` is a separate frozen/extrapolated copy when `T_flow_lagged`; otherwise it must
    be the SAME array object as `primary_variables["T"]` so it tracks the in-loop updates below.
    """
    keys = [key for key, field in _FIELD_KEYS if physics.solves(field)]
    solve_u, solve_p, solve_T = "u" in keys, "p" in keys, "T" in keys
    # Newton-Raphson controls
    iter_max = controls["iterative_controls"]["iter_max"]
    NR_tol = controls["iterative_controls"]["NR_tol"]
    NR_step = controls["iterative_controls"]["NR_step"]
    aitken_on = controls["iterative_controls"]["aitken_relaxation"]
    aitken_bounds = tuple(controls["iterative_controls"]["aitken_bounds"])
    aitken_start = controls["iterative_controls"]["aitken_start"]
    one_pass = controls["iterative_controls"]["one_pass"] and T_flow_lagged
    converged = False

    if one_pass:
        # Overwrite bad input when doing a one-pass solve
        aitken_on = False
        NR_step = 1.0

    # Initialize
    iteration = 0
    NR = {"all": np.zeros(iter_max + 1)}  # Store iteration residuals for debugging
    R = {}
    aitken_state: dict[str, Any] = {}
    fixed_dofs_global = {}
    vars_orig = {}
    key_dofs = {}  # key -> that key's global indices (in d_n/d_new) for its dofs
    nn_all = 0  # Running total of dof offset
    d_n = np.empty((0, 1))
    nE: npt.NDArray[np.int_] = np.empty((0, 1), dtype=int)
    aitken_state["variables_km1"] = np.empty((0, 1))
    aitken_state["relax_dofs"] = np.empty((0,), dtype=int)
    for key in keys:
        # Initialize residuals
        NR[key] = np.zeros(iter_max + 1)
        d_n = np.vstack((d_n, primary_variables[key][:, 0:1]))
        vars_orig[key] = primary_variables[key].copy()

        local_dofs = primary_variables[key + "_dofs"]
        key_dofs[key] = nn_all + np.arange(len(local_dofs))
        fixed_dofs_global[key] = fixed_dofs[key] + nn_all
        nE = np.vstack((nE, fixed_dofs_global[key].reshape(-1, 1)))
        nn_all += len(local_dofs)

        # Initialize relaxation
        free_dofs_key = np.setdiff1d(_relax_dofs(key, local_dofs, mesh, mechanics), fixed_dofs[key])
        free_dofs_key = free_dofs_key.astype(int) + key_dofs[key][0]
        aitken_state["relax_dofs"] = np.concatenate((aitken_state["relax_dofs"], free_dofs_key))

    nE = nE.ravel()
    aitken_state["aitken_coefficient"] = np.ones((iter_max + 2, 1))
    aitken_state["delta_variables_km1"] = np.zeros((nn_all, 1))

    # Run-type labels
    if set(keys) == {"u", "p"}:
        label = "MONOLITHIC HYDRO-MECHANIC"
    elif len(keys) > 1:
        label = "MONOLITHIC"
    elif solve_u:
        label = "ITERATIVE DISPLACEMENT"
    elif solve_p:
        label = "ITERATIVE PRESSURE"
    else:
        label = "ITERATIVE TEMPERATURE"

    if verbose:
        print(f"      {label} NEWTON RAPHSON: ITERATION   ", end="")
    while True:
        if verbose:
            print(f"\r     {label} NEWTON RAPHSON: ITERATION {iteration + 1}   ", end="", flush=True)

        # Fracture aperture from the contact state of every displacement time level
        aperture = (
            fracture_aperture(mechanics.interface, primary_variables["u"])
            if mechanics.interface.weights.size
            else _NO_APERTURE
        )

        # Integrate residual vectors
        F_vec: dict[str, Any] = {}
        if solve_p or solve_T:
            F_vec = integrate_residuals(
                mesh,
                material,
                primary_variables,
                controls,
                cv,
                T_flow,
                physics=physics,
                T_flow_lagged=T_flow_lagged,
                mechanics=mechanics,
                aperture=aperture,
            )
        # Force balance from the stored linear matrices
        if solve_u:
            F_vec["F_u_internal"] = internal_force(mechanics, primary_variables, material, mesh["nn"], ca)
            K_uu = mechanics.K_uu
            if mechanics.M_uu.shape[0]:
                K_uu = sp.csc_matrix(K_uu + ca[0] * mechanics.M_uu)
            if mechanics.interface.weights.size:
                # Fracture contact and cohesion, the only nonlinear terms of the force balance
                history = primary_variables["history"]
                delta_max_c = history.committed.get(COHESIVE_HISTORY, _NO_HISTORY)
                u_levels = primary_variables["u"]
                u_rate = u_levels[:, 0:3] @ cv  # (nsd*nn, 3) @ (3,)
                face_force, K_face, delta, t_v = fracture_interface_force(
                    mechanics.interface, u_levels[:, 0], delta_max_c, u_rate, cv[0], dt_n
                )
                if delta.size:
                    history.trial[COHESIVE_HISTORY] = np.maximum(delta_max_c, delta)
                if t_v.size:
                    # dissipation t_v . ([[u]]^n - [[u]]^(n-1)) added to the committed total
                    u_step = u_levels[:, 0] - u_levels[:, 1]
                    jump_step = (mechanics.interface.jump @ u_step).reshape(t_v.shape)
                    dissipated = np.sum(t_v * jump_step, axis=1)
                    history.trial[VISCOUS_DISSIPATION] = history.committed[VISCOUS_DISSIPATION] + dissipated
                    history.trial[VISCOUS_TRACTION] = t_v
                F_vec["F_u_internal"] = F_vec["F_u_internal"] + face_force
                K_uu = sp.csc_matrix(K_uu + K_face)

        # Update reaction flow rates in-loop when coupled, for the enthalpy boundary flux
        if solve_p and solve_T:
            primary_variables["q"] = _reaction_flow_rates(F_vec, F_ext, fixed_dofs, dirichlet_dofs)

        # Calculate enthalpy boundary fluxes using reaction flow rates
        if solve_T:
            F_enth_bnd = integrate_enthlpy_bndry_residual(
                mesh, material, primary_variables, controls, T_flow, BC_config, T_flow_lagged=T_flow_lagged
            )
            F_vec["F_enth_bnd"] = F_enth_bnd["F_enth_bnd"]

        # Compuate residuals
        R = compute_residuals(F_vec, F_ext, keys)

        # Debug output of the fields whose residual is checked this iteration
        if export_unconverged:
            _export_iteration(iteration, controls, material, mesh, BC_config, primary_variables, mechanics)

        # Check convergence
        converged, iteration_error = _check_convergence_NR(
            R, fixed_dofs, dirichlet_dofs, NR, mesh, iteration, NR_tol
        )

        if converged and iteration > 0:
            if verbose:
                print(" - CONVERGED")
            break
        elif _stagnated(NR["all"], iteration, NR_tol):
            converged = True
            if verbose:
                print(f" - CONVERGED (stagnated at {iteration_error:.2e} > NR_tol)")
            break
        else:  # Update iteration
            # Integrate Jacobian
            jacobian: dict[str, Any] = {}
            if solve_p or solve_T:
                jacobian = integrate_tangent_matrices(
                    mesh,
                    material,
                    primary_variables,
                    controls,
                    cv,
                    T_flow,
                    physics=physics,
                    T_flow_lagged=T_flow_lagged,
                    mechanics=mechanics,
                    aperture=aperture,
                )

            # Warnings for poor conditioning
            if iteration == 0 and solve_p:
                warn_nonmonotone_pressure_diffusion(jacobian["Kpp"])

            # Assemble Jacobian - different by case
            if solve_u and not (solve_p or solve_T):
                # Mechanics only
                A = K_uu
                b = R["u"]

            elif solve_u and solve_p and solve_T:
                # Monolithic displacement, pressure and temperature
                K_enth_bnd = integrate_enthlpy_bndry_tangent(
                    mesh,
                    material,
                    primary_variables,
                    controls,
                    T_flow,
                    BC_config,
                    monolithic=True,
                    Kpp=jacobian["Kpp"],
                    KpT=jacobian["KpT"],
                    Kpu=jacobian["Kpu"],
                )
                A = cast(
                    sp.csc_matrix[Any],
                    sp.bmat(
                        [
                            [K_uu, mechanics.K_up, mechanics.K_uT],
                            [jacobian["Kpu"], jacobian["Kpp"], jacobian["KpT"]],
                            [
                                jacobian["KTu"] - K_enth_bnd["K_enth_bnd_Tu"],
                                jacobian["KTp"] - K_enth_bnd["K_enth_bnd_Tp"],
                                jacobian["KTT"] - K_enth_bnd["K_enth_bnd"],
                            ],
                        ],
                        format="csc",
                    ),
                )

                b = np.vstack((R["u"], R["p"], R["T"]))

            elif solve_u and solve_T:
                # Monolithic displacement and temperature
                K_enth_bnd = integrate_enthlpy_bndry_tangent(
                    mesh,
                    material,
                    primary_variables,
                    controls,
                    T_flow,
                    BC_config,
                    T_flow_lagged=T_flow_lagged,
                )
                A = cast(
                    sp.csc_matrix[Any],
                    sp.bmat(
                        [
                            [K_uu, mechanics.K_uT],
                            [jacobian["KTu"], jacobian["KTT"] - K_enth_bnd["K_enth_bnd"]],
                        ],
                        format="csc",
                    ),
                )

                b = np.vstack((R["u"], R["T"]))

            elif solve_u and solve_p:
                # Monolithic displacement and pressure
                A = cast(
                    sp.csc_matrix[Any],
                    sp.bmat(
                        [
                            [K_uu, mechanics.K_up],
                            [jacobian["Kpu"], jacobian["Kpp"]],
                        ],
                        format="csc",
                    ),
                )

                b = np.vstack((R["u"], R["p"]))

            elif solve_p and solve_T:
                # Monolithic
                K_enth_bnd = integrate_enthlpy_bndry_tangent(
                    mesh,
                    material,
                    primary_variables,
                    controls,
                    T_flow,
                    BC_config,
                    monolithic=True,
                    Kpp=jacobian["Kpp"],
                    KpT=jacobian["KpT"],
                )

                # Total combined matrix
                A = cast(
                    sp.csc_matrix[Any],
                    sp.bmat(
                        [
                            [jacobian["Kpp"], jacobian["KpT"]],
                            [
                                jacobian["KTp"] - K_enth_bnd["K_enth_bnd_Tp"],
                                jacobian["KTT"] - K_enth_bnd["K_enth_bnd"],
                            ],
                        ],
                        format="csc",
                    ),
                )

                b = np.vstack((R["p"], R["T"]))

            elif solve_p:
                # Pressure only
                A = jacobian["Kpp"]
                b = R["p"]

            else:
                # Temperature only
                K_enth_bnd = integrate_enthlpy_bndry_tangent(
                    mesh,
                    material,
                    primary_variables,
                    controls,
                    T_flow,
                    BC_config,
                    T_flow_lagged=T_flow_lagged,
                )
                A = jacobian["KTT"] - K_enth_bnd["K_enth_bnd"]
                b = R["T"]

            # Solve system of equations
            # Scale each field's rows to a common magnitude before the solve
            row_scale = _field_row_scale(A, key_dofs)
            A = sp.csc_matrix(sp.diags(row_scale) @ A)
            b = row_scale[:, None] * np.asarray(b)
            delta_var = _partition_and_solve_matrix(A, -b, np.arange(0, nn_all), nE)

            d_n = d_n + NR_step * delta_var.reshape(-1, 1)

            # Check for nan or inf data and break
            if np.any(np.isnan(d_n)) or np.any(np.isinf(d_n)):
                if verbose:
                    print(" - Nan or Inf detected, CONVERGENCE FAILED. Resetting timestep.")
                break

            # Apply relaxation
            if iteration == aitken_start:
                for key in keys:
                    aitken_state["variables_km1"] = np.vstack(
                        (aitken_state["variables_km1"], primary_variables[key][:, 0:1])
                    )
            if aitken_on and iteration >= aitken_start:
                (
                    d_new,
                    aitken_state["delta_variables_km1"],
                    aitken_state["aitken_coefficient"][iteration + 1],
                    aitken_state["variables_km1"],
                ) = _aitken_relax(
                    iteration - aitken_start,
                    d_n,
                    aitken_state["variables_km1"],
                    aitken_state["relax_dofs"],
                    aitken_state["delta_variables_km1"],
                    aitken_state["aitken_coefficient"][iteration],
                    aitken_bounds,
                )
            else:
                d_new = d_n

            d_n = d_new
            # Write the updated global vector back into each key's primary variable.
            # Uses in-place slice assignment (not a new array) so that a `T_flow`
            # passed in as the SAME object as primary_variables["T"] (monolithic
            # case) stays aliased and reflects this update automatically.
            for key in keys:
                primary_variables[key][:, 0:1] = d_n[key_dofs[key]]

            iteration += 1

        if one_pass:
            converged = True
            if verbose:
                print(" - CONVERGED IN ONE PASS")
            break

        if iteration >= iter_max:
            if verbose:
                print(" - CONVERGENCE FAILED")
            break

    # Revert the keys if not converged
    if not converged:
        for key in keys:
            primary_variables[key] = vars_orig[key]
    # Caclulate reaction fluxes once when pressure is solved alone; coupled is calculated in-loop.
    if solve_p and not solve_T:
        primary_variables["q"] = _reaction_flow_rates(F_vec, F_ext, fixed_dofs, dirichlet_dofs)

    if debug_plot_on or iteration >= iter_max or not converged:
        _plot_nr_residuals(NR, aitken_state, iteration, iter_max, keys)

    return primary_variables, converged, iteration_error


# ------------------------ Primary time integration function with operator splitting ----------------
def operator_split_time_integration(  # noqa
    controls: dict[str, Any],
    mesh: dict[str, Any],
    material: dict[str, Any],
    primary_variables: dict[str, Any],
    boundary_conditions: dict[str, Any],
    BC_config: list[dict[str, Any]],
    dt: list[float],
    n: int,
    verbose: bool,
    *,
    mechanics: MechanicsMatrices = _NO_MECHANICS,
) -> tuple[dict[str, Any], list[float], bool, float]:
    """Assemble FEM matrices and vectors and solve for solution at next timestep.

    Solution uses operator splitting with an extrapolated, lagged temperature. Error
    is calculated between the predicted and calculated temperatures for convergence.
    """
    debug_pressure_plot_on = False
    debug_temperature_plot_on = False
    export_unconverged = False  # write the fields every Newton iteration to <output>/unconverged
    # extract vectors of nodal dofs
    T = primary_variables["T"]
    u_orig = primary_variables["u"].copy()
    p_orig = primary_variables["p"].copy()
    T_orig = primary_variables["T"].copy()
    rate_order = controls["rate_order"]
    physics = controls["physics"]

    # calculate rate coefficients
    cv, ca, cx = rate_coefficients(dt, rate_order)

    # Apply fixed boundary conditions; off nodes hold their values
    fixed_dofs, dirichlet_dofs, F_ext = _apply_boundary_conditions(
        boundary_conditions, primary_variables, mesh, physics
    )

    # Extrapolate temperature for free DoFs
    T_extrap = cx[0] * T[:, 1] + cx[1] * T[:, 2] + cx[2] * T[:, 3]
    T_fix = fixed_dofs["T"].astype(int)
    T_extrap[T_fix] = T[T_fix, 0]
    T[:, 0] = T_extrap

    # Frozen snapshot for flow-property evaluations: column 0 is the extrapolated/fixed
    # guess (kept constant through the temperature Newton loop, unlike T itself), columns
    # 1-3 are the real time history, needed as-is by the mass-storage rate terms.
    T_flow = T.copy()
    primary_variables["T_flow"] = T_flow[:, 0].copy()  # for output

    # Update displacement and pressure with extrapolated temperature
    if physics.solves("flow") or physics.solves("mechanics"):
        primary_variables, p_converged, _ = _newton_raphson_loop(
            controls,
            material,
            mesh,
            primary_variables,
            BC_config,
            T_flow,  # Extrapolated tempearture
            cv,
            ca,
            fixed_dofs,
            dirichlet_dofs,
            F_ext,
            verbose,
            debug_pressure_plot_on,
            export_unconverged,
            physics=physics.with_off("heat"),
            T_flow_lagged=False,
            dt_n=dt[0],
            mechanics=mechanics,
        )
    else:
        p_converged = True  # pressure is off everywhere: nothing to solve

    # Update temperature only if pressure converged
    if not p_converged:
        T_converged = False  # temperature not attempted
    elif physics.solves("heat"):
        # Update temperature. Use extrapolated temperature for property evaluation,
        # for consistency with pressure.
        primary_variables, T_converged, _ = _newton_raphson_loop(
            controls,
            material,
            mesh,
            primary_variables,
            BC_config,
            T_flow,
            cv,
            ca,
            fixed_dofs,
            dirichlet_dofs,
            F_ext,
            verbose,
            debug_temperature_plot_on,
            export_unconverged,
            physics=physics.with_off("flow").with_off("mechanics"),
            T_flow_lagged=True,
            dt_n=dt[0],
            mechanics=mechanics,
        )
    else:
        T_converged = True  # temperature is off everywhere: nothing to solve

    # ---- Check for extrapolation error
    if p_converged and T_converged:
        # Check for extrapolation timestep error
        converged, timestep_error = _check_convergence_extrap(primary_variables["T"], T_extrap, controls, n)
    else:
        primary_variables["u"] = u_orig.copy()
        primary_variables["p"] = p_orig.copy()
        primary_variables["T"] = T_orig.copy()
        converged = False
        timestep_error = -1  # flag for Newton-Raphson failure

    return primary_variables, dt, converged, timestep_error


# --------------- Monolithic time integration with Newton-Raphson iterations.
def monolithic_time_integration(  # noqa
    controls: dict[str, Any],
    mesh: dict[str, Any],
    material: dict[str, Any],
    primary_variables: dict[str, Any],
    boundary_conditions: dict[str, Any],
    BC_config: list[dict[str, Any]],
    dt: list[float],
    n: int,  # noqa
    verbose: bool,
    *,
    mechanics: MechanicsMatrices = _NO_MECHANICS,
) -> tuple[dict[str, Any], list[float], bool, float]:
    """Assemble FEM matrices and vectors and solve for solution at next timestep.

    This monolithic iterative scheme uses Newton-Raphson to solve for temperature
    and pressure monolithically.
    """
    debug_plot_on = True
    export_unconverged = False  # write the fields every Newton iteration to <output>/unconverged
    rate_order = controls["rate_order"]
    physics = controls["physics"]

    # calculate rate coefficients
    cv, ca, cx = rate_coefficients(dt, rate_order)

    # Apply fixed boundary conditions; off nodes hold their values
    fixed_dofs, dirichlet_dofs, F_ext = _apply_boundary_conditions(
        boundary_conditions, primary_variables, mesh, physics
    )
    if not (physics.solves("flow") or physics.solves("heat") or physics.solves("mechanics")):
        return primary_variables, dt, True, 0.0

    primary_variables, converged, iteration_error = _newton_raphson_loop(
        controls,
        material,
        mesh,
        primary_variables,
        BC_config,
        primary_variables["T"],  # T_flow must alias primary_variables["T"] for monolithic
        cv,
        ca,
        fixed_dofs,
        dirichlet_dofs,
        F_ext,
        verbose,
        debug_plot_on,
        export_unconverged,
        physics=physics,
        T_flow_lagged=False,
        dt_n=dt[0],
        mechanics=mechanics,
    )

    return primary_variables, dt, converged, iteration_error
