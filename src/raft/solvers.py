"""Time integration solvers for subsurface systems."""

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from .matrixassembly import (
    integrate_enthlpy_bndry_residual,
    integrate_enthlpy_bndry_tangent,
    integrate_residuals,
    integrate_tangent_matrices,
    warn_nonmonotone_pressure_diffusion,
)


def rate_coefficients(dt: list[float], order: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate the rate of change coefficients."""
    # dt_nm0 = t_n - t_(n-1)
    # dt_nm1 = t_(n-1) - t_(n-2)
    # dt_nm2 = t_(n-2) - t_(n-3)
    dt_nm0 = dt[0]
    dt_nm1 = dt[1]
    dt_nm2 = dt[2]
    cv = np.ones(3)  # rate of change coefficients
    cx = np.ones(3)  # extrapolation coefficients
    if order > 1:
        cv[0] = (2 * dt_nm0 + dt_nm1) / dt_nm0 / (dt_nm0 + dt_nm1)
        cv[1] = -(dt_nm0 + dt_nm1) / dt_nm0 / dt_nm1
        cv[2] = dt_nm0 / dt_nm1 / (dt_nm0 + dt_nm1)

        cx[0] = (dt_nm0 + dt_nm1 + dt_nm2) * (dt_nm0 + dt_nm1) / (dt_nm1 + dt_nm2) / dt_nm1
        cx[1] = -dt_nm0 * (dt_nm0 + dt_nm1 + dt_nm2) / dt_nm1 / dt_nm2
        cx[2] = dt_nm0 * (dt_nm0 + dt_nm1) / (dt_nm1 + dt_nm2) / dt_nm2
    else:
        cv[0] = 1 / dt_nm0
        cv[1] = -1 / dt_nm0
        cv[2] = 0

        cx[0] = 1 + dt_nm0 / dt_nm1
        cx[1] = -dt_nm0 / dt_nm1
        cx[2] = 0
    return cv, cx


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


def _unpack_boundary_conditions(
    boundary_conditions: dict[str, Any],
    p_dofs: npt.NDArray[np.float64],
    T_dofs: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Parse Dirichlet and Neumann BCs into fixed dofs/values and force vectors."""
    T_fix = boundary_conditions["T_dirichlet"][:, 0].astype(int)
    T_bar = boundary_conditions["T_dirichlet"][:, 1:2]
    p_fix = boundary_conditions["p_dirichlet"][:, 0].astype(int)
    p_bar = boundary_conditions["p_dirichlet"][:, 1:2]

    F_extT = np.zeros((len(T_dofs), 1))
    F_extp = np.zeros((len(p_dofs), 1))
    F_extT[boundary_conditions["T_neumann"][:, 0].astype(int)] = boundary_conditions["T_neumann"][:, 1:2]
    F_extp[boundary_conditions["p_neumann"][:, 0].astype(int)] = boundary_conditions["p_neumann"][:, 1:2]
    return T_fix, T_bar, p_fix, p_bar, F_extT, F_extp


def compute_residuals_comp(
    F_vec: dict[str, Any],
    F_extp: npt.NDArray[np.float64],
    F_extT: npt.NDArray[np.float64],
    *,
    fully_coupled: bool = False,
    pressure_only: bool = False,
    temperature_only: bool = False,
) -> dict[str, npt.NDArray[np.float64]]:
    """Assemble integrated residuals alongside external force vectors."""
    if pressure_only or fully_coupled:
        F_Pstorage = F_vec["F_Pstorage"]
        F_Pdiffusion = F_vec["F_Pdiffusion"]

        R = {"p": F_Pstorage - F_Pdiffusion - F_extp}

    if temperature_only or fully_coupled:
        F_Tstorage = F_vec["F_Tstorage"]
        F_Tdiffusion = F_vec["F_Tdiffusion"]
        F_Tadvection = F_vec["F_Tadvection"]
        F_Tstab = F_vec["F_Tstab"]
        F_enth_bnd = F_vec.get("F_enth_bnd", 0)
        RT = F_Tstorage + F_Tdiffusion - F_Tadvection + F_Tstab - F_extT - F_enth_bnd

        if fully_coupled:
            R["T"] = RT
        else:
            R = {"T": RT}

    return R


def compute_residuals(
    F_vec: dict[str, Any],
    F_ext: dict[str, Any],
    *,
    fully_coupled: bool = False,
    pressure_only: bool = False,
    temperature_only: bool = False,
) -> dict[str, npt.NDArray[np.float64]]:
    """Assemble integrated residuals alongside external force vectors."""
    if pressure_only or fully_coupled:
        F_Pstorage = F_vec["F_Pstorage"]
        F_Pdiffusion = F_vec["F_Pdiffusion"]

        R = {"p": F_Pstorage - F_Pdiffusion - F_ext["p"]}

    if temperature_only or fully_coupled:
        F_Tstorage = F_vec["F_Tstorage"]
        F_Tdiffusion = F_vec["F_Tdiffusion"]
        F_Tadvection = F_vec["F_Tadvection"]
        F_Tstab = F_vec["F_Tstab"]
        F_enth_bnd = F_vec.get("F_enth_bnd", 0)
        RT = F_Tstorage + F_Tdiffusion - F_Tadvection + F_Tstab - F_ext["T"] - F_enth_bnd

        if fully_coupled:
            R["T"] = RT
        else:
            R = {"T": RT}

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
    n_DoFs = np.arange(0, mesh["nn_total"])  # number of DoFs
    nn = mesh["nn_total"]

    for key in R:
        nF = np.setdiff1d(n_DoFs, fixed_dofs[key]).astype(int)  # free DoFs

        numerator = np.linalg.norm(R[key][nF])
        flux_scale = np.linalg.norm(R[key][fixed_dofs[key]])

        NR[key][iteration] = numerator / flux_scale / nn if flux_scale > 1.0e-12 else numerator / nn
        NR["all"][iteration] += NR[key][iteration] ** 2

    NR["all"][iteration] = NR["all"][iteration] ** 0.5

    converged = bool(NR["all"][iteration] < NR_tol)

    return converged, NR["all"][iteration]


def _plot_nr_residuals(
    NR: dict[str, Any],
    aitken_state: dict[str, Any],
    iteration: int,
    iter_max: int,
    residual_keys: list[str],
) -> None:
    """Plot Newton-Raphson residual history and Aitken damping coefficients."""
    labels = {"T": "Temperature", "p": "Pressure"}
    styles = {"T": "r-", "p": "b-"}

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
    max_init_relax: float = 1.0,
    bounds: tuple[float, float] | None = None,
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
    max_init_relax : float
        Control parameter flag (NaN: no relaxation, <1e-8: fixed factor, >=1e-8: Aitken).
    bounds : tuple, optional
        Relaxation parameter bounds (min_relax, max_relax). Defaults to (0.001, 1.0).
    """
    if bounds is None:
        min_relax = 0.001
        max_relax = 10.0
    else:
        min_relax, max_relax = bounds

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

            delta = max(min(max_relax, abs(aitken_para)), min_relax) * del_sign

    if np.isnan(delta):
        delta = 1.0

    # Apply relaxation to damp the current guess vector
    sol_mp1_damped = (1.0 - delta) * sol_m + delta * sol_mp1

    # Return elements matching the recursive structure needed for next iterations
    return sol_mp1_damped, res_mp1, delta, sol_mp1_damped


def _newton_raphson_loop(  # noqa
    controls: dict[str, Any],
    material: dict[str, Any],
    mesh: dict[str, Any],
    primary_variables: dict[str, Any],
    BC_config: list[dict[str, Any]],
    T_flow: npt.NDArray[np.float64],
    cv: npt.NDArray[np.float64],  # rate coefficients
    keys: list[str],
    fixed_dofs: dict[str, Any],
    F_ext: dict[str, Any],
    verbose: bool,
    debug_plot_on: bool,
    *,
    pressure_only: bool = False,
    temperature_only: bool = False,
    fully_coupled: bool = False,
) -> tuple[dict[str, Any], bool, float]:
    """Perform non-linear Newton-Raphson iterations.

    `T_flow` must be the SAME array object as `primary_variables["T"]` when
    `fully_coupled=True` (so it tracks the in-loop updates below); for
    `temperature_only`/`pressure_only` it should be a separate frozen/lagged copy.
    """
    # Newton-Raphson controls
    iter_max = controls["iterative_controls"]["iter_max"]
    NR_tol = controls["iterative_controls"]["NR_tol"]
    aitken_on = controls["iterative_controls"]["aitken_relaxation"]
    NR_step = controls["iterative_controls"]["NR_step"]
    converged = False

    # Initaliaze
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
        free_dofs_key = np.setdiff1d(local_dofs, fixed_dofs[key]).astype(int) + key_dofs[key][0]
        aitken_state["variables_km1"] = np.vstack(
            (aitken_state["variables_km1"], primary_variables[key][:, 0:1])
        )
        aitken_state["relax_dofs"] = np.concatenate((aitken_state["relax_dofs"], free_dofs_key))

    nE = nE.ravel()
    aitken_state["aitken_coefficient"] = np.ones((iter_max + 2, 1))
    aitken_state["delta_variables_km1"] = np.zeros((nn_all, 1))

    # Run-type labels
    if fully_coupled:
        label = "FULLY COUPLED"

        # Indexing for updating reaction fluxes each iteration
        p_dofs_local = primary_variables["p_dofs"]
        nF_p = np.setdiff1d(p_dofs_local, fixed_dofs["p"]).astype(int)
    elif pressure_only:
        label = "ITERATIVE PRESSURE"
    else:
        label = "ITERATIVE TEMPERATURE"

    if verbose:
        print(f"      {label} NEWTON RAPHSON: ITERATION   ", end="")
    while True:
        if verbose:
            print(f"\r     {label} NEWTON RAPHSON: ITERATION {iteration + 1}   ", end="", flush=True)

        # Integrate residual vectors
        F_vec = integrate_residuals(
            mesh,
            material,
            primary_variables,
            controls,
            cv,
            T_flow,
            pressure_only=pressure_only,
            temperature_only=temperature_only,
        )

        # Update reaction flow rates (fully-coupled only)
        if fully_coupled:
            q = np.zeros((len(p_dofs_local), 1))
            R_p = compute_residuals(F_vec, F_ext, pressure_only=True)["p"]
            q[fixed_dofs["p"]] = R_p[fixed_dofs["p"]]
            q[nF_p] = q[nF_p] + F_ext["p"][nF_p]
            primary_variables["q"] = q

        # Calculate enthalpy boundary fluxes using reaction flow rates
        if temperature_only or fully_coupled:
            F_enth_bnd = integrate_enthlpy_bndry_residual(
                mesh, material, primary_variables, controls, T_flow, BC_config, T_flow_lagged=temperature_only
            )
            F_vec["F_enth_bnd"] = F_enth_bnd["F_enth_bnd"]

        # Compuate residuals
        R = compute_residuals(
            F_vec,
            F_ext,
            pressure_only=pressure_only,
            temperature_only=temperature_only,
            fully_coupled=fully_coupled,
        )

        # Check convergence
        converged, iteration_error = _check_convergence_NR(R, fixed_dofs, NR, mesh, iteration, NR_tol)

        if converged and iteration > 0:
            if verbose:
                print(" - CONVERGED")
            break
        else:  # Update iteration
            # Integrate Jacobian
            jacobian = integrate_tangent_matrices(
                mesh,
                material,
                primary_variables,
                controls,
                cv,
                T_flow,
                pressure_only=pressure_only,
                temperature_only=temperature_only,
                fully_coupled=fully_coupled,
                # T_flow is a frozen copy (not the unknown T) only in the
                # temperature-only lin-extrap loop; for fully_coupled it is
                # aliased to the actual unknown, so its derivative must flow through.
                T_flow_lagged=temperature_only,
            )

            # Warnings for poor conditioning
            if iteration == 0 and (fully_coupled or pressure_only):
                warn_nonmonotone_pressure_diffusion(jacobian["Kpp"])

            # Assemble Jacobian - different by case
            if pressure_only:
                A = jacobian["Kpp"]
                b = R["p"]
            elif temperature_only:
                K_enth_bnd = integrate_enthlpy_bndry_tangent(
                    mesh, material, primary_variables, controls, T_flow, BC_config, T_flow_lagged=True
                )
                A = jacobian["KTT"] - K_enth_bnd["K_enth_bnd"]
                b = R["T"]
            else:  # fully_coupled
                K_enth_bnd = integrate_enthlpy_bndry_tangent(
                    mesh,
                    material,
                    primary_variables,
                    controls,
                    T_flow,
                    BC_config,
                    fully_coupled=True,
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

            # Solve system of equations
            delta_var = _partition_and_solve_matrix(A, -b, np.arange(0, nn_all), nE)

            d_n = d_n + NR_step * delta_var.reshape(-1, 1)

            # Check for nan or inf data and break
            if np.any(np.isnan(d_n)) or np.any(np.isinf(d_n)):
                if verbose:
                    print(" - Nan or Inf detected, CONVERGENCE FAILED. Resetting timestep.")
                break

            # Apply relaxation
            if aitken_on:
                (
                    d_new,
                    aitken_state["delta_variables_km1"],
                    aitken_state["aitken_coefficient"][iteration + 1],
                    aitken_state["variables_km1"],
                ) = _aitken_relax(
                    iteration,
                    d_n,
                    aitken_state["variables_km1"],
                    aitken_state["relax_dofs"],
                    aitken_state["delta_variables_km1"],
                    aitken_state["aitken_coefficient"][iteration],
                )
            else:
                d_new = d_n

            d_n = d_new
            # Write the updated global vector back into each key's primary variable.
            # Uses in-place slice assignment (not a new array) so that a `T_flow`
            # passed in as the SAME object as primary_variables["T"] (fully_coupled
            # case) stays aliased and reflects this update automatically.
            for key in keys:
                primary_variables[key][:, 0:1] = d_n[key_dofs[key]]

            iteration += 1

        if iteration >= iter_max:
            if verbose:
                print(" - CONVERGENCE FAILED")
            break

    # Revert the keys if not converged
    if not converged:
        for key in keys:
            primary_variables[key] = vars_orig[key]
    # Caclulate reaction fluxes
    if pressure_only:
        # Reaction fluxes only need computing once. Fully_coupled is calculated in-loop.
        p_dofs_local = primary_variables["p_dofs"]
        nF_p = np.setdiff1d(p_dofs_local, fixed_dofs["p"]).astype(int)
        q = np.zeros((len(p_dofs_local), 1))
        q[fixed_dofs["p"]] = R["p"][fixed_dofs["p"]]
        q[nF_p] = q[nF_p] + F_ext["p"][nF_p]
        primary_variables["q"] = q

    if debug_plot_on:
        _plot_nr_residuals(NR, aitken_state, iteration, iter_max, keys)

    return primary_variables, converged, iteration_error


# ------------------------ Primary time integration function with linearized extrapoloation --------
def lin_extrap_time_integration(  # noqa
    controls: dict[str, Any],
    mesh: dict[str, Any],
    material: dict[str, Any],
    primary_variables: dict[str, Any],
    boundary_conditions: dict[str, Any],
    BC_config: list[dict[str, Any]],
    dt: list[float],
    n: int,
    verbose: bool,
) -> tuple[dict[str, Any], list[float], bool, float]:
    """Assemble FEM matrices and vectors and solve for solution at next timestep.

    Solution uses a linearized extrapoloation which lags temperature. Error
    is calculated between the predicted and calculated temperatures for convergence.
    """
    debug_pressure_plot_on = False
    debug_temperature_plot_on = False
    # extract vectors of nodal dofs
    p = primary_variables["p"]
    T = primary_variables["T"]
    p_orig = primary_variables["p"].copy()
    T_orig = primary_variables["T"].copy()
    p_dofs = primary_variables["p_dofs"]
    T_dofs = primary_variables["T_dofs"]
    rate_order = controls["rate_order"]

    # calculate rate coefficients
    cv, cx = rate_coefficients(dt, rate_order)

    # unpack boundary conditions
    T_fix, T_bar, p_fix, p_bar, F_extT, F_extp = _unpack_boundary_conditions(
        boundary_conditions, p_dofs, T_dofs
    )
    fixed_dofs = {"p": p_fix, "T": T_fix}
    F_ext = {"p": F_extp, "T": F_extT}

    # Apply fixed boundary conditions
    p[p_fix, :] = p_bar
    T[T_fix, :] = T_bar
    nF_T = np.setdiff1d(T_dofs, T_fix).astype(int)  # free temperature DoFs

    # Extrapolate temperature for free Dofs
    T_extrap = cx[0] * T[:, 1] + cx[1] * T[:, 2] + cx[2] * T[:, 3]
    T[nF_T, 0] = T_extrap[nF_T]

    # Frozen snapshot for flow-property evaluations: column 0 is the extrapolated/fixed
    # guess (kept constant through the temperature Newton loop, unlike T itself), columns
    # 1-3 are the real time history, needed as-is by the mass-storage rate terms.
    T_flow = T.copy()

    # Update pressure with extrapolated temperature
    primary_variables, p_converged, _ = _newton_raphson_loop(
        controls,
        material,
        mesh,
        primary_variables,
        BC_config,
        T_flow,  # Extrapolated tempearture
        cv,
        ["p"],
        fixed_dofs,
        F_ext,
        verbose,
        debug_pressure_plot_on,
        pressure_only=True,
    )

    # Update temperature only if pressure converged
    if p_converged:
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
            ["T"],
            fixed_dofs,
            F_ext,
            verbose,
            debug_temperature_plot_on,
            temperature_only=True,
        )

    # ---- Check for extrapolation error
    if p_converged and T_converged:
        # Check for extrapolation timestep error
        converged, timestep_error = _check_convergence_extrap(primary_variables["T"], T_extrap, controls, n)
    else:
        primary_variables["p"] = p_orig.copy()
        primary_variables["T"] = T_orig.copy()
        converged = False
        timestep_error = -1  # flag for Newton-Raphson failure

    return primary_variables, dt, converged, timestep_error


# --------------- Fully coupled time integration with Newton-Raphson iterations.
def fully_coupled_time_integration(  # noqa
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
    """Assemble FEM matrices and vectors and solve for solution at next timestep.

    This fully-coupled iterative scheme uses Newton-Raphson to solve for temperature
    and pressure monolithically.
    """
    debug_plot_on = False
    p_dofs = primary_variables["p_dofs"]
    T_dofs = primary_variables["T_dofs"]
    rate_order = controls["rate_order"]

    # calculate rate coefficients
    cv, cx = rate_coefficients(dt, rate_order)

    # unpack boundary conditions
    T_fix, T_bar, p_fix, p_bar, F_extT, F_extp = _unpack_boundary_conditions(
        boundary_conditions, p_dofs, T_dofs
    )
    fixed_dofs = {"p": p_fix, "T": T_fix}
    F_ext = {"p": F_extp, "T": F_extT}

    # Apply fixed boundary conditions
    primary_variables["p"][p_fix, :] = p_bar
    primary_variables["T"][T_fix, :] = T_bar

    primary_variables, converged, iteration_error = _newton_raphson_loop(
        controls,
        material,
        mesh,
        primary_variables,
        BC_config,
        primary_variables["T"],  # T_flow must alias primary_variables["T"] for fully_coupled
        cv,
        ["p", "T"],
        fixed_dofs,
        F_ext,
        verbose,
        debug_plot_on,
        fully_coupled=True,
    )

    return primary_variables, dt, converged, iteration_error
