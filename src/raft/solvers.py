"""Time integration solvers for subsurface systems."""

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from .matrixassembly import (
    integrate_NLpressure_matrices,
    integrate_nonlinear_matrices,
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
    sim_control: dict[str, Any],
    n: int,
) -> tuple[bool, float]:
    """Check convergence for the linear extrapolation."""
    rate_order = sim_control["rate_order"]
    rel_tol_T = sim_control["rel_error_T"]

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
    T_bar = boundary_conditions["T_dirichlet"][:, 1]
    p_fix = boundary_conditions["p_dirichlet"][:, 0].astype(int)
    p_bar = boundary_conditions["p_dirichlet"][:, 1]

    F_extT = np.zeros((len(T_dofs), 1))
    F_extp = np.zeros((len(p_dofs), 1))
    F_extT[boundary_conditions["T_neumann"][:, 0].astype(int)] = boundary_conditions["T_neumann"][
        :, 1
    ].reshape(-1, 1)
    F_extp[boundary_conditions["p_neumann"][:, 0].astype(int)] = boundary_conditions["p_neumann"][
        :, 1
    ].reshape(-1, 1)
    return T_fix, T_bar, p_fix, p_bar, F_extT, F_extp


def _partition_and_solve_matrix(
    A: sp.csc_matrix[Any],
    b: npt.NDArray[np.float64],
    dofs_all: npt.NDArray[np.float64],
    nE: npt.NDArray[np.float64],  # essential DoF indices
    dE: npt.NDArray[np.float64],  # essential DoF values
) -> npt.NDArray[np.float64]:
    """Partitions and solves a matrix given a set of essential DoFs. Returns full vector of DoFs."""
    nF = np.setdiff1d(dofs_all, nE)  # free DoFs

    KFF = A[np.ix_(nF, nF)]  # type: ignore[index]
    KFE = A[np.ix_(nF, nE)]  # type: ignore[index]

    dF = spsolve(KFF, (b[nF] - KFE @ dE.reshape(-1, 1)).ravel())

    d = np.empty((len(dofs_all), 1))
    d[nE, 0] = dE
    d[nF, 0] = dF.ravel()

    return d.ravel()


# ------------------------ Primary time integration function with linearized extrapoloation --------
def lin_extrap_time_integration(
    sim_control: dict[str, Any],
    mesh: dict[str, Any],
    material: dict[str, Any],
    matrices_linear: dict[str, Any],
    primary_variables: dict[str, Any],
    boundary_conditions: dict[str, Any],
    dt: list[float],
    n: int,
    verbose: bool,
) -> tuple[dict[str, Any], list[float], bool, float]:
    """Assemble FEM matrices and vectors and solve for solution at next timestep.

    Solution uses a linearized extrapoloation which lags temperature. Error
    is calculated between the predicted and calculated temperatures for convergence.
    """
    # extract vectors of nodal dofs
    p = primary_variables["p"]
    T = primary_variables["T"]
    p_dofs = primary_variables["p_dofs"]
    T_dofs = primary_variables["T_dofs"]
    # extract controls and tolerances
    rate_order = sim_control["rate_order"]
    adv_on = sim_control["advection_on"]
    prez_on = sim_control["pressurization_on"]
    buoy_on = sim_control["buoyancy_on"]
    steady_pres = sim_control["steady_pressure"]
    steady_temp = sim_control["steady_temperature"]
    mu_nonlinear = sim_control["viscosity_nonconstant"]
    # extract mesh characteristics
    nn = mesh["nn_total"]

    # calculate rate coefficients
    cv, cx = rate_coefficients(dt, rate_order)

    # unpack boundary conditions
    T_fix, T_bar, p_fix, p_bar, F_extT, F_extp = _unpack_boundary_conditions(
        boundary_conditions, p_dofs, T_dofs
    )

    # Begin the one-pass extrapolated linearization by solving pressure
    # first using an extrapolated pressure
    # One-pass update: solve pressure, then temperature

    T_extrap = cx[0] * T[:, 1] + cx[1] * T[:, 2] + cx[2] * T[:, 3]  # Extrapolated temperature
    T_dot = cv[0] * T_extrap + cv[1] * T[:, 1] + cv[2] * T[:, 2]  # Rate of temperature change

    KM = matrices_linear["KM"] * (1 - steady_pres)  # Pressure storage matrix
    KpT = matrices_linear["KpT"]  # Pressurization matrix

    if mu_nonlinear:
        print("     Assembling non-linear pressure matrices...")
        matrices_pressure_nonlinear = integrate_NLpressure_matrices(
            mesh, material, primary_variables, sim_control
        )
        Kpp = matrices_pressure_nonlinear["Kpp"]  # Pressure conductance matrix
        KbT = matrices_pressure_nonlinear["KbT"]  # Buoyancy matrix
    else:
        Kpp = matrices_linear["Kpp"]  # Pressure conductance matrix
        KbT = matrices_linear["KbT"]  # Buoyancy matrix

    if verbose:
        print("     Solving pressure equations...")
    F_pT = prez_on * (KpT @ T_dot).reshape(-1, 1)  # Pressurization force vector
    F_bT = buoy_on * (KbT @ (T_extrap - material["T_0"])).reshape(-1, 1)  # Buoyancy force vector

    # Assemble combined matrices
    A = (cv[0] * KM) + Kpp
    pdot = (cv[1] * p[:, 1] + cv[2] * p[:, 2]).reshape(-1, 1)
    b = -KM @ pdot + F_extp + F_bT + F_pT

    nE = p_fix.astype(int)  # essential DoFs
    nF = np.setdiff1d(p_dofs, nE).astype(int)  # free DoFs
    p[:, 0] = _partition_and_solve_matrix(A, b, p_dofs, nE, p_bar)

    # Calculate reaction flow rates at prescribed DoFs
    KEE = A[np.ix_(nE, nE)]
    KEF = A[np.ix_(nE, nF)]
    pF = p[nF, 0]
    q = np.zeros((nn, 1))
    q[p_fix] = KEE @ p_bar.reshape(-1, 1) + KEF @ pF.reshape(-1, 1) - b[nE]
    q = q + F_extp

    # Update nonlinear matrices with new pressure
    if verbose:
        print("     Assembling non-linear temperature matrices...")
    matrices_non_linear = integrate_nonlinear_matrices(mesh, material, primary_variables, sim_control)

    KTA = (matrices_non_linear["KATT"] + matrices_non_linear["K_stab_TT"]) * adv_on  # advection matrix
    KTT = matrices_linear["KTT"]  # thermal conduction matrix
    C = matrices_linear["C"] * (1 - steady_temp)  # thermal storage matrix

    if verbose:
        print("     Solving temperature equations...")
    A = (cv[0] * C) + KTA + KTT
    Tdot = (cv[1] * T[:, 1] + cv[2] * T[:, 2]).reshape(-1, 1)
    b = -C @ Tdot + F_extT

    T[:, 0] = _partition_and_solve_matrix(A, b, T_dofs, T_fix, T_bar)

    # Check for extrapolation error
    converged, timestep_error = _check_convergence_extrap(T, T_extrap, sim_control, n)

    primary_variables["p"] = p
    primary_variables["T"] = T
    primary_variables["q"] = q  # outlet flow rates
    return primary_variables, dt, converged, timestep_error


def _check_convergence_fullycoupled(
    R: dict[str, Any],
    fixed_dofs: dict[str, Any],
    NR: dict[str, Any],
    mesh: dict[str, Any],
    iteration: int,
    tolerance: float,
) -> tuple[bool, float]:

    n_DoFs = np.arange(0, mesh["nn"])  # number of DoFs
    ne = mesh["ne"]

    for key in R:
        nF = np.setdiff1d(n_DoFs, fixed_dofs[key]).astype(int)  # free DoFs

        NR[key][iteration] = np.linalg.norm(R[key][nF]) / ne  # norm of the residual per element
        NR["all"][iteration] += NR[key][iteration] ** 2

    NR["all"][iteration] = NR["all"][iteration] ** 0.5

    converged = bool(NR["all"][iteration] < tolerance)

    return converged, NR["all"][iteration]


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
        max_relax = 100.0
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


# --------------- Fully coupled time integration with Newton-Raphson iterations.
def fully_coupled_time_integration(  # noqa
    sim_control: dict[str, Any],
    mesh: dict[str, Any],
    material: dict[str, Any],
    matrices_linear: dict[str, Any],
    primary_variables: dict[str, Any],
    boundary_conditions: dict[str, Any],
    dt: list[float],
    n: int,  # noqa
    verbose: bool,
) -> tuple[dict[str, Any], list[float], bool, float]:
    """Assemble FEM matrices and vectors and solve for solution at next timestep.

    This fully-coupled iterative scheme uses Newton-Raphson to solve for temperature
    and pressure monolithically.
    """
    # extract vectors of nodal dofs
    p = primary_variables["p"]
    T = primary_variables["T"]
    p_dofs = primary_variables["p_dofs"]
    T_dofs = primary_variables["T_dofs"]
    # extract controls and tolerances
    rate_order = sim_control["rate_order"]
    iter_max = sim_control["iterative_controls"]["iter_max"]
    NR_tol = sim_control["iterative_controls"]["NR_tol"]
    aitken_on = sim_control["iterative_controls"]["aitken_relaxation"]

    adv_on = sim_control["advection_on"]
    prez_on = sim_control["pressurization_on"]
    buoy_on = sim_control["buoyancy_on"]
    steady_pres = sim_control["steady_pressure"]
    steady_temp = sim_control["steady_temperature"]
    mu_nonlinear = sim_control["viscosity_nonconstant"]

    # extract mesh characteristics
    nn = mesh["nn"]  # noqa
    T_dofs = T_dofs + nn  # shift for combined p-T vector

    # calculate rate coefficients
    cv, cx = rate_coefficients(dt, rate_order)

    # parse boundary conditions
    # unpack boundary conditions
    T_fix, T_bar, p_fix, p_bar, F_extT, F_extp = _unpack_boundary_conditions(
        boundary_conditions, p_dofs, T_dofs
    )

    # fixed dofs for error calculation
    fixed_dofs = {}
    fixed_dofs["p"] = p_fix
    fixed_dofs["T"] = T_fix

    # free and fixed dofs for partitioning with combined [p;T] vector
    nE = np.hstack((p_fix, T_fix + nn))  # fixed DoFs
    nF = np.setdiff1d(np.arange(0, 2 * nn), nE).astype(int)  # free DoFs

    # Apply fixed boundary conditions
    p[p_fix, :] = p_bar.reshape(-1, 1)
    T[T_fix, :] = T_bar.reshape(-1, 1)

    # Extract linear matrices
    KM = sp.csc_matrix((nn, nn)) if steady_pres else matrices_linear["KM"]  # Pressure storage matrix
    C = sp.csc_matrix((nn, nn)) if steady_temp else matrices_linear["C"]  # thermal storage matrix
    KpT = matrices_linear["KpT"]  # Pressurization matrix
    KTT = matrices_linear["KTT"]  # thermal conduction matrix

    # Begin Newton-Raphson iterative solver
    converged = False
    iteration = 0  # noqa
    # Store iteration residuals for debugging
    NR = {}
    R = {}
    NR["all"] = np.zeros(iter_max + 1)
    NR["T"] = np.zeros(iter_max + 1)
    NR["p"] = np.zeros(iter_max + 1)
    # Initialize relaxation
    aitken_state = {}
    aitken_state["aitken_coefficient"] = np.ones((iter_max + 2, 1))
    aitken_state["variables_km1"] = np.vstack((p[:, 0].reshape(-1, 1), T[:, 0].reshape(-1, 1)))
    aitken_state["delta_variables_km1"] = np.zeros((2 * nn, 1))
    aitken_state["relax_dofs"] = nF

    if verbose:
        print("      FULLY COUPLED NEWTON RAPHSON: ITERATION   ", end="")

    while True:
        if verbose:
            print(f"\r     FULLY COUPLED NEWTON RAPHSON: ITERATION {iteration + 1}   ", end="", flush=True)

        # Compute residual vectors
        matrices_non_linear = integrate_nonlinear_matrices(mesh, material, primary_variables, sim_control)

        if mu_nonlinear:
            Kpp = matrices_non_linear["Kpp"]  # Pressure conductance matrix
            KbT = matrices_non_linear["KbT"]  # Buoyancy matrix
            KpT_mu_b = matrices_non_linear["KpT_mu_b"]
            KpT_mu_p = matrices_non_linear["KpT_mu_p"]
            KTA_mu = matrices_non_linear["KATT_mu"] + matrices_non_linear["K_stab_TT_mu"]
        else:
            Kpp = matrices_linear["Kpp"]  # Pressure conductance matrix
            KbT = matrices_linear["KbT"]  # Buoyancy matrix
            KpT_mu_b = sp.csc_matrix((nn, nn))
            KpT_mu_p = sp.csc_matrix((nn, nn))
            KTA_mu = sp.csc_matrix((nn, nn))

        Tdot = (cv[0] * T[:, 0] + cv[1] * T[:, 1] + cv[2] * T[:, 2]).reshape(-1, 1)
        pdot = (cv[0] * p[:, 0] + cv[1] * p[:, 1] + cv[2] * p[:, 2]).reshape(-1, 1)
        T_n = T[:, 0].reshape(-1, 1)
        p_n = p[:, 0].reshape(-1, 1)

        F_bT = (
            (KbT @ (T[:, 0] - material["T_0"])).reshape(-1, 1) if buoy_on else np.zeros((nn, 1))
        )  # Buoyancy force vector
        F_pT = (KpT @ Tdot).reshape(-1, 1) if prez_on else np.zeros((nn, 1))  # Pressurization force vector

        KTA = (matrices_non_linear["KATT"] + matrices_non_linear["K_stab_TT"]) * adv_on  # advection matrix

        # Pressure residual
        R["p"] = -KM @ pdot - Kpp @ p_n + F_extp + F_bT + F_pT
        # Temperature residual
        R["T"] = -C @ Tdot - (KTT + KTA) @ T_n + F_extT

        # Check convergence
        converged, iteration_error = _check_convergence_fullycoupled(
            R, fixed_dofs, NR, mesh, iteration, NR_tol
        )

        if converged and iteration > 1:
            if verbose:
                print(" - CONVERGED")
            break
        else:  # Update iteration
            # Diagonal matrices

            KATp = matrices_non_linear["KATp"]
            K_stab_Tp = matrices_non_linear["K_stab_Tp"]

            # Assemble Jacobian matrix
            App = -KM * cv[0] - Kpp  # correct
            ApT = prez_on * (cv[0] * KpT) + buoy_on * (KbT + KpT_mu_b) + KpT_mu_p  # issues?
            ATp = -(KATp + K_stab_Tp) * adv_on  # correct
            ATT = -cv[0] * C - (KTT + KTA) + KTA_mu  # correct

            # Total combined matrix
            d = np.vstack((p[:, 0].reshape(-1, 1), T[:, 0].reshape(-1, 1)))
            A = cast(sp.csc_matrix[Any], sp.bmat([[App, ApT], [ATp, ATT]], format="csc"))

            b = np.vstack((R["p"], R["T"]))

            delta_var = _partition_and_solve_matrix(A, -b, np.arange(0, 2 * nn), nE, np.zeros(np.size(nE, 0)))

            NR_step = 0.5
            d += NR_step * delta_var.reshape(-1, 1)

            # Apply relaxation
            if aitken_on:
                (
                    d_new,
                    aitken_state["delta_variables_km1"],
                    aitken_state["aitken_coefficient"][iteration + 1],
                    aitken_state["variables_km1"],
                ) = _aitken_relax(
                    iteration,
                    d,
                    aitken_state["variables_km1"],
                    aitken_state["relax_dofs"],
                    aitken_state["delta_variables_km1"],
                    aitken_state["aitken_coefficient"][0],
                )
            else:
                d_new = d

            p[:, 0] = d_new[p_dofs, 0]
            T[:, 0] = d_new[T_dofs, 0]
            iteration += 1

        if iteration >= iter_max:
            print(" - CONVERGENCE FAILED")
            break

    # Calculate reaction flow rates at prescribed DoFs
    q = np.zeros((nn, 1))
    q[p_fix] = R["p"][p_fix]
    # save variables
    primary_variables["p"] = p
    primary_variables["T"] = T
    primary_variables["q"] = q  # outlet flow rates

    if verbose:
        print()  # new line for that output doesn't get overwritten
        # Plot residuals
        fig, ax1 = plt.subplots(figsize=(8, 8))
        ax2 = ax1.twinx()

        ax1.semilogy(np.arange(1, iteration + 1), NR["T"][range(0, iteration)], "r-")
        ax1.semilogy(np.arange(1, iteration + 1), NR["p"][range(0, iteration)], "b-")
        ax1.semilogy(np.arange(1, iteration + 1), NR["all"][range(0, iteration)], "ko-")
        plt.ylabel("residual")
        plt.legend(["Temperature", "Pressure", "Combined"])

        ax2.plot(
            np.arange(1, iteration + 1),
            np.array(aitken_state["aitken_coefficient"])[range(0, iteration)],
            "go-",
        )
        plt.ylabel("damping coefficient")

        plt.xlim([0, iter_max])
        # plt.ylim([1e-15, 1e2])
        plt.xlabel("iteration")

    return primary_variables, dt, converged, NR["all"][iteration - 1]
