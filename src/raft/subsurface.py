"""Subsurface Simulator."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.sparse.linalg import spsolve

from .matrixassembly import integrate_linear_matrices, integrate_nonlinear_matrices
from .parseboundaryconditions import parsebcs
from .postprocessing import postprocessing
from .preprocessing import preprocessing


def rate_coefficients(dt, order):
    """Calculate the rate of change coefficients."""
    # dt1 = t_n - t_(n-1)
    # dt2 = t_(n-1) - t_(n-2)
    # dt3 = t_(n-2) - t_(n-3)
    dt1 = dt[0]
    dt2 = dt[1]
    dt3 = dt[2]
    cv = np.ones(3)  # rate of change coefficients
    cx = np.ones(3)  # extrapolation coefficients
    if order > 1:
        cv[0] = (2 * dt1 + dt2) / dt1 / (dt1 + dt2)
        cv[1] = -(dt1 + dt2) / dt1 / dt2
        cv[2] = dt1 / dt2 / (dt1 + dt2)

        cx[0] = (dt1 + dt2 + dt3) * (dt1 + dt2) / (dt2 + dt3) / dt2
        cx[1] = -dt1 * (dt1 + dt2 + dt3) / dt2 / dt3
        cx[2] = dt1 * (dt1 + dt2) / (dt2 + dt3) / dt3
    else:
        cv[0] = 1 / dt1
        cv[1] = -1 / dt1
        cv[2] = 0

        cx[0] = 1 + dt1 / dt2
        cx[1] = -dt1 / dt2
        cx[2] = 0
    return cv, cx


def _update_time(t, dt, t_end, dt_max):
    """Update the current time and timestep."""
    dt[2] = dt[1]
    dt[1] = dt[0]
    if dt[0] < dt_max:
        dt[0] *= 1.25  # Note: arbitrary timestep increasing rate. Could be made variable.
    t += dt[0]
    if t > t_end and t != t_end:
        t -= dt[0]
        dt[0] = t_end - t
        t = t_end
    return t, dt


def _check_convergence(T, T_extrap, dt, sim_control, n, verbose):
    rel_tol_T = sim_control["rel_error_T"]
    dt_min = sim_control["timestep_min"]
    rate_order = sim_control["rate_order"]
    converged = 1  # default to converged, and only break if not converged

    # First few timesteps affected by initial conditions, do not check
    if n < rate_order + 1:
        return converged

    # Calculate error
    extrapolation_error = max(abs(T[:, 0] - T_extrap) / T[:, 0])
    if extrapolation_error > rel_tol_T:
        if dt[0] != dt_min:
            cut_factor = 0.85 * 4 / (1 + 3 * extrapolation_error / rel_tol_T)
            dt[0] = max(cut_factor * dt[0], dt_min)
            converged = 0
            if verbose:
                print(
                    f"     Extrapolation error = {extrapolation_error * 100:.2f} %, reducing timestep by {cut_factor:.2f} to dt = {dt[0]:.2e}"  # noqa: E501
                )
        else:
            converged = 1
            if verbose:
                print(
                    f"     Minimum timestep reached, error is {extrapolation_error * 100:.2f} %. Proceeding with simulation."  # noqa: E501
                )
    return converged


# ------------------------ Primary time integration function --------
def _lin_extrap_time_integration(
    sim_control: dict[str, Any],
    mesh: dict[str, Any],
    material: dict[str, Any],
    matrices_linear: dict[str, Any],
    primary_variables: dict[str, Any],
    boundary_conditions: dict[str, Any],
    dt: npt.NDArray[np.float64],
    n: int,
    verbose: bool,
):
    """Assemble FEM matrices and vectors and solve for solution at next timestep."""
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

    converged = 0
    while not converged:
        # calculate rate coefficients
        cv, cx = rate_coefficients(dt, rate_order)

        # parse boundary conditions
        # dirichlet - fixed
        T_fix = boundary_conditions["T_dirichlet"][:, 0].astype(int)
        T_bar = boundary_conditions["T_dirichlet"][:, 1]
        p_fix = boundary_conditions["p_dirichlet"][:, 0].astype(int)
        p_bar = boundary_conditions["p_dirichlet"][:, 1]
        # neumann - external force vectors
        F_extT = np.zeros((len(T_dofs), 1))
        F_extp = np.zeros((len(p_dofs), 1))
        F_extT[boundary_conditions["T_neumann"][:, 0].astype(int)] = boundary_conditions["T_neumann"][
            :, 1
        ].reshape(-1, 1)
        F_extp[boundary_conditions["p_neumann"][:, 0].astype(int)] = boundary_conditions["p_neumann"][
            :, 1
        ].reshape(-1, 1)

        # Begin the one-pass extrapolated linearization by solving pressure
        # first using an extrapolated pressure
        # One-pass update: solve pressure, then temperature
        if verbose:
            print("     Solving pressure equations...")

        T_extrap = cx[0] * T[:, 1] + cx[1] * T[:, 2] + cx[2] * T[:, 3]  # Extrapolated temperature
        T_dot = cv[0] * T_extrap + cv[1] * T[:, 1] + cv[2] * T[:, 2]  # Rate of temperature change

        KM = matrices_linear["KM"] * (1 - steady_pres)  # Pressure storage matrix
        Kpp = matrices_linear["Kpp"]  # Pressure conductance matrix
        KpT = matrices_linear["KpT"]  # Pressurization matrix
        KbT = matrices_linear["KbT"]  # Buoyancy matrix

        F_pT = prez_on * (KpT @ T_dot).reshape(-1, 1)  # Pressurization force vector
        F_bT = buoy_on * (KbT @ (T_extrap - material["T_0"])).reshape(-1, 1)  # Buoyancy force vector

        # Assemble combined matrices
        A = (cv[0] * KM) + Kpp
        b = -KM @ (cv[1] * p[:, 1] + cv[2] * p[:, 2]).reshape(-1, 1) + F_extp + F_bT + F_pT

        nE = p_fix.astype(int)  # essential DoFs
        nF = np.setdiff1d(p_dofs, nE).astype(int)  # free DoFs
        dE = p_bar
        KFF = A[np.ix_(nF, nF)]
        KFE = A[np.ix_(nF, nE)]
        pF = spsolve(KFF, (b[nF] - KFE @ dE.reshape(-1, 1)).ravel())
        p[nE, 0] = dE
        p[nF, 0] = pF.ravel()

        # Calculate reaction flow rates at prescribed DoFs
        KEE = A[np.ix_(nE, nE)]
        KEF = A[np.ix_(nE, nF)]
        qE = KEE @ dE.reshape(-1, 1) + KEF @ pF.reshape(-1, 1) - b[nE]

        # Update nonlinear matrices with new pressure
        if verbose:
            print("     Assembling non-linear matrices...")
        matrices_non_linear = integrate_nonlinear_matrices(mesh, material, primary_variables)

        KTA = (matrices_non_linear["KATT"] + matrices_non_linear["K_stab_TT"]) * adv_on  # advection matrix
        KTT = matrices_linear["KTT"]  # thermal conduction matrix
        C = matrices_linear["C"] * (1 - steady_temp)  # thermal storage matrix

        if verbose:
            print("     Solving temperature equations...")
        A = (cv[0] * C) + KTA + KTT
        b = -C @ (cv[1] * T[:, 1] + cv[2] * T[:, 2]).reshape(-1, 1) + F_extT

        nE = T_fix  # essential DoFs
        nF = np.setdiff1d(T_dofs, nE)  # free DoFs
        dE = T_bar
        KFF = A[np.ix_(nF, nF)]
        KFE = A[np.ix_(nF, nE)]
        TF = spsolve(KFF, (b[nF] - KFE @ dE.reshape(-1, 1)).ravel())
        T[nE, 0] = dE
        T[nF, 0] = TF.ravel()

        # Check for extrapolation error
        converged = _check_convergence(T, T_extrap, dt, sim_control, n, verbose)

    primary_variables["p"] = p
    primary_variables["T"] = T
    primary_variables["q_out"] = qE  # outlet flow rates
    primary_variables["T_out"] = T[p_fix, 0]  # outlet temperatures
    return primary_variables, dt


# -------------------------------- Main simulation run function -------------
def main(config_name: str | None = None, config_dir: Path | None = None, verbose: bool = True):
    """Main simulation run."""
    if config_name is None:
        parser = argparse.ArgumentParser(description="RAFT - Reservoir Analysis for TANGO")
        parser.add_argument("config", help="Name of the config file (e.g. config_example.yaml) in CONFIG_DIR")
        args = parser.parse_args()
        config_name = args.config

    if verbose:
        print("RAFT - RESERVOIR ANALYSIS FOR TANGO \n")
        print("Beginning simulation " + config_name + "\n")

    # ----- Load config and mesh
    config, mesh = preprocessing(config_name, config_dir, verbose)
    material = config["material"]

    # ----- Initialize simulation parameters

    ## Numerical controls
    t = config["simulation"]["t_start"]
    t_end = config["simulation"]["t_end"]
    dt = config["simulation"]["timestep"] * np.ones(3)

    ## Initialize vectors
    primary_variables = {}  # storage structure for all primary variables and results
    p_dofs = np.arange(0, mesh["nn"])
    T_dofs = np.arange(0, mesh["nn"])
    primary_variables["p_dofs"] = p_dofs
    primary_variables["T_dofs"] = T_dofs
    # Vectors of primary DoFs - ndof x 3
    # [t_n, t_(n-1), t_(n-2)]
    p = np.zeros((len(p_dofs), 4))
    T = np.zeros((len(T_dofs), 4))

    # Linear Matrix Assembly
    #   note for future use: If a no-flow boundary is desired in the direction of buoyancy,
    #                        there should be an edge integral perform to remove its contribution
    #                        along the upper surfaces.
    if verbose:
        print("     Assembling linear matrices...")
    matrices_linear, material, mesh = integrate_linear_matrices(mesh, material)

    # --- Initial Conditions
    # assume uniform, non-uniform to be implemented in future.
    p[p_dofs, :] = config["initial_conditions"]["p_surf"]
    T[T_dofs, :] = config["initial_conditions"]["T_surf"]
    material["T_0"] = T[:, 1]  # initial condition reference temperature
    primary_variables["p"] = p
    primary_variables["T"] = T

    outlet_nodes = config["boundary_conditions"]["pressure_fixed"][0]["nodes"]
    q_out = np.zeros((len(outlet_nodes), 1))
    primary_variables["q_out"] = q_out
    primary_variables["T_out"] = T[outlet_nodes, 0]

    # --- Export initial condition
    if verbose:
        print("Export initial condition")
    n = 0  # Time step counter
    postprocessing(n, mesh, config, primary_variables, t, verbose)
    n += 1

    # ---- Time integration
    while True:
        if verbose:
            print(f"Timestep {n}, t = {t:.2e}, {(100 * t / t_end):.2f} % complete")

        # Update Boundary Conditions
        boundary_conditions = parsebcs(config["boundary_conditions"], mesh, primary_variables, t)

        # Perform time integration
        primary_variables, dt = _lin_extrap_time_integration(
            config["simulation"],
            mesh,
            material,
            matrices_linear,
            primary_variables,
            boundary_conditions,
            dt,
            n,
            verbose,
        )

        # Post-processing
        if verbose:
            print("     Postprocessing...")
        postprocessing(n, mesh, config, primary_variables, t, verbose)

        # Update time step
        t, dt = _update_time(t, dt, t_end, config["simulation"]["timestep_max"])
        n += 1
        # Check end condition
        if t >= t_end:
            if verbose:
                print("Simulation complete. Output files are stored in: \n" + config["output_directory_full"])
            break

        # Initialize next timestep
        # use roll to update primary variables, column 0 gets overwritten in time integration
        primary_variables["p"][:, [3, 2, 1]] = primary_variables["p"][:, [2, 1, 0]]
        primary_variables["T"][:, [3, 2, 1]] = primary_variables["T"][:, [2, 1, 0]]

    return primary_variables, mesh


if __name__ == "__main__":
    main()
