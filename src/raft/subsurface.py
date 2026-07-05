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


def _update_time(
    t: float, dt: list[float], dt_max: float, critical_times: npt.NDArray[np.float64]
) -> tuple[float, list[float]]:
    """Update the current time and timestep."""
    # update timestep
    dt[2] = dt[1]
    dt[1] = dt[0]
    if dt[0] < dt_max:
        dt[0] = min(1.25 * dt[0], dt_max)  # Note: arbitrary timestep increasing rate. Could be made variable.
    t += dt[0]

    # find next critical time and prevent overshoot
    upcoming = [ct for ct in critical_times if ct > t - dt[0]]
    if upcoming:
        t_next = min(upcoming)
        if t > t_next and t != t_next:
            t -= dt[0]
            dt[0] = t_next - t
            t = t_next

    return t, dt


def get_critical_times(bc_config: dict[str, Any], t_end: float) -> npt.NDArray[np.float64]:
    """Collect all times from BC groups using the 'table' function."""
    times: set[float] = set()
    # add end times
    times.add(t_end)
    # search for any boundary condition change times
    for key in ("temperature_fixed", "pressure_fixed", "temperature_flux", "fluid_flux"):
        for group in bc_config.get(key, []):
            if group.get("function") == "table":
                times.update(group["values"][0])
            elif group.get("function") == "ramp":
                times.add(float(group["values"][1]))
    return np.array(sorted(times), dtype=np.float64)


def _check_convergence(
    T: npt.NDArray[np.float64],
    T_extrap: npt.NDArray[np.float64],
    dt: list[float],
    sim_control: dict[str, Any],
    n: int,
    verbose: bool,
) -> int:
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


def _partition_and_solve_matrix(
    A: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    dofs_all: npt.NDArray[np.float64],
    nE: npt.NDArray[np.float64],  # essential DoF indices
    dE: npt.NDArray[np.float64],  # essential DoF values
) -> npt.NDArray[np.float64]:
    """Partitions and solves a matrix given a set of essential DoFs. Returns full vector of DoFs."""
    nF = np.setdiff1d(dofs_all, nE)  # free DoFs

    KFF = A[np.ix_(nF, nF)]
    KFE = A[np.ix_(nF, nE)]

    dF = spsolve(KFF, (b[nF] - KFE @ dE.reshape(-1, 1)).ravel())

    d = np.empty((len(dofs_all), 1))
    d[nE, 0] = dE
    d[nF, 0] = dF.ravel()

    return d.ravel()


# ------------------------ Primary time integration function --------
def _lin_extrap_time_integration(
    sim_control: dict[str, Any],
    mesh: dict[str, Any],
    material: dict[str, Any],
    matrices_linear: dict[str, Any],
    primary_variables: dict[str, Any],
    boundary_conditions: dict[str, Any],
    dt: list[float],
    n: int,
    verbose: bool,
) -> tuple[dict[str, Any], list[float]]:
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
    # extract mesh characteristics
    nn = mesh["nn"]

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
            print("     Assembling non-linear matrices...")
        matrices_non_linear = integrate_nonlinear_matrices(mesh, material, primary_variables)

        KTA = (matrices_non_linear["KATT"] + matrices_non_linear["K_stab_TT"]) * adv_on  # advection matrix
        KTT = matrices_linear["KTT"]  # thermal conduction matrix
        C = matrices_linear["C"] * (1 - steady_temp)  # thermal storage matrix

        if verbose:
            print("     Solving temperature equations...")
        A = (cv[0] * C) + KTA + KTT
        b = -C @ (cv[1] * T[:, 1] + cv[2] * T[:, 2]).reshape(-1, 1) + F_extT

        T[:, 0] = _partition_and_solve_matrix(A, b, T_dofs, T_fix, T_bar)

        # Check for extrapolation error
        converged = _check_convergence(T, T_extrap, dt, sim_control, n, verbose)

    primary_variables["p"] = p
    primary_variables["T"] = T
    primary_variables["q"] = q  # outlet flow rates
    return primary_variables, dt


def _assign_initial_gradient(
    nodes: npt.NDArray[np.float64], y0: float, slope: float, direction: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Gradient calculation for initial conditions based on arbitrary direction."""
    s = direction / np.linalg.norm(direction)  # unit vector in gradient direction

    x = np.dot(nodes, s)
    x_min = np.min(x)
    y = y0 + slope * (x - x_min)

    return np.asarray(y, dtype=np.float64)


# ------------------ Solve for initial condifions
def _assign_initial_conditions(
    config_ini: dict[str, Any],
    sim_control: dict[str, Any],
    mesh: dict[str, Any],
    material: dict[str, Any],
    matrices_linear: dict[str, Any],
    primary_variables: dict[str, Any],
    boundary_conditions: dict[str, Any],
    n: int,
    verbose: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assign initial conditions."""
    if verbose:
        print("Assigning initial conditions.")

    assign_type = config_ini["assignment"]
    supported_types = ["constant", "gradient", "BC_compatible"]

    if assign_type == "constant":
        # Apply constant uniform initial conditions
        primary_variables["p"][:, :] = config_ini["p_ini"]
        primary_variables["T"][:, :] = config_ini["T_ini"]

    elif assign_type == "gradient":
        # Apply gradient initial conditions
        # temperature
        T = _assign_initial_gradient(
            mesh["nodes"], config_ini["T_ini"], config_ini["T_grad"], config_ini["grad_direction_T"]
        )

        # pressure
        p = _assign_initial_gradient(
            mesh["nodes"], config_ini["p_ini"], config_ini["p_grad"], config_ini["grad_direction_p"]
        )

        primary_variables["p"][:, :] = np.matlib.repmat(p.reshape(-1, 1), 1, 4)
        primary_variables["T"][:, :] = np.matlib.repmat(T.reshape(-1, 1), 1, 4)
    elif assign_type == "BC_compatible":
        # Create dummy sim_control for SS initial condition.
        #
        # In general, we should use a non-linear solver to find the steady-state
        # condition for pressure and temperature. However, we can neglect pressurization
        # since it's time-dependent, and we can neglect buoyancy because the steady-state
        # initial condition has no reference temperature. Buoyancy would instead
        # be accounted for in the hydrostatic pressure, which does not affect flow.
        #
        # This reduces the problem to a one-way p -> T, just like in our linearized
        # extrapolation. Note that future non-linear implementations may change this.
        sim_control_SS = sim_control.copy()
        sim_control_SS["pressurization_on"] = 0
        sim_control_SS["buoyancy_on"] = 0
        sim_control_SS["steady_pressure"] = 1
        sim_control_SS["steady_temperature"] = 1
        material["T_0"] = np.zeros((mesh["nn"],))

        primary_variables, dt = _lin_extrap_time_integration(
            sim_control_SS,
            mesh,
            material,
            matrices_linear,
            primary_variables,
            boundary_conditions,
            [1, 1, 1],
            n,
            verbose,
        )

        p = primary_variables["p"][:, 0]
        T = primary_variables["T"][:, 0]
        primary_variables["p"][:, :] = np.matlib.repmat(p.reshape(-1, 1), 1, 4)
        primary_variables["T"][:, :] = np.matlib.repmat(T.reshape(-1, 1), 1, 4)
    else:
        raise ValueError(
            f"Unknown initial condition assignment type: '{assign_type}'\n"
            f"Supported shapes: {', '.join(supported_types)}\n"
        )

    # use initial conditions as the reference temperature for buoyancy
    material["T_0"] = primary_variables["T"][:, 0]  # initial condition reference temperature

    return primary_variables, material


# -------------------------------- Main simulation run function -------------
def main(
    config_name: str | None = None, config_dir: Path | None = None, verbose: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    n = 0  # Time step counter
    t = config["simulation"]["t_start"]
    t_end = config["simulation"]["t_end"]
    # initialize timesteps and store 3 timesteps (necessary for higher order)
    # dt = [dt_n, dt_(n-1), dt_(n-2)]
    dt = config["simulation"]["timestep"] * np.ones(3)
    dt_max = config["simulation"]["timestep_max"]
    # Gather any important time points as specified in the boundary conditions
    critical_times = get_critical_times(config["boundary_conditions"], t_end)

    ## Initialize vectors
    primary_variables = {}  # storage structure for all primary variables and results
    p_dofs = np.arange(0, mesh["nn"])
    T_dofs = np.arange(0, mesh["nn"])
    primary_variables["p_dofs"] = p_dofs
    primary_variables["T_dofs"] = T_dofs
    # Vectors of primary DoFs - ndof x 4
    # [t_n, t_(n-1), t_(n-2), ...]
    primary_variables["p"] = np.zeros((len(p_dofs), 4))
    primary_variables["T"] = np.zeros((len(T_dofs), 4))

    # Linear Matrix Assembly
    #   note for future use: If a no-flow boundary is desired in the direction of buoyancy,
    #                        there should be an edge integral perform to remove its contribution
    #                        along the upper surfaces.
    if verbose:
        print("     Assembling linear matrices...")
    matrices_linear, material, mesh = integrate_linear_matrices(mesh, material)

    # --- Initial Conditions
    # Update Boundary Conditions for initial step
    boundary_conditions = parsebcs(config["boundary_conditions"], mesh, primary_variables, t)
    q = np.zeros((mesh["nn"], 1))
    primary_variables["q"] = q

    [primary_variables, T_0] = _assign_initial_conditions(
        config["initial_conditions"],
        config["simulation"],
        mesh,
        material,
        matrices_linear,
        primary_variables,
        boundary_conditions,
        n,
        verbose,
    )

    # --- Export initial condition
    if verbose:
        print("Export initial condition")
    postprocessing(n, mesh, config, material, primary_variables, t, verbose)
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
        postprocessing(n, mesh, config, material, primary_variables, t, verbose)

        # Update time step
        t, dt = _update_time(t, dt, dt_max, critical_times)
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
