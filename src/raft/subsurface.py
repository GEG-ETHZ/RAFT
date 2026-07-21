"""Subsurface simulator - main time integration loop."""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .matrixassembly import integrate_linear_matrices
from .parseboundaryconditions import parsebcs
from .postprocessing import postprocessing
from .preprocessing import preprocessing
from .solvers import fully_coupled_time_integration, lin_extrap_time_integration


def _update_time(  # noqa
    solver_type: str,
    t: float,
    dt: list[float],
    sim_control: dict[str, Any],
    critical_times: npt.NDArray[np.float64],
    converged: bool,
    timestep_error: float,
    verbose: bool,
) -> tuple[float, list[float], bool]:
    """Update the current time and timestep."""
    dt_max = sim_control["timestep_max"]
    dt_min = sim_control["timestep_min"]
    rel_tol_T = sim_control["rel_error_T"]

    if not converged:  # Decrease the timestep
        if dt[0] == dt_min:
            # minimum timestep reached, proceed with simulation
            match solver_type:
                case "linearized":
                    if verbose:
                        print(
                            f"     Minimum timestep reached, error is {timestep_error * 100:.2f} %. Proceeding with simulation."  # noqa: E501
                        )
                case "fully_coupled":
                    if verbose:
                        print(
                            "     Minimum timestep reached, unable to converge to solution."  # noqa: E501
                        )
                    sys.exit()
        else:
            if solver_type == "linearized":
                # Cut timestep in proportion with error
                cut_factor = 0.85 * 4 / (1 + 3 * timestep_error / rel_tol_T)
            else:
                cut_factor = 0.5
            t = t - dt[0]
            dt[0] = max(cut_factor * dt[0], dt_min)
            t = t + dt[0]
            if verbose:
                print(
                    f"     Timestep error = {timestep_error * 100:.2f} %, reducing timestep by {cut_factor:.2f} to dt = {dt[0]:.2e}"  # noqa: E501
                )
            return t, dt, False

    # otherwise, increase the timestep
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
            dt[0] = max(t_next - t, dt_min)
            t += dt[0]

    return t, dt, True


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


def _assign_initial_gradient(
    nodes: npt.NDArray[np.float64], y0: float, slope: float, direction: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Gradient calculation for initial conditions based on arbitrary direction."""
    s = direction / np.linalg.norm(direction)  # unit vector in gradient direction

    x = np.dot(nodes, s)
    x_min = np.min(x)
    y = y0 + slope * (x - x_min)

    return np.asarray(y, dtype=np.float64)


def _broadcast_to_history(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Broadcast a per-node scalar field into the 4-column time-history layout."""
    return np.tile(values.reshape(-1, 1), (1, 4))


def _build_steady_state_controls(sim_control: dict[str, Any]) -> dict[str, Any]:
    """Build a simulation-control dict for the steady-state initial condition solve.

    Pressurization is neglected since it is time-dependent, and buoyancy is neglected
    because the steady-state initial condition has no reference temperature. Buoyancy
    is instead accounted for in the hydrostatic pressure, which does not affect flow.
    """
    sim_control_SS = sim_control.copy()
    sim_control_SS["pressurization_on"] = 0
    sim_control_SS["buoyancy_on"] = 0
    sim_control_SS["steady_pressure"] = 1
    sim_control_SS["steady_temperature"] = 1
    return sim_control_SS


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
        mesh["nodes"], config_ini["T_ini"], config_ini["T_grad"], config_ini["grad_direction_T"]
    )
    p = _assign_initial_gradient(
        mesh["nodes"], config_ini["p_ini"], config_ini["p_grad"], config_ini["grad_direction_p"]
    )

    primary_variables["p"][:, :] = _broadcast_to_history(p)
    primary_variables["T"][:, :] = _broadcast_to_history(T)
    return primary_variables


def _assign_ic_bc_compatible(
    sim_control: dict[str, Any],
    mesh: dict[str, Any],
    material: dict[str, Any],
    matrices_linear: dict[str, Any],
    primary_variables: dict[str, Any],
    boundary_conditions: dict[str, Any],
    n: int,
    verbose: bool,
) -> dict[str, Any]:
    """Solve for a steady-state initial condition compatible with the boundary conditions."""
    if verbose:
        print("   Solving for steady-state initial condition compatible with boundary conditions:")

    sim_control_SS = _build_steady_state_controls(sim_control)
    material["T_0"] = np.zeros((mesh["nn"],))

    if sim_control_SS["viscosity_nonconstant"]:
        # viscosity is a function of temperature, fully coupled steady-state
        # solution is required.
        if verbose:
            print("   Non-linear properties present, using sequential scheme to estimate fields...")

        # Use one-pass linear to generate approximate solution
        num_passes = 1
        for _ in range(num_passes):
            primary_variables, _, _, _ = lin_extrap_time_integration(
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

        # Perform fully coupled solution
        if verbose:
            print("   Solving for fully-coupled steady-state condition...")

        sim_control_SS["fully_coupled"] = True
        primary_variables, _, converged, _ = fully_coupled_time_integration(
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
        if not converged:
            raise RuntimeError("Failed to converge during assignment of initial conditions.")
    else:
        # viscosity is a constant.
        # This reduces the problem to a one-way p -> T, just like in
        # our linearized extrapolation.
        primary_variables, _, _, _ = lin_extrap_time_integration(
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
    primary_variables["p"][:, :] = _broadcast_to_history(p)
    primary_variables["T"][:, :] = _broadcast_to_history(T)
    return primary_variables


# ------------------ Solve for initial conditions
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
        print("Assigning initial conditions...")

    assign_type = config_ini["assignment"]
    supported_types = ["constant", "gradient", "BC_compatible"]

    if assign_type == "constant":
        primary_variables = _assign_ic_constant(config_ini, primary_variables)
    elif assign_type == "gradient":
        primary_variables = _assign_ic_gradient(config_ini, mesh, primary_variables)
    elif assign_type == "BC_compatible":
        primary_variables = _assign_ic_bc_compatible(
            sim_control,
            mesh,
            material,
            matrices_linear,
            primary_variables,
            boundary_conditions,
            n,
            verbose,
        )
    else:
        raise ValueError(
            f"Unknown initial condition assignment type: '{assign_type}'\n"
            f"Supported types: {', '.join(supported_types)}\n"
        )

    # use initial conditions as the reference temperature for buoyancy
    material["T_0"] = primary_variables["T"][:, 0]  # initial condition reference temperature

    return primary_variables, material


# ------------------------------------ Time integration decider
def _perform_time_integration(
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
    """Choose between time integration types."""
    solver_type = sim_control["solver_type"]
    supported_types = ["linearized", "fully_coupled"]

    if solver_type == "linearized":
        primary_variables, dt, converged, timestep_error = lin_extrap_time_integration(
            sim_control,
            mesh,
            material,
            matrices_linear,
            primary_variables,
            boundary_conditions,
            dt,
            n,
            verbose,
        )
    elif solver_type == "fully_coupled":
        sim_control["fully_coupled"] = True
        primary_variables, dt, converged, timestep_error = fully_coupled_time_integration(
            sim_control,
            mesh,
            material,
            matrices_linear,
            primary_variables,
            boundary_conditions,
            dt,
            n,
            verbose,
        )
    else:
        raise ValueError(
            f"Unknown solver type: '{solver_type}'\nSupported types: {', '.join(supported_types)}\n"
        )

    return primary_variables, dt, converged, timestep_error


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

    if verbose:
        print("RAFT - RESERVOIR ANALYSIS FOR TANGO \n")
        print("Beginning simulation " + config_name + "\n")

    # ----- Load config and mesh
    config, mesh = preprocessing(config_name, config_dir, verbose)
    material = config["material"]
    controls = config["simulation"]
    BC_config = config["boundary_conditions"]

    # ----- Initialize simulation parameters

    ## Numerical controls
    n = 0  # Time step counter
    t = controls["t_start"]
    t_end = controls["t_end"]
    # initialize timesteps and store 3 timesteps (necessary for higher order)
    # dt = [dt_n, dt_(n-1), dt_(n-2)]
    dt = controls["timestep"] * np.ones(3)
    # Gather any important time points as specified in the boundary conditions
    critical_times = get_critical_times(config["boundary_conditions"], t_end)

    ## Initialize vectors
    primary_variables = {}  # storage structure for all primary variables and results
    p_dofs = np.arange(0, mesh["nn_total"])
    T_dofs = np.arange(0, mesh["nn_total"])
    primary_variables["p_dofs"] = p_dofs
    primary_variables["T_dofs"] = T_dofs
    # Vectors of primary DoFs - ndof x 4
    # [t_n, t_(n-1), t_(n-2), ...]
    primary_variables["p"] = np.ones((len(p_dofs), 4))
    primary_variables["T"] = 300 * np.ones((len(T_dofs), 4))

    # Linear Matrix Assembly
    #   note for future use: If a no-flow boundary is desired in the direction of buoyancy,
    #                        there should be an edge integral perform to remove its contribution
    #                        along the upper surfaces.
    if verbose:
        print("     Assembling linear matrices...")
    matrices_linear, material, mesh = integrate_linear_matrices(mesh, material, controls)

    # --- Initial Conditions
    # Update Boundary Conditions for initial step
    boundary_conditions = parsebcs(BC_config, mesh, primary_variables, t, controls)
    q = np.zeros((mesh["nn_total"], 1))
    primary_variables["q"] = q

    [primary_variables, material] = _assign_initial_conditions(
        config["initial_conditions"],
        controls,
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
        boundary_conditions = parsebcs(BC_config, mesh, primary_variables, t, controls)

        # Perform time integration
        primary_variables, dt, converged, timestep_error = _perform_time_integration(
            controls,
            mesh,
            material,
            matrices_linear,
            primary_variables,
            boundary_conditions,
            dt,
            n,
            verbose,
        )

        # Check error and update time step
        t, dt, converged = _update_time(
            controls["solver_type"], t, dt, controls, critical_times, converged, timestep_error, verbose
        )

        # Initialize next timestep
        # use roll to update primary variables, column 0 gets overwritten in time integration
        if converged:
            if verbose:
                print("     Postprocessing...")
            postprocessing(n, mesh, config, material, primary_variables, t, verbose)
            n += 1
            primary_variables["p"][:, [3, 2, 1]] = primary_variables["p"][:, [2, 1, 0]]
            primary_variables["T"][:, [3, 2, 1]] = primary_variables["T"][:, [2, 1, 0]]

        # Check end condition
        if t > t_end:
            if verbose:
                print("Simulation complete. Output files are stored in: \n" + config["output_directory_full"])
            break

    return primary_variables, mesh


if __name__ == "__main__":
    main()
