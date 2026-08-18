"""Subsurface simulator - main time integration loop."""

import argparse
import datetime
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .logging_utils import RunLogger
from .matrixassembly import integrate_linear_matrices
from .parseboundaryconditions import evaluateBCfunctions, parsebcs
from .postprocessing import export_domain_properties, postprocessing, prepare_output_directory
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
    nsd = np.size(nodes, 1)
    s = s[0:nsd]

    x = np.dot(nodes, s)
    x_min = np.min(x)
    y = y0 + slope * (x - x_min)

    return np.asarray(y, dtype=np.float64)


def _broadcast_to_history(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Broadcast a per-node scalar field into the 4-column time-history layout."""
    return np.tile(values.reshape(-1, 1), (1, 4))


def _build_steady_state_controls(sim_control: dict[str, Any]) -> dict[str, Any]:
    """Build a simulation-control dict for the steady-state initial condition solve.

    Pressurization is neglected since it is time-dependent.
    """
    sim_control_SS = sim_control.copy()
    sim_control_SS["pressurization_on"] = 0
    sim_control_SS["steady_pressure"] = 1
    sim_control_SS["steady_temperature"] = 1
    sim_control_SS["fully_coupled"] = False
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
        mesh["nodes_total"], config_ini["T_ini"], config_ini["T_grad"], config_ini["grad_direction_T"]
    )
    p = _assign_initial_gradient(
        mesh["nodes_total"], config_ini["p_ini"], config_ini["p_grad"], config_ini["grad_direction_p"]
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
    # Initialized reference values -> set as current values and neglect
    # strains when establishing initial conditions
    material["T_0"] = primary_variables["T"][:, 0]
    material["p_0"] = primary_variables["p"][:, 0]

    if sim_control_SS["viscosity_nonconstant"]:
        # viscosity is a function of temperature, fully coupled steady-state
        # solution is required.
        if verbose:
            print("   Non-linear properties present, using iterative sequential scheme to estimate fields...")

        # Use one-pass linear to generate approximate solution
        primary_variables_ini = primary_variables.copy()
        tries = [2, 6, 10]
        for num_passes in tries:
            primary_variables = primary_variables_ini.copy()
            for i in range(num_passes):
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
                # For debugging
                # postprocessing(
                #     n + i, mesh, sim_control, material, boundary_conditions, primary_variables, 0, verbose
                # )  # noqa
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
            if converged:
                break
            else:
                print(
                    "   Unable to find compatible boundary condition, trying again with more sequential iterations."  # noqa
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

    material["p_0"] = primary_variables["p"][:, 0].copy()
    material["T_0"] = primary_variables["T"][:, 0].copy()
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
        sim_control["fully_coupled"] = False
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


def _initialize_WB_checks(BC_config: dict[str, Any]) -> dict[str, Any]:
    """Initialize current and previous flag checks for each wellbore group."""
    for group in BC_config["wellbore"]:
        # Set flags
        group["curr_BC"] = "flux"
        group["prev_BC"] = "flux"
        # Initialize constraint type
        group["pres_constraint"] = "max"
        # Copy existing constraints
        group["values_flux"] = group["values"]
        group["function_flux"] = group["function"]
    return BC_config


def _update_WB_flags(BC_config: dict[str, Any]) -> dict[str, Any]:
    """Update wellbore flags from previous timestep."""
    for group in BC_config["wellbore"]:
        group["prev_BC"] = group["curr_BC"]

    return BC_config


def _announce_WB_change(group: dict[str, Any], constraint: str, verbose: bool) -> None:
    """Print WB constraint violation."""
    if verbose:
        tag = group.get("tag")
        if tag is None:
            tag = "(untagged)"

        announcement = f"  {constraint} constraint exceeded in wellbore '{tag}'. Switching constraint."
        print(f"\033[96m {announcement} \033[00m")


def _check_BC_constraints(
    BC_config: dict[str, Any], primary_variables: dict[str, Any], t: float, WB_on: int, verbose: bool
) -> tuple[bool, dict[str, Any]]:
    """Check whether the BC constraints are satisfied, and update the flags accordingly."""
    if not WB_on:
        return True, BC_config

    constraints_satisfied = True
    p = primary_variables["p"][:, 0]
    q = primary_variables["q"][:, 0]

    WB_config = BC_config["wellbore"]
    n_WB = len(WB_config)
    for indx in range(n_WB):
        group = WB_config[indx]
        curr_BC = group["curr_BC"]
        prev_BC = group["prev_BC"]
        # Only 1 swap allowed per timestep to prevent oscillations
        if curr_BC != prev_BC:
            continue

        if curr_BC == "flux":
            p_local = p[group["nodes"]]
            p_max = group["constraints"]["maximum_pressure"]
            p_min = group["constraints"]["minimum_pressure"]
            # Check max pressure
            if np.any(p_local >= p_max):
                WB_config[indx]["curr_BC"] = "pressure"
                WB_config[indx]["pres_constraint"] = "max"
                constraints_satisfied = False
                _announce_WB_change(group, "Max pressure", verbose)
                break
            # Check min pressure
            if np.any(p_local <= p_min):
                WB_config[indx]["curr_BC"] = "pressure"
                WB_config[indx]["pres_constraint"] = "min"
                constraints_satisfied = False
                _announce_WB_change(group, "Min pressure", verbose)
                break

        if curr_BC == "pressure":
            q_local = q[group["nodes"]]
            # Determine flux constraint
            group_local = group.copy()
            group_local["function"] = group_local["function_flux"]
            group_local["values"] = group_local["values_flux"]
            group_local = evaluateBCfunctions(group_local, t, np.empty((0, 0)))
            q_goal = group_local["BCvalues"]

            if "flux_weight_matrix" in group:
                weight_matrix = group["flux_weight_matrix"]
                q_goal = np.sum(weight_matrix @ np.asanyarray(q_goal))

            # Check against current fluxes. If exceeded, switch back over.
            if np.any(np.abs(q_local) >= np.abs(q_goal)):
                WB_config[indx]["curr_BC"] = "flux"
                constraints_satisfied = False
                _announce_WB_change(group, "Fluid flux", verbose)
                break

    return constraints_satisfied, BC_config


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
        material = config["material"]
        BC_config = config["boundary_conditions"]
        controls = config["simulation"]
        # Copy config name and directory to controls
        controls["config_name"] = config["config_name"]
        controls["config_dir"] = config["config_dir"]

        if controls["output_log"]:
            run_logger.attach_file(prepare_output_directory(controls))

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
        primary_variables["p"] = 1e5 * np.ones((len(p_dofs), 4))
        primary_variables["T"] = 300 * np.ones((len(T_dofs), 4))

        ## Wellbore boundary conditions
        WB_on = controls["wellbore_check"]
        if WB_on:
            BC_config = _initialize_WB_checks(BC_config)

        # Linear Matrix Assembly
        #   note for future use: If a no-flow boundary is desired in the direction of buoyancy,
        #                        there should be an edge integral perform to remove its contribution
        #                        along the upper surfaces.
        if verbose:
            print("     Exporting domain material properties...")
        export_domain_properties(mesh, controls, material, BC_config)

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
        postprocessing(n, mesh, controls, material, BC_config, primary_variables, t, verbose)
        n += 1

        # ---- Time integration
        while True:
            if verbose:
                print(f"Timestep {n}, t = {t:.2e}, {(100 * t / t_end):.2f} % complete")

            while True:
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

                # Check BC constraints satisfied
                BC_satisfied, BC_config = _check_BC_constraints(
                    BC_config, primary_variables, t, WB_on, verbose
                )
                if BC_satisfied:
                    break

            # Check error and update time step
            t0 = t
            t, dt, converged = _update_time(
                controls["solver_type"], t, dt, controls, critical_times, converged, timestep_error, verbose
            )

            # Initialize next timestep
            if converged:
                if verbose:
                    print("     Postprocessing...")
                postprocessing(n, mesh, controls, material, BC_config, primary_variables, t0, verbose)
                n += 1
                # rollover update primary variables, column 0 gets overwritten in time integration
                primary_variables["p"][:, [3, 2, 1]] = primary_variables["p"][:, [2, 1, 0]]
                primary_variables["T"][:, [3, 2, 1]] = primary_variables["T"][:, [2, 1, 0]]
                # update any WB boundary condition flags
                if WB_on:
                    BC_config = _update_WB_flags(BC_config)

            # Check end condition
            if t > t_end:
                if verbose:
                    print(
                        "Simulation complete. Output files are stored in: \n"
                        + controls["output_directory_full"]
                    )
                break

        return primary_variables, mesh


if __name__ == "__main__":
    main()
