"""Subsurface Simulator."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import yaml
from matrixassembly import integrate_linear_matrices, integrate_nonlinear_matrices
from parseboundaryconditions import parsebcs
from postprocessing import postprocessing
from scipy.sparse.linalg import spsolve


def load_config(path):
    """Load config file."""
    with open(path) as f:
        config = yaml.safe_load(f)
    config["config_name"] = path
    return config


def load_mesh(config_mesh):
    """Load mesh and define mesh structures."""
    from meshmaker import loadmesh

    script_dir = Path(__file__).parent
    config_path = script_dir.parent / "config" / config_mesh["file"]
    [nodes, conn] = loadmesh(str(config_path), config_mesh["dimensions"])
    mesh = {
        "nodes": nodes,
        "conn": conn - 1,
    }

    # Extract key values
    mesh["nn"] = np.shape(nodes)[0]  # number of nodes
    mesh["ne"] = np.shape(conn)[0]  # number of elements
    mesh["nne"] = np.shape(conn)[1]  # number of nodes per element
    mesh["nsd"] = np.shape(nodes)[1]  # number of space dimensions

    # Define number of quadrature points
    match (mesh["nsd"], mesh["nne"]):
        case (3, 8):  # B8
            nq = 2
        case (3, 27):  # B27
            nq = 3
        case (2, 4):  # Q4
            nq = 2
        case (2, 9):  # Q9
            nq = 3
    mesh["n_quadrature"] = nq

    return mesh


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


def _update_time(t, dt, t_end):
    """Update the current time and timestep."""
    dt[2] = dt[1]
    dt[1] = dt[0]
    t += dt[0]
    if t > t_end and t != t_end:
        t -= dt[0]
        dt[0] = t_end - t
        t = t_end
    return t, dt


def _lin_extrap_time_integration(
    sim_control: dict,
    mesh: dict,
    material: dict,
    matrices_linear: dict,
    p: npt.NDArray[np.float64],
    T: npt.NDArray[np.float64],
    boundary_conditions: dict,
    dt: npt.NDArray[np.float64],
):
    """Assemble FEM matrices and vectors and solve for solution at next timestep."""
    # extract controls
    rate_order = sim_control["rate_order"]
    adv_on = sim_control["advection_on"]
    prez_on = sim_control["pressurization_on"]
    buoy_on = sim_control["buoyancy_on"]

    # calculate rate coefficients
    cv, cx = rate_coefficients(dt, rate_order)

    # parse boundary conditions
    # dirichlet - fixed
    T_fix = boundary_conditions["T_dirichlet"][:, 0].astype(int)
    T_bar = boundary_conditions["T_dirichlet"][:, 1]
    p_fix = boundary_conditions["p_dirichlet"][:, 0].astype(int)
    p_bar = boundary_conditions["p_dirichlet"][:, 1]
    # neumann - external force vectors
    nn = mesh["nn"]
    F_extT = np.zeros((nn, 1))
    F_extp = np.zeros((nn, 1))
    F_extT[boundary_conditions["T_neumann"][:, 0].astype(int)] = boundary_conditions["T_neumann"][
        :, 1
    ].reshape(-1, 1)
    F_extp[boundary_conditions["p_neumann"][:, 0].astype(int)] = boundary_conditions["p_neumann"][
        :, 1
    ].reshape(-1, 1)

    # Begin the one-pass extrapolated linearization by solving pressure
    # first using an extrapolated pressure
    # One-pass update: solve pressure, then temperature

    T_extrap = cx[0] * T[:, 0] + cx[1] * T[:, 1] + cx[2] * T[:, 2]  # Extrapolated temperature
    T_dot = cv[0] * T_extrap + cv[1] * T[:, 1] + cv[2] * T[:, 2]  # Rate of temperature change

    KM = matrices_linear["KM"]  # Pressure storage matrix
    Kpp = matrices_linear["Kpp"]  # Pressure conductance matrix
    KpT = matrices_linear["KpT"]  # Pressurization matrix
    KbT = matrices_linear["KbT"]  # Buoyancy matrix

    F_pT = prez_on * (-KpT @ T_dot).reshape(-1, 1)  # Pressurization force vector
    F_bT = buoy_on * (-KbT @ (T_extrap - material["T_0"])).reshape(-1, 1)  # Buoyancy force vector

    # Assemble combined matrices
    A = (cv[0] * KM) + Kpp
    b = -KM @ (cv[1] * p[:, 1] + cv[2] * p[:, 2]).reshape(-1, 1) + F_extp + F_bT + F_pT

    nE = p_fix  # essential DoFs
    nF = np.setdiff1d(np.arange(nn), nE)  # free DoFs
    dE = p_bar
    KFF = A[np.ix_(nF, nF)]
    KFE = A[np.ix_(nF, nE)]
    pF = spsolve(KFF, (b[nF] - KFE @ dE.reshape(-1, 1)).ravel())
    p[nE, 0] = dE
    p[nF, 0] = pF.ravel()

    # Update nonlinear matrices with new pressure
    matrices_non_linear = integrate_nonlinear_matrices(mesh, material, p[:, 0], T[:, 0])
    KTA = (matrices_non_linear["KATT"] + matrices_non_linear["K_stab_TT"]) * adv_on  # advection matrix
    KTT = matrices_linear["KTT"]  # thermal conduction matrix
    C = matrices_linear["C"]  # thermal storage matrix

    A = (cv[0] * C) + KTA + KTT
    b = -C @ (cv[1] * T[:, 1] + cv[2] * T[:, 2]).reshape(-1, 1) + F_extT

    nE = T_fix  # essential DoFs
    nF = np.setdiff1d(np.arange(nn), nE)  # free DoFs
    dE = T_bar
    KFF = A[np.ix_(nF, nF)]
    KFE = A[np.ix_(nF, nE)]
    TF = spsolve(KFF, (b[nF] - KFE @ dE.reshape(-1, 1)).ravel())
    T[nE, 0] = dE
    T[nF, 0] = TF.ravel()

    return p, T


def main():
    """Main simulation run."""
    config_name = "config_1.yaml"

    print("GRAFT - GEOTHERMAL RESERVOIR ANALYSIS FOR TANGO \n")
    print("Beginning simulation " + config_name + "\n")

    # ----- Load onfig file
    config = load_config(config_name)
    material = config["material"]

    # ----- Initialize simulation parameters

    ## Load Mesh
    mesh = load_mesh(config["mesh"])
    ## Initialize vectors
    p_dofs = np.arange(0, mesh["nn"])
    T_dofs = np.arange(0, mesh["nn"])
    # Vectors of primary DoFs - ndof x 3
    # [t_n, t_(n-1), t_(n-2)]
    p = np.zeros((len(p_dofs), 3))
    T = np.zeros((len(p_dofs), 3))

    ## Numerical controls
    t = config["simulation"]["t_start"]
    t_end = config["simulation"]["t_end"]
    dt = config["simulation"]["timestep"] * np.ones(3)

    # Linear Matrix Assembly
    #   note for future use: If a no-flow boundary is desired in the direction of buoyancy,
    #                        there should be an edge integral perform to remove its contribution
    #                        along the upper surfaces.
    matrices_linear, material, mesh = integrate_linear_matrices(mesh, material)

    # --- Initial Conditions
    # assume uniform, non-uniform to be implemented in future.
    p[p_dofs, :] = config["initial_conditions"]["p_surf"]
    T[T_dofs, :] = config["initial_conditions"]["T_surf"]
    material["T_0"] = T[:, 1]  # initial condition reference temperature

    # --- Export initial condition
    n = 0  # Time step counter
    postprocessing(n, mesh, config, p[:, 0], T[:, 0])
    n += 1

    # ---- Boundary Conditions
    boundary_conditions = parsebcs(config["boundary_conditions"], mesh, p, T, t)  # noqa: F841

    # ---- Time integration
    while True:
        print(f"Timestep n, t = {t}, {100 * t / t_end} % complete")
        # Perform time integration
        p, T = _lin_extrap_time_integration(
            config["simulation"], mesh, material, matrices_linear, p, T, boundary_conditions, dt
        )

        # Post-processing
        postprocessing(n, mesh, config, p[:, 0], T[:, 0])

        # Update time step
        t, dt = _update_time(t, dt, t_end)
        n += 1
        p = np.roll(p, shift=1, axis=1)
        T = np.roll(T, shift=1, axis=1)

        # Check end condition
        if t >= t_end:
            print(
                "Simulation complete. Output files are stored in: \n"
                + config["simulation"]["output_directory"]
            )
            break

    return p, T


if __name__ == "__main__":
    main()
