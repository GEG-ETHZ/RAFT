"""Subsurface Simulator."""

from pathlib import Path

import numpy as np
import yaml
from matrixassembly import integratelinearmatices


def load_config(path):
    """Load config file."""
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def load_mesh(config_mesh):
    """Load mesh and define mesh structures."""
    from meshmaker import loadmesh

    script_dir = Path(__file__).parent
    config_path = script_dir.parent / "config" / config_mesh["file"]
    [nodes, conn] = loadmesh(str(config_path), config_mesh["dimensions"])
    mesh = {
        "nodes": nodes,
        "conn": conn,
    }

    # Extract key values
    mesh["nn"] = np.shape(nodes)[0]  # number of nodes
    mesh["ne"] = np.shape(conn)[0]  # number of elements
    mesh["nne"] = np.shape(conn)[1]  # number of nodes per element
    mesh["nsd"] = np.shape(nodes)[1]  # number of space dimensions

    return mesh


def rate_coefficient(dt1, dt2, order):
    """Calculate the rate of change coefficients."""
    # dt1 = t^n - t^(n-1)
    # dt2 = t^(n-1) - t^(n-2)
    cv = np.ones((3, 1))
    if order == 2:
        cv[0] = (2 * dt1 + dt2) / dt1 / (dt1 + dt2)
        cv[1] = -(dt1 + dt2) / dt1 / dt2
        cv[2] = dt1 / dt2 / (dt1 + dt2)
    else:
        cv[0] = 1 / dt1
        cv[1] = -1 / dt1
        cv[2] = 0
    return cv


def main():
    """Main simulation run."""
    # ----- Config file
    config_name = "config_1.yaml"
    config = load_config(config_name)

    # ----- Initialize simulation parameters

    ## Load Mesh
    mesh = load_mesh(config["mesh"])
    ## Initialize vectors
    p_dofs = np.arange(0, mesh["nn"])
    T_dofs = np.arange(0, mesh["nn"])
    p = np.zeros((len(p_dofs), 3))
    T = np.zeros((len(p_dofs), 3))

    ## Numerical controls
    t = config["simulation"]["t_start"]  # noqa: F841
    t_end = config["simulation"]["t_end"]  # noqa: F841
    dt = config["simulation"]["timestep"]
    cv = rate_coefficient(dt, dt, config["simulation"]["rate_order"])  # noqa: F841

    # Linear Matrix Assembly
    matrices_linear = integratelinearmatices(mesh, config["material"])  # noqa: F841

    # --- Initial Conditions
    # assume uniform
    p[p_dofs, :] = config["initial_conditions"]["p_surf"]
    T[T_dofs, :] = config["initial_conditions"]["T_surf"]
    # non-uniform to be implemented in future.

    # ---- Boundary Conditions

    # Force Vector Assembly

    # Matrix Inversion

    # Solve


if __name__ == "__main__":
    main()
