"""Subsurface Simulator."""

from pathlib import Path

import numpy as np
import yaml


def load_config(path):
    """Load config file."""
    with open(path) as f:
        config = yaml.safe_load(f)

    # Derived properties (computed here, not stored in YAML)
    mat = config["material"]
    phi = mat["porosity"]
    biot = mat["biot_coeff"]
    Kf = mat["fluid"]["bulk_mod"]
    Ks = mat["solid"]["bulk_mod"]

    # storage modulus
    M = Kf * Ks / (phi * Ks + (biot - phi) * Kf)
    # effective expansion modulus
    a_e = biot * mat["solid"]["therm_expansion"] + phi * (
        mat["fluid"]["therm_expansion"] - mat["solid"]["therm_expansion"]
    )
    # effective conductivity
    lambda_e = phi * mat["fluid"]["therm_cond"] + (1 - phi) * mat["solid"]["therm_cond"]
    # effective heat capacity
    cp_e = (
        phi * mat["fluid"]["specific_heat"] * mat["fluid"]["density"]
        + (1 - phi) * mat["solid"]["specific_heat"] * mat["solid"]["density"]
    )

    config["material"]["storage_mod"] = M
    config["material"]["therm_cond_eff"] = lambda_e
    config["material"]["therm_expansion_eff"] = a_e
    config["material"]["volumetric_heat_capacity_eff"] = cp_e

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

    return mesh


def rate_coefficient(dt1, dt2, order):
    """Calculate the rate of change coefficients."""
    # dt1 = t^n - t^(n-1)
    # dt2 = t^(n-1) - t^(n-2)
    cv = np.arange(0, 2)
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
    primary_vars = np.zeros((2 * mesh["nn"], 3))
    primary_vars[p_dofs, :] = config["initial_conditions"]["p_surf"]
    primary_vars[T_dofs, :] = config["initial_conditions"]["T_surf"]

    ## Numerical controls
    t = config["simulation"]["t_start"]  # noqa: F841
    t_end = config["simulation"]["t_end"]  # noqa: F841
    dt = config["simulation"]["dt"]
    cv = rate_coefficient(dt, dt, config["simulation"]["rate_order"])  # noqa: F841

    # ---- Boundary Conditions

    # Initial Conditions

    # Matrix Assembly

    # Force Vector Assembly

    # Matrix Inversion

    # Solve


if __name__ == "__main__":
    main()
