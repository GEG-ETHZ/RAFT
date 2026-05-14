"""Postprocessing of FEM file."""

import os

import numpy as np
from writevtk import write_mesh2vtk


def compute_fluid_flux(mesh, mat, p):
    """Computes fluid flux at each node using nodal averaging.

    Returns nsd x nn array of fluxes at each node.
    """
    from shapefunctions import getBmatrix

    nn = mesh["nn"]
    nsd = mesh["nsd"]
    ne = mesh["ne"]
    nne = mesh["nne"]

    conn = mesh["conn"]
    nodes = mesh["nodes"]

    # Initialize output arrays
    q = np.zeros((nsd, nn))
    count = np.zeros(nn)

    # Material properties
    kf = mat["permeability"]
    mu = mat["fluid"]["viscosity"]

    # Local coordinates (xi) based on element type
    if nne == 4:  # Q4 element
        xi = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    elif nne == 9:  # Q9 element
        xi = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    elif nne == 8 or nne == 27:  # B8 element
        xi = np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ]
        )
    elif nne == 27:  # B27 element
        xi = np.array(
            [
                [-1, -1, -1],
                [-1, 1, -1],
                [1, 1, -1],
                [1, -1, -1],
                [-1, -1, 1],
                [-1, 1, 1],
                [1, 1, 1],
                [1, -1, 1],
                [-1, 0, -1],
                [0, 1, -1],
                [1, 0, -1],
                [0, -1, -1],
                [-1, 0, 1],
                [0, 1, 1],
                [1, 0, 1],
                [0, -1, 1],
                [-1, -1, 0],
                [-1, 1, 0],
                [1, 1, 0],
                [1, -1, 0],
                [0, 0, -1],
                [0, 0, 1],
                [-1, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, 0],
            ]
        )
    else:
        raise ValueError(f"Unsupported number of nodes per element: {nne}")

    # Loop over elements
    for e in range(ne):
        enodes = conn[e, :]  # element nodes
        xI = nodes[enodes, :nsd]  # element physical coordinates
        pe = p[enodes]  # element nodal pressures

        # Loop over nodes within the element
        for n in range(nne):
            B, _ = getBmatrix(xi[n, :], xI)

            # Fluid flux calculation
            flux_val = -kf / mu * (B @ pe)
            q[:, enodes[n]] += flux_val

            count[enodes[n]] += 1

    # Nodal averaging of fluxes
    # Avoid division by zero for unused nodes
    nonzero_count = np.where(count == 0, 1, count)

    for n in range(nsd):
        q[n, :] /= nonzero_count

    return q


def postprocessing(n, mesh, config, p, T):  # noqa
    """Postprocessing of FEM data.

    Calculates fluxes, stresses,and Peclet numbers, then exports the results to a VTK file.
    """
    # --- Path Management
    base_path = config["simulation"]["output_directory"]
    io_path = os.path.join(base_path, config["config_name"])

    if not os.path.exists(io_path):
        os.makedirs(io_path)

    name = f"Solution.vtk.{n}"
    filename = os.path.join(io_path, name)

    # --- Calculate Fluxes ---
    # Assumes compute_fluid_flux is ported from ComputeFluidFlux.m
    flux = compute_fluid_flux(mesh, config["material"], p)

    # Calculate Peclet Number
    if mesh["nsd"] == 3:
        q = np.sqrt(flux[0, :] ** 2 + flux[1, :] ** 2 + flux[2, :] ** 2)
    else:
        q = np.sqrt(flux[0, :] ** 2 + flux[1, :] ** 2)

    pe = (config["material"]["fluid"]["specific_heat"] * config["material"]["fluid"]["density"] * q) / config[
        "material"
    ]["therm_cond_eff"]

    # --- Sanitize and Collect Scalar Data ---
    def sanitize(data, threshold=1e-9):
        data[np.abs(data) < threshold] = threshold
        return data

    # Collect scalar data in an dictionary for export
    scalardata = [
        {"name": "T", "data": sanitize(T.copy())},
        {"name": "p", "data": sanitize(p.copy())},
        {"name": "Pe", "data": sanitize(pe)},
    ]

    # --- Vector Data ---
    vectordata = [{"name": "flux", "data": flux.T}]

    # --- Final Export ---
    # Assumes write_mesh_to_vtk is ported from WriteMesh2VTK_v5.m
    write_mesh2vtk(filename, "TANGO", mesh["nodes"], mesh["conn"], scalardata, vectordata)
