"""Postprocessing of FEM file."""

import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .writevtk import write_mesh2vtk

# Default output directory: /main_folder/outputs/
# (two levels up from /main_folder/src/postprocessing.py)
_DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs"


def compute_fluid_flux(
    mesh: dict[str, Any], mat: dict[str, Any], p: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Computes fluid flux at each node using nodal averaging.

    Returns nsd x nn array of fluxes at each node.
    """
    from .shapefunctions import getBmatrix

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


def postprocessing(
    n: int,
    mesh: dict[str, Any],
    config: dict[str, Any],
    primary_variables: dict[str, Any],
    verbose: bool,
):
    """Postprocessing of FEM data.

    Calculates fluxes, stresses,and Peclet numbers, then exports the results to a VTK file.
    """
    # --- Path Management
    base_path = config["simulation"].get("output_directory", _DEFAULT_OUTPUT_DIR)
    config_stem = Path(config["config_name"]).stem
    io_path = os.path.join(base_path, config_stem)
    config["output_directory_full"] = io_path

    if not os.path.exists(io_path):
        os.makedirs(io_path)

    name = f"Solution.vtk.{n}"
    filename = os.path.join(io_path, name)

    # --- Output, frequency is controlled by user user input
    if (n % config["simulation"]["output_frequency"]) == 0 and config["simulation"]["output_vtk"]:
        if verbose:
            print("     Writing to output file...")
        # ----- Extract primary variables
        p = primary_variables["p"][:, 0]
        T = primary_variables["T"][:, 0]

        # --- Calculate fluxes
        # Assumes compute_fluid_flux is ported from ComputeFluidFlux.m
        flux = compute_fluid_flux(mesh, config["material"], p)

        # --- Sanitize and collect scalar data
        def sanitize(data, threshold=1e-9):
            data[np.abs(data) < threshold] = threshold
            return data

        # Collect scalar data in an dictionary for export
        scalardata = [
            {"name": "T", "data": sanitize(T.copy())},
            {"name": "p", "data": sanitize(p.copy())},
        ]

        # --- Vector data
        vectordata = [{"name": "flux", "data": sanitize(flux.T)}]

        # ----- Export to vtk
        write_mesh2vtk(filename, "TANGO", mesh["nodes"], mesh["conn"], scalardata, vectordata)

    return config
