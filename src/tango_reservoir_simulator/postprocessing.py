"""Postprocessing of FEM file."""

import numpy as np
from shapefunctions import getBmatrix


def compute_fluid_flux(mesh, mat, p):
    """Computes fluid flux at each node using nodal averaging.

    Returns nsd x nn array of fluxes at each node.
    """
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
    mu = mat["viscosity"]

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
