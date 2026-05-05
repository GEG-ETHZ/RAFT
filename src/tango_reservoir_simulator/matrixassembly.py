"""Assembles global linear FEM matrices."""

import numpy as np
from quadrature import getquadrature
from scipy.sparse import csc_matrix, lil_matrix
from shapefunctions import getBmatrix, getNmatrix


def _add_block(A: lil_matrix, rows, cols, block: np.ndarray):  # noqa: N803
    """Scatter dense element matrix into sparse global matrix."""
    rows = np.asarray(rows).ravel()
    cols = np.asarray(cols).ravel()
    for li, gi in enumerate(rows):
        for lj, gj in enumerate(cols):
            A[gi, gj] += block[li, lj]


def integratelinearmatices(mesh: dict, mat: dict) -> dict:
    """Integrate linear matrices outside of time integration loop."""
    # ----- material properties
    kf = mat["permeability"]
    mu = mat["fluid"]["viscosity"]
    alpha_s = mat["solid"]["therm_expansion"]
    alpha_f = mat["fluid"]["therm_expansion"]
    rho_f = mat["fluid"]["density"]
    phi = mat["porosity"]
    biot = mat["biot_coeff"]
    Kf = mat["fluid"]["bulk_mod"]
    Ks = mat["solid"]["bulk_mod"]

    # Derived properties (computed here, not stored in YAML)
    M = Kf * Ks / (phi * Ks + (biot - phi) * Kf)  # storage modulus
    comp_f = 1 / M  # compressibility

    # effective expansion modulus
    alpha_e = biot * alpha_s + phi * (alpha_f - alpha_s)

    # effective conductivity
    lambda_e = phi * mat["fluid"]["therm_cond"] + (1 - phi) * mat["solid"]["therm_cond"]

    # effective volumetric heat capacity
    m_e = (
        phi * mat["fluid"]["specific_heat"] * mat["fluid"]["density"]
        + (1 - phi) * mat["solid"]["specific_heat"] * mat["solid"]["density"]
    )

    # store derived properties for future use
    mat["storage_mod"] = M
    mat["therm_cond_eff"] = lambda_e
    mat["therm_expansion_eff"] = alpha_e
    mat["volumetric_heat_capacity_eff"] = m_e

    # ----- mesh data
    nn = mesh["nn"]
    ne = mesh["ne"]
    nne = mesh["nne"]
    nsd = mesh["nsd"]
    conn = mesh["conn"]  # (ne, nne)  - connectivity matrix
    nodes = mesh["nodes"]  # (nn,  nsd) - nodal degrees of freedom

    # ----- gravity vector
    g = np.zeros((nsd, 1))
    g[nsd] = -9.81

    # ----- Pre-allocate sparse matrices (use lil for efficient assembly)
    Kpp = lil_matrix((nn, nn))  # Pressure conductance matrix
    KM = lil_matrix((nn, nn))  # Pressure storage matrix
    KpT = lil_matrix((nn, nn))  # Pressurization matrix
    C = lil_matrix((nn, nn))  # Thermal capacity matrix
    KTT = lil_matrix((nn, nn))  # Thermal conductance matrix
    KbT = lil_matrix((nn, nn))  # Pressure buoyancy matrix - body

    # ----- Quadrature points
    W, Q = getquadrature(2, nsd)  # body
    nq = len(W)

    # ----- Matrix integration element loop
    for i in range(ne):
        enodes = conn[i, :]  # local element nodes

        # Element matrices
        Kppe = np.zeros((nne, nne))
        KMe = np.zeros((nne, nne))
        KpTe = np.zeros((nne, nne))
        KbTe = np.zeros((nne, nne))
        Ce = np.zeros((nne, nne))
        KTTe = np.zeros((nne, nne))

        for q in range(nq):
            xi = Q[q, :]
            Wi = W[q]

            B, Je = getBmatrix(xi, nodes[enodes, :])
            N = getNmatrix(xi, nodes[enodes, :])

            wJe = Wi * np.linalg.det(Je)

            KMe += N.T @ (comp_f * N) * wJe
            Kppe += B.T @ (kf / mu * B) * wJe
            KpTe += N.T @ (alpha_e * N) * wJe
            KbTe += B.T @ (kf / mu * alpha_f * rho_f * g.T) @ N * wJe
            Ce += N.T @ (m_e * N) * wJe
            KTTe += B.T @ (lambda_e * B) * wJe

        # ----- Assemble element contributions into global matrices
        # Using index arrays for lil_matrix:
        _add_block(Kpp, enodes, enodes, Kppe)
        _add_block(KM, enodes, enodes, KMe)
        _add_block(KpT, enodes, enodes, KpTe)
        _add_block(KbT, enodes, enodes, KbTe)
        _add_block(C, enodes, enodes, Ce)
        _add_block(KTT, enodes, enodes, KTTe)

    # ----- Convert to CSC for efficient arithmetic
    return {
        "Kpp": csc_matrix(Kpp),
        "KM": csc_matrix(KM),
        "KpT": csc_matrix(KpT),
        "C": csc_matrix(C),
        "KTT": csc_matrix(KTT),
        "KbT": csc_matrix(KbT),
    }, mat
