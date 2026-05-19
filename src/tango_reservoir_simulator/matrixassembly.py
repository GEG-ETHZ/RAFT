"""Assembles global linear FEM matrices."""

from typing import Any

import numpy as np
from scipy.sparse import csc_matrix, lil_matrix
from shapefunctions import getBmatrix, getNmatrix, getquadrature


def _add_block(A: lil_matrix, rows, cols, block: np.ndarray):
    """Scatter dense element matrix into sparse global matrix."""
    rows = np.asarray(rows).ravel()
    cols = np.asarray(cols).ravel()
    for li, gi in enumerate(rows):
        for lj, gj in enumerate(cols):
            A[gi, gj] += block[li, lj]


def integrate_linear_matrices(mesh: dict[str, Any], mat: dict[str, Any]) -> dict[str, Any]:
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
    g[nsd - 1] = -9.81

    # ----- Pre-allocate sparse matrices (use lil for efficient assembly)
    Kpp = lil_matrix((nn, nn))  # Pressure conductance matrix
    KM = lil_matrix((nn, nn))  # Pressure storage matrix
    KpT = lil_matrix((nn, nn))  # Pressurization matrix
    C = lil_matrix((nn, nn))  # Thermal capacity matrix
    KTT = lil_matrix((nn, nn))  # Thermal conductance matrix
    KbT = lil_matrix((nn, nn))  # Pressure buoyancy matrix - body
    L_char = np.zeros((ne, 1))  # Characteristic element lengths
    V = 0.0  # Total volume

    # ----- Quadrature points
    nq = mesh["n_quadrature"]
    Q, W = getquadrature(nq, nsd)
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
        Ve = 0

        for q in range(nq):
            xi = Q[q, :]
            Wi = W[q]

            B, Je = getBmatrix(xi, nodes[enodes, :])
            N = getNmatrix(xi)

            wJe = Wi * np.linalg.det(Je)  # Weight * Jacobian, used in every integrand

            KMe += N.T @ (comp_f * N) * wJe
            Kppe += B.T @ (kf / mu * B) * wJe
            KpTe += N.T @ (alpha_e * N) * wJe
            KbTe += N.T @ (kf / mu * alpha_f * rho_f * g.T) @ B * wJe
            Ce += N.T @ (m_e * N) * wJe
            KTTe += B.T @ (lambda_e * B) * wJe
            Ve += wJe

        # ----- Assemble element contributions into global matrices
        # Using index arrays for lil_matrix:
        _add_block(Kpp, enodes, enodes, Kppe)
        _add_block(KM, enodes, enodes, KMe)
        _add_block(KpT, enodes, enodes, KpTe)
        _add_block(KbT, enodes, enodes, KbTe)
        _add_block(C, enodes, enodes, Ce)
        _add_block(KTT, enodes, enodes, KTTe)
        V += Ve
        L_char[i] = Ve ** (1 / 3)

    # ----- Convert to CSC for efficient arithmetic
    mesh["L_char"] = L_char
    return (
        {
            "Kpp": csc_matrix(Kpp),
            "KM": csc_matrix(KM),
            "KpT": csc_matrix(KpT),
            "C": csc_matrix(C),
            "KTT": csc_matrix(KTT),
            "KbT": csc_matrix(KbT),
        },
        mat,
        mesh,
    )


def integrate_nonlinear_matrices(
    mesh: dict[str, Any],
    mat: dict[str, Any],
    primary_variables: dict[str, Any],
) -> dict[str, Any]:
    """Integrate non-linear matrices inside time integration loop."""
    p = primary_variables["p"][:, 0]
    T = primary_variables["T"][:, 0]  # noqa: F841

    # ----- material properties
    kf = mat["permeability"]
    mu = mat["fluid"]["viscosity"]
    cp_f = mat["fluid"]["specific_heat"]
    rho_f = mat["fluid"]["density"]
    lambda_e = mat["therm_cond_eff"]

    advec_conduct = cp_f * rho_f * kf / mu  # advection conductivity
    # advection velocity = advection_conduct * pressure gradient

    # ----- mesh data
    nn = mesh["nn"]
    ne = mesh["ne"]
    nne = mesh["nne"]
    nsd = mesh["nsd"]
    conn = mesh["conn"]  # (ne, nne)  - connectivity matrix
    nodes = mesh["nodes"]  # (nn,  nsd) - nodal degrees of freedom

    # ----- Pre-allocate sparse matrices (use lil for efficient assembly)
    KATT = lil_matrix((nn, nn))  # Temperature advection tangent matrix wrt temperature
    K_stab_TT = lil_matrix((nn, nn))  # Advection stabilization wrt temperature

    # These following two matrices are not used in the linearized extrapolation.
    # They are the derivatives w.r.t pressure.
    # They are commented out as they are only needed in a fully-coupled solver.
    # potential fully-coupled version in the future.

    # K_stab_TP = lil_matrix((nn, nn))  # Advection stabilization wrt pressure
    # KATP = lil_matrix((nn, nn))  # Temperature advection tangent matrix wrt pressure

    # ----- Quadrature points
    nq = mesh["n_quadrature"]
    Q, W = getquadrature(nq, nsd)
    nq = len(W)

    # ----- Matrix integration element loop
    for i in range(ne):
        enodes = conn[i, :]  # local element nodes
        pe = p[enodes].reshape(-1, 1)  # element nodal pressures
        # Te = T[enodes].reshape(-1, 1)  # element nodal temperatures
        enodes_coords = nodes[enodes, :]
        L_vec = (enodes_coords.max(axis=0) - enodes_coords.min(axis=0)).reshape(-1, 1)  # element lengths

        # Element matrices
        KATTe = np.zeros((nne, nne))
        K_stab_TTe = np.zeros((nne, nne))
        # KATPe = np.zeros((nne, nne))
        # K_stab_TPe = np.zeros((nne, nne))

        for q in range(nq):
            xi = Q[q, :]
            Wi = W[q]

            B, Je = getBmatrix(xi, nodes[enodes, :])
            N = getNmatrix(xi)

            # gradients
            gradp = B @ pe
            # gradT = B @ Te

            # fluxes
            flux = -advec_conduct * gradp
            flux_mag = np.linalg.norm(flux)

            # SUPG Stabilization
            if flux_mag == 0:
                tau = 0
            else:
                # Effective length weighted by fluxes
                h_eff = flux_mag / np.linalg.norm(flux / L_vec)
                # Pecelt number
                Pe = h_eff * flux_mag / lambda_e
                # SUPG stabilization coefficient (Shakib, Hughes, & Zohan (1991))
                tau = h_eff / (2 * flux_mag) * (1 + 9 / Pe**2) ** (-0.5)

            # Resdiual of the discretized form, dropping 2nd order terms
            # Rth = flux.T @ gradT
            # 2nd order terms are usually negligible, but may be necessary for higher order elements.
            # - lambda_e * G * Te
            # - m_e * cv1 * N * Te
            # G = d^2/dx^2 ( N )

            # Integrate
            wJe = Wi * np.linalg.det(Je)  # Weight * Jacobian, used in every integrand
            KATTe += N.T @ (flux.T @ B) * wJe
            K_stab_TTe += B.T * tau @ flux @ (flux.T @ B) * wJe
            # with 2nd order terms
            # B.T * tau * advec_conduct @ gradp * ( flux.T*B  - lambda_e * G - m_e * cv1 * N ) * wJe

            # Derivatives wrt pressure
            # KATPe += N.T * advec_conduct @ gradT.T @ B * wJe
            # K_stab_TPe += B.T * tau * advec_conduct * (Rth + advec_conduct * gradp.T @ gradT) @ B * wJe

        # ----- Assemble element contributions into global matrices
        # Using index arrays for lil_matrix:
        _add_block(KATT, enodes, enodes, KATTe)
        _add_block(K_stab_TT, enodes, enodes, K_stab_TTe)
        # _add_block(KATP, enodes, enodes, KATPe)
        # _add_block(K_stab_TP, enodes, enodes, K_stab_TPe)

    # ----- Convert to CSC for efficient arithmetic
    return {
        "KATT": csc_matrix(KATT),
        "K_stab_TT": csc_matrix(K_stab_TT),
        # "KATP": csc_matrix(KATP),
        # "K_stab_TP": csc_matrix(K_stab_TP),
    }
