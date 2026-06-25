"""Assembles global linear FEM matrices."""

import math
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.sparse import csc_matrix

from .shapefunctions import getBmatrix, getNmatrix, getquadrature


def _calc_detJe(Je: npt.NDArray[np.float64], nsd: int):
    # set Jacobian formula based on space dimensions
    if nsd == 2:
        return Je[0, 0] * Je[1, 1] - Je[0, 1] * Je[1, 0]
    else:
        return (
            Je[0, 0] * (Je[1, 1] * Je[2, 2] - Je[1, 2] * Je[2, 1])
            - Je[0, 1] * (Je[1, 0] * Je[2, 2] - Je[1, 2] * Je[2, 0])
            + Je[0, 2] * (Je[1, 0] * Je[2, 1] - Je[1, 1] * Je[2, 0])
        )


def _make_sparse(
    vals: npt.NDArray[np.float64],
    row_vec: npt.NDArray[np.int32],
    col_vec: npt.NDArray[np.int32],
    nn: int,
) -> csc_matrix:
    return csc_matrix((vals, (row_vec, col_vec)), shape=(nn, nn))


def rotation_matrices(rot_x, rot_y, rot_z):
    """Return individual rotation matrices (angles in degrees)."""
    rot_x = np.radians(rot_x)
    rot_y = np.radians(rot_y)
    rot_z = np.radians(rot_z)

    cx, sx = np.cos(rot_x), np.sin(rot_x)
    cy, sy = np.cos(rot_y), np.sin(rot_y)
    cz, sz = np.cos(rot_z), np.sin(rot_z)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    return Rx, Ry, Rz


# Combined rotation (if needed)


def integrate_linear_matrices(
    mesh: dict[str, Any], mat: dict[str, Any]
) -> tuple[dict[str, csc_matrix[Any]], dict[str, Any], dict[str, Any]]:
    """Integrate linear matrices outside of time integration loop."""
    # ----- material properties
    mu = mat["fluid"]["viscosity"]
    alpha_s = mat["solid"]["therm_expansion"]
    alpha_f = mat["fluid"]["therm_expansion"]
    rho_f = mat["fluid"]["density"]
    phi = mat["porosity"]
    biot = mat["biot_coeff"]
    Kf = mat["fluid"]["bulk_mod"]
    Ks = mat["solid"]["bulk_mod"]

    # physical properties
    nsd = mesh["nsd"]
    Rx, Ry, Rz = rotation_matrices(mat["rotation_x"], mat["rotation_y"], mat["rotation_z"])
    R = Rx @ Ry @ Rz  # general rotation matrix

    # turn permeability into list if homogeneous
    kf = mat["permeability"]
    kf_list = [kf] * nsd if not isinstance(kf, list) else kf
    kf = R @ np.diag(kf_list) @ R.T

    # Derived properties (computed here, not stored in YAML)
    M = Kf * Ks / (phi * Ks + (biot - phi) * Kf)  # storage modulus
    comp_f = 1 / M  # compressibility

    # effective expansion modulus
    alpha_e = biot * alpha_s + phi * (alpha_f - alpha_s)

    # effective conductivity
    ks = mat["solid"]["therm_cond"]
    ks_list = [ks] * nsd if not isinstance(ks, list) else ks
    lambda_e = phi * mat["fluid"]["therm_cond"] * np.eye(nsd) + (1 - phi) * R @ np.diag(ks_list) @ R.T

    # effective volumetric heat capacity
    m_e = (
        phi * mat["fluid"]["specific_heat"] * mat["fluid"]["density"]
        + (1 - phi) * mat["solid"]["specific_heat"] * mat["solid"]["density"]
    )

    # store derived properties for future use
    mat["permeability"] = kf_list  # list of kf_x,y,z
    mat["storage_mod"] = M
    mat["therm_cond_eff"] = np.diag(lambda_e)  # list lambda_x,y,z
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
    ng = np.zeros((nsd, 1))
    ng[nsd - 1] = 1.0  # gravity normal
    g = -9.81 * ng
    kf_b = ng.T @ kf @ ng  # kf component in gravity direction

    # ----- Vectorized matrix assembly
    # Assemble lists of rows, columns, and entries and then assemble the sparse
    # matrix all at once.
    # For the scalar pressure and temperature, there are nne * nne entries
    # per element.
    total_entries = ne * nne * nne
    vec = {name: np.zeros(total_entries) for name in ("Kpp", "KM", "KpT", "KbT", "C", "KTT")}
    row_vec = np.empty(total_entries, dtype=np.int32)
    col_vec = np.empty(total_entries, dtype=np.int32)

    # Matrices to assemble:
    # Kpp = Pressure conductance matrix
    # KM =  Pressure storage matrix
    # KpT =  Pressurization matrix
    # C = Thermal capacity matrix
    # KTT =  Thermal conductance matrix
    # KbT = Pressure buoyancy matrix - body

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

            wJe = Wi * _calc_detJe(Je, nsd)  # Weight * Jacobian, used in every integrand

            KMe += N.T @ (comp_f * N) * wJe
            Kppe += B.T @ (kf / mu @ B) * wJe
            KpTe += N.T @ (alpha_e * N) * wJe
            KbTe += N.T @ (kf_b / mu * alpha_f * rho_f @ g.T) @ B * wJe
            Ce += N.T @ (m_e * N) * wJe
            KTTe += B.T @ (lambda_e @ B) * wJe
            Ve += wJe

        # ----- Assemble element contributions into global matrices
        # Build global (row, col) index pairs for this element block once
        sl = slice(i * nne * nne, (i + 1) * nne * nne)
        # enodes repeated/tiled gives all (row,col) combos in row-major order
        row_vec[sl] = np.repeat(enodes, nne)  # [e0,e0,..,e1,e1,..]
        col_vec[sl] = np.tile(enodes, nne)  # [e0,e1,..,e0,e1,..]

        vec["Kpp"][sl] = Kppe.ravel()
        vec["KM"][sl] = KMe.ravel()
        vec["KpT"][sl] = KpTe.ravel()
        vec["KbT"][sl] = KbTe.ravel()
        vec["C"][sl] = Ce.ravel()
        vec["KTT"][sl] = KTTe.ravel()

        V += Ve
        L_char[i] = Ve ** (1 / 3)

    mesh["L_char"] = L_char
    return (
        {name: _make_sparse(vec[name], row_vec, col_vec, nn) for name in vec},
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
    kf = np.diag(mat["permeability"])
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
    total_entries = ne * nne * nne
    vec = {name: np.zeros(total_entries) for name in ("KATT", "K_stab_TT")}
    row_vec = np.empty(total_entries, dtype=np.int32)
    col_vec = np.empty(total_entries, dtype=np.int32)
    # KATT = Temperature advection tangent matrix wrt temperature
    # K_stab_TT = l Advection stabilization wrt temperature

    # These following two matrices are not used in the linearized extrapolation.
    # They are the derivatives w.r.t pressure.
    # They are commented out as they are only needed in a fully-coupled solver.
    # potential fully-coupled version in the future.

    # K_stab_TP = Advection stabilization wrt pressure
    # KATP = Temperature advection tangent matrix wrt pressure

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
            flux = -np.diag(advec_conduct * gradp).reshape(-1, 1)
            flux_mag = math.sqrt(flux[0, 0] ** 2 + flux[1, 0] ** 2 + flux[2, 0] ** 2)

            # SUPG Stabilization
            if flux_mag == 0:
                tau = 0
            else:
                # Effective length weighted by fluxes
                h_eff = flux_mag / math.sqrt(
                    (flux[0, 0] / L_vec[0, 0]) ** 2
                    + (flux[1, 0] / L_vec[1, 0]) ** 2
                    + (flux[2, 0] / L_vec[2, 0]) ** 2
                )
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
            wJe = Wi * _calc_detJe(Je, nsd)  # Weight * Jacobian, used in every integrand
            KATTe += N.T @ (flux.T @ B) * wJe
            K_stab_TTe += B.T * tau @ flux @ (flux.T @ B) * wJe
            # with 2nd order terms
            # B.T * tau * advec_conduct @ gradp * ( flux.T*B  - lambda_e * G - m_e * cv1 * N ) * wJe

            # Derivatives wrt pressure
            # KATPe += N.T * advec_conduct @ gradT.T @ B * wJe
            # K_stab_TPe += B.T * tau * advec_conduct * (Rth + advec_conduct * gradp.T @ gradT) @ B * wJe

        # ----- Assemble element contributions into global matrices
        sl = slice(i * nne * nne, (i + 1) * nne * nne)
        row_vec[sl] = np.repeat(enodes, nne)
        col_vec[sl] = np.tile(enodes, nne)

        vec["KATT"][sl] = KATTe.ravel()
        vec["K_stab_TT"][sl] = K_stab_TTe.ravel()
        # vec["KATP"][sl]      = KATPe.ravel()
        # vec["K_stab_TP"][sl] = K_stab_TPe.ravel()

    # ----- Return and convert to sparse matrix
    return {name: _make_sparse(vec[name], row_vec, col_vec, nn) for name in vec}
