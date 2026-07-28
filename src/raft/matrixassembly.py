"""Assembles global linear FEM matrices."""

from __future__ import annotations

import bisect
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.sparse import csc_matrix, hstack, vstack

from .matprops import get_viscosity
from .shapefunctions import GNMATRIX_FUNCS, NMATRIX_FUNCS, getB, getquadrature


def _get_gravity(nsd: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    ng = np.zeros((nsd, 1))
    ng[nsd - 1] = 1.0  # gravity normal
    return -9.81 * ng, ng


def _make_sparse(
    vals: npt.NDArray[np.float64],
    row_vec: npt.NDArray[np.int32],
    col_vec: npt.NDArray[np.int32],
    nn_1: int,
    nn_2: int,
) -> csc_matrix:
    return csc_matrix((vals, (row_vec, col_vec)), shape=(nn_1, nn_2))


def _lump_diagonal(matrix: csc_matrix) -> csc_matrix:
    """Row-sum lump a matrix, returning a diagonal csc matrix of the same shape."""
    diag_vals = np.asarray(matrix.sum(axis=1)).ravel()
    n = matrix.shape[0]
    idx = np.arange(n, dtype=np.int32)
    return csc_matrix((diag_vals, (idx, idx)), shape=(n, n))


def _assemble_global_matrices(
    vec: dict[str, npt.NDArray[np.float64]],
    conn: list[list[int]],
    nn: int,
    nne_list: list[int],
    elem_indx: list[int],
    elem_order: npt.NDArray[np.int32],  # sequence the values were written
) -> dict[str, csc_matrix]:
    """Assemble vector of rows and columns all at once and assemble sparse matrix."""
    # For vectorized sparse matrix assembly, we feed in 3 vectors:
    # vec: each entry of the element-by-element sparse matrices
    # row_vec: each entry indicates the row that the entry in vec belongs to
    # col_vec:each entry indicates the column that the entry in vec belongs to
    #
    # Rather than assemble the row and column vectors element by element,
    # this function computes them in one go.
    #
    # conn is a ragged list, so use elem_indx to convert chunks to arrays
    # nne for every raw element index
    nne_of_elem = np.empty(elem_indx[-1] + 1, dtype=np.int32)
    prev = -1
    for end, nne in zip(elem_indx, nne_list):
        nne_of_elem[prev + 1 : end + 1] = nne
        prev = end

    nne_seq = nne_of_elem[elem_order]

    row_chunks, col_chunks = [], []
    start = 0
    for i in range(1, len(elem_order) + 1):
        if i == len(elem_order) or nne_seq[i] != nne_seq[start]:
            nne = nne_seq[start]
            run = elem_order[start:i]
            conn_run = np.asarray([conn[e] for e in run], dtype=np.int32)
            row_chunks.append(np.repeat(conn_run, nne, axis=1).ravel())
            col_chunks.append(np.tile(conn_run, (1, nne)).ravel())
            start = i

    row_vec = np.concatenate(row_chunks)
    col_vec = np.concatenate(col_chunks)

    return {name: _make_sparse(vec[name], row_vec, col_vec, nn, nn) for name in vec}


def _get_quad_points_and_shape_functions(
    num_quad_base: int,
    nsd: int,
    etype: str,
) -> tuple[npt.NDArray[np.float64], int, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate list of weights, shape functions, and shape function gradients at quadrature points."""
    # ----- Quadrature points
    Q, W = getquadrature(num_quad_base, nsd, etype)
    nq = len(W)

    # ---- Evaluate element type function
    getN = NMATRIX_FUNCS[etype]  # Shape function matrix
    getGN = GNMATRIX_FUNCS[etype]  # Gradients of shape function matrix
    # Evaluate N and GN at the parent coordinates once
    N_q = []
    GN_q = []
    for q in range(nq):
        xi = Q[q, :]
        N_q.append(getN(xi))
        GN_q.append(getGN(xi))

    return W, nq, np.array(N_q), np.asarray(GN_q)


def _initiate_vectors(
    list_of_matrices: list[str], nne: list[int], elem_indx: list[int]
) -> dict[str, npt.NDArray[np.float64]]:
    """Create the dictionary of vectorized matrix entries."""
    # calculate the number of entries
    prev = -1
    total_entries = 0
    for n, idx in zip(nne, elem_indx):
        ne_i = idx - prev
        total_entries += n**2 * ne_i
        prev = idx
    return {name: np.zeros(total_entries) for name in list_of_matrices}


def _store_vectorized_element_matrix(
    vec: dict[str, npt.NDArray[np.float64]],
    element_matrices: dict[str, npt.NDArray[np.float64]],
    start: int,
    stop: int,
) -> None:
    """Ravel element matrices and store them in the element-entry vector."""
    for name, Ee in element_matrices.items():
        vec[name][start:stop] = Ee.ravel()


def _initialize_element_matrices(
    list_of_matrices: list[str], nne_1: int, nne_2: int
) -> dict[str, npt.NDArray[np.float64]]:
    """Create a dict of zero-initialized (nne_1, nne_2) element matrices, keyed by name."""
    return {name: np.zeros((nne_1, nne_2)) for name in list_of_matrices}


def _vectorized_quad_point_integration(
    left_matrix: npt.NDArray[np.float64],
    integrand: npt.NDArray[np.float64] | float,
    right_matrix: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    scalar: bool,
) -> npt.NDArray[np.float64]:
    """Perform vectorized matrix integration over quadrature points.

    Takes in a left matrix that gets transposed, a middle integrand, and a right matrix.
    The left matrix should have dimensions nq x a x nne.
    The right matrix should have dimensions nq x b x nne.
    The integrand should have dimensions a x b or nq x a x b.
    """
    if scalar:
        integrand = np.asarray(integrand, dtype=np.float64)
    left_T = left_matrix.transpose(0, 2, 1)

    # choose between multiply types for scalar vs. non-scalar material properties
    middle = integrand * right_matrix if scalar else integrand @ right_matrix

    K = left_T @ middle * weights.reshape(-1, 1, 1)

    result: npt.NDArray[np.float64] = K.sum(0)
    return result


# Q4 boundary edges are traversed as (fixed_axis, fixed_value, t_sign): the local parameter
# `t` from the edge projection maps to whichever of xi/eta isn't pinned to +-1 on that edge.
_Q4_EDGE_MAP = (
    (1, -1.0, 1.0),  # edge 0-1: eta = -1, xi = +t
    (0, 1.0, 1.0),  # edge 1-2: xi = 1, eta = +t
    (1, 1.0, -1.0),  # edge 2-3: eta = 1, xi = -t
    (0, -1.0, -1.0),  # edge 3-0: xi = -1, eta = -t
)


def _get_parent_coordinates(  # noqa
    etype: str, nodes: npt.NDArray[np.float64], goal_pts: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Determine the parent coordinates of a physical point in an isoparametric element.

    Assumes that the point is on the element boundary.
    """
    parent_coords = np.empty((np.size(nodes, 1), 1, np.size(goal_pts, -1)))
    x_targets = goal_pts[:, 0, 0]
    y_targets = goal_pts[:, 0, 1]

    if etype == "Q4":
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        x_nodes = nodes[:, 0]
        y_nodes = nodes[:, 1]
        tol = 1.0e-7

        for gp in range(np.size(goal_pts, 2)):
            x_target = x_targets[gp]
            y_target = y_targets[gp]
            for idx, (i, j) in enumerate(edges):
                dx = x_nodes[j] - x_nodes[i]
                dy = y_nodes[j] - y_nodes[i]

                # Use the larger projection direction to avoid division by zero
                # on perfectly horizontal or vertical edges.
                if abs(dx) >= abs(dy):
                    t = 2.0 * (x_target - x_nodes[i]) / dx - 1.0
                    # Check if the actual y matches the expected y along this line segment
                    y_expected = y_nodes[i] + 0.5 * (t + 1.0) * dy
                    is_collinear = abs(y_target - y_expected) < tol
                else:
                    t = 2.0 * (y_target - y_nodes[i]) / dy - 1.0
                    # Check if the actual x matches the expected x along this line segment
                    x_expected = x_nodes[i] + 0.5 * (t + 1.0) * dx
                    is_collinear = abs(x_target - x_expected) < tol

                # The point must be collinear AND fall within the segment endpoints [-1, 1]
                if is_collinear and (-1.0 - tol <= t <= 1.0 + tol):
                    t = max(-1.0, min(1.0, t))  # Clean up minor numerical fuzz

                    # Map the 1D parameter 't' back to 2D parent space based on the edge index
                    fixed_axis, fixed_val, t_sign = _Q4_EDGE_MAP[idx]
                    free_val = t_sign * t
                    xi, eta = (fixed_val, free_val) if fixed_axis == 0 else (free_val, fixed_val)
                    break

            parent_coords[gp] = np.array([xi, eta])

    elif etype == "T3":
        # Direct mapping - linear in the target coordinates, so solve for every
        # goal point at once instead of looping.
        x1, x2, x3 = nodes[0, 0], nodes[1, 0], nodes[2, 0]
        y1, y2, y3 = nodes[0, 1], nodes[1, 1], nodes[2, 1]

        # Determinant (2x the physical area of the triangle)
        D = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)

        xi_all = ((y3 - y1) * (x_targets - x1) - (x3 - x1) * (y_targets - y1)) / D
        eta_all = (-(y2 - y1) * (x_targets - x1) + (x2 - x1) * (y_targets - y1)) / D
        for gp in range(np.size(goal_pts, 2)):
            parent_coords[gp] = np.array([xi_all[gp], eta_all[gp]])

    return parent_coords


def _create_fracture_coupling_vectors(
    conn_f: list[list[int]], conn_fm: list[list[int]], conn_m: list[list[int]]
) -> tuple[
    int,
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    int,
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    int,
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """Create the row and column vectors for vectorized fracture coupling matrix assembly."""
    count_vec_ff = 0  # Fracture-fracture on-diagonal counts
    row_list_ff = []
    col_list_ff = []

    count_vec_fm = 0  # Fracture-matrix off-diagonal counts
    row_list_fm = []
    col_list_fm = []

    count_vec_mm = 0  # Matrix-matrix on-diagonal counts
    row_list_mm = []
    col_list_mm = []

    ne_f = len(conn_f)
    for e_f in range(ne_f):
        e_nodes_f = conn_f[e_f]
        nne_f = len(e_nodes_f)
        for e_m in conn_fm[e_f]:
            e_nodes_m = conn_m[e_m]
            nne_m = len(e_nodes_m)

            # ff
            row_list_ff.append(
                np.repeat(np.array(e_nodes_f, dtype=np.int32).reshape(-1, 1), nne_f, axis=1).ravel()
            )
            col_list_ff.append(np.tile(np.array(e_nodes_f, dtype=np.int32), (1, nne_f)).ravel())
            count_vec_ff += nne_f * nne_f
            # fm
            row_list_fm.append(
                np.repeat(np.array(e_nodes_m, dtype=np.int32).reshape(-1, 1), nne_f, axis=1).ravel()
            )
            col_list_fm.append(np.tile(np.array(e_nodes_f, dtype=np.int32), (1, nne_m)).ravel())
            count_vec_fm += nne_f * nne_m
            # mm
            row_list_mm.append(
                np.repeat(np.array(e_nodes_m, dtype=np.int32).reshape(-1, 1), nne_m, axis=1).ravel()
            )
            col_list_mm.append(np.tile(np.array(e_nodes_m, dtype=np.int32), (1, nne_m)).ravel())
            count_vec_mm += nne_m * nne_m

    row_vec_ff = np.concatenate(row_list_ff).astype(np.int32)
    col_vec_ff = np.concatenate(col_list_ff).astype(np.int32)
    row_vec_fm = np.concatenate(row_list_fm).astype(np.int32)
    col_vec_fm = np.concatenate(col_list_fm).astype(np.int32)
    row_vec_mm = np.concatenate(row_list_mm).astype(np.int32)
    col_vec_mm = np.concatenate(col_list_mm).astype(np.int32)

    return (
        count_vec_ff,
        row_vec_ff,
        col_vec_ff,
        count_vec_fm,
        row_vec_fm,
        col_vec_fm,
        count_vec_mm,
        row_vec_mm,
        col_vec_mm,
    )


def _integrate_fracture_matrix_coupling(
    e: int,
    enodes_f: list[int],
    nodes_f: npt.NDArray[np.float64],
    N_q: npt.NDArray[np.float64],
    wJe_q: npt.NDArray[np.float64],
    mesh: dict[str, Any],
    *,
    kf_p: float | None = None,
    mu_q: npt.NDArray[np.float64] | float | None = None,
    h: float | None = None,
) -> list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Integrate one fracture element's coupling contribution to its neighboring matrix elements.

    `mu_q` may be a constant (linear viscosity) or
    a per-quadrature-point array (nonlinear viscosity).
    Thermal coupling is included when `h` is provided; pressure coupling is included
    when both `kf_p` and `mu_q` are provided.
    """
    conn_m = mesh["conn"]
    nodes_m = mesh["nodes"]
    etype_list_m = mesh["element_type"]
    elem_indx_m = mesh["elem_indx"]
    conn_fm = mesh["fracture_mesh"]["matrix_conn"]

    entries: list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]] = []

    for e_m in conn_fm[e]:
        enodes_m = conn_m[e_m]
        nne_m = len(enodes_m)

        etype = etype_list_m[bisect.bisect_left(elem_indx_m, e_m)]
        getN_m = NMATRIX_FUNCS[etype]
        Q = _get_parent_coordinates(etype, nodes_m[enodes_m, :], N_q @ nodes_f[enodes_f, :])
        N_qm = np.array([getN_m(Q[q, :].flatten()) for q in range(np.size(Q, -1))])

        ff_matrices: dict[str, npt.NDArray[np.float64]] = {}
        fm_matrices: dict[str, npt.NDArray[np.float64]] = {}
        mm_matrices: dict[str, npt.NDArray[np.float64]] = {}

        if h is not None:
            ff_matrices["KTT_ff"] = _vectorized_quad_point_integration(N_q, h, N_q, wJe_q, True)
            fm_matrices["KTT_fm"] = _vectorized_quad_point_integration(N_qm, -h, N_q, wJe_q, True)
            mm_matrices["KTT_mm"] = _vectorized_quad_point_integration(N_qm, h, N_qm, wJe_q, True)

        if kf_p is not None and mu_q is not None:
            kf_p_mu = kf_p / mu_q
            ff_matrices["Kpp_ff"] = _vectorized_quad_point_integration(N_q, -kf_p_mu, N_q, wJe_q, True)
            fm_matrices["Kpp_fm"] = _vectorized_quad_point_integration(N_qm, kf_p_mu, N_q, wJe_q, True)
            mm_matrices["Kpp_mm"] = _vectorized_quad_point_integration(N_qm, -kf_p_mu, N_qm, wJe_q, True)

        entries.append((nne_m, ff_matrices, fm_matrices, mm_matrices))

    return entries


def _stack_fracture_matrices(
    body_matrices: dict[str, csc_matrix[Any]],
    fracture_matrices: dict[str, csc_matrix[Any]],
    nn_m: int,
    nn_f: int,
) -> dict[str, csc_matrix[Any]]:
    """Stack the fracture and matrix matrices along with their off-diagonal coupling matrices."""
    for name in body_matrices:
        # On-diagonal fracture matrix
        if name in fracture_matrices:
            fracture_matrix = fracture_matrices[name].tocoo()
        else:
            fracture_matrix = csc_matrix((nn_f, nn_f)).tocoo()

        # on-diagonal correction to the fracture's own block
        if name + "_ff" in fracture_matrices:
            fracture_matrix = fracture_matrix + fracture_matrices[name + "_ff"].tocoo()

        # Off-diagonal coupling matrix
        if name + "_fm" in fracture_matrices:
            diag_matrix = fracture_matrices[name + "_fm"].tocoo()
        else:
            diag_matrix = csc_matrix((nn_m, nn_f)).tocoo()

        # on-diagonal correction to the matrix's own block
        matrix_matrix = body_matrices[name].tocoo()
        if name + "_mm" in fracture_matrices:
            matrix_matrix = matrix_matrix + fracture_matrices[name + "_mm"].tocoo()

        row1 = hstack([matrix_matrix.tocoo(), diag_matrix])
        row2 = hstack([diag_matrix.T, fracture_matrix])
        body_matrices[name] = vstack([row1, row2]).tocsc()
    return body_matrices


def _get_region_mesh_data(
    mesh: dict[str, Any], is_fracture: bool
) -> tuple[
    int,  # nn
    int,  # nsd
    list[list[int]],  # conn
    list[int],  # nne_list
    list[str],  # etype_list
    list[int],  # elem_indx
    list[int],  # nq_list
    list[list[int]],  # region_lists
    int,  # num_regions
    dict[str, Any],  # fmesh; {} sentinel when not is_fracture
    npt.NDArray[np.float64],  # nodes_f; empty sentinel when not is_fracture
]:
    """Resolve mesh/region data from the bulk mesh, or the fracture mesh when `is_fracture`."""
    if is_fracture:
        fmesh = mesh["fracture_mesh"]
        return (
            fmesh["nn"],
            mesh["nsd"] - 1,
            fmesh["conn"],
            fmesh["nne"],
            fmesh["element_type"],
            fmesh["elem_indx"],
            fmesh["n_quadrature"],
            fmesh["region_lists"],
            len(fmesh["region_lists"]),
            fmesh,
            fmesh["nodes"],
        )
    return (
        mesh["nn"],
        mesh["nsd"],
        mesh["conn"],
        mesh["nne"],
        mesh["element_type"],
        mesh["elem_indx"],
        mesh["n_quadrature"],
        mesh["region_lists"],
        mesh["num_regions"],
        {},
        np.empty((0, 0)),
    )


@dataclass
class _CachedCouplingVectors:
    """Fracture/matrix coupling row/col index vectors cached on `fmesh` by
    `integrate_linear_matrices`'s fracture pass and reused every timestep by
    `integrate_nonlinear_matrices` and `integrate_NLpressure_matrices`.
    """

    row_vec_ff: npt.NDArray[np.int32]
    col_vec_ff: npt.NDArray[np.int32]
    row_vec_fm: npt.NDArray[np.int32]
    col_vec_fm: npt.NDArray[np.int32]
    row_vec_mm: npt.NDArray[np.int32]
    col_vec_mm: npt.NDArray[np.int32]


def _get_region_flow_data(
    mesh: dict[str, Any],
    mat: dict[str, Any],
    fmesh: dict[str, Any],
    nodes_f: npt.NDArray[np.float64],
    is_fracture: bool,
    r: int,
) -> tuple[
    npt.NDArray[np.float64] | float,  # kf, permeability
    bool,  # isotropic_perm
    npt.NDArray[np.float64],  # nodes_r
    float,  # w, fracture aperture; NaN sentinel when not is_fracture
    npt.NDArray[np.float64],  # u, fracture tangent; NaN sentinel when not is_fracture
    float,  # kf_p, transverse permeability, coupling only; NaN sentinel when not is_fracture
    npt.NDArray[np.float64] | float,  # comp_f, pore/fracture fluid compressibility - KM only
]:
    """Resolve per-region permeability, node-projection, and storage data."""
    # Fracture case
    if is_fracture:
        w = mat["fracture"]["aperture"][r]
        kf = w**3 / 12  # longitudinal permeability, cubic law
        kf_p = w / 12  # transverse permeability, coupling only
        comp_f = w / mat["fluid"]["bulk_mod"]
        u = fmesh["fracture_tangent"][r].reshape(-1, 1)
        nodes_r = (nodes_f @ u).reshape(-1, 1)  # project mesh nodes into local coordinates
        return kf, True, nodes_r, w, u, kf_p, comp_f

    # Bulk case
    kf = mat["solid"]["permeability_eff"][r]
    isotropic_perm = bool(isinstance(kf, (float, int)))
    comp_f = np.asarray(1 / mat["solid"]["storage_mod"][r])
    return kf, isotropic_perm, mesh["nodes"], float("nan"), np.full((1, 1), np.nan), float("nan"), comp_f


def _get_region_thermal_data(
    mat: dict[str, Any], is_fracture: bool, r: int
) -> tuple[
    npt.NDArray[np.float64] | float,  # lambda_e, thermal conductivity
    bool,  # isotropic_cond
    float,  # h, interface heat transfer coeff, coupling only; NaN sentinel when not is_fracture
    npt.NDArray[np.float64] | float,  # m_e, volumetric heat capacity - C only
    npt.NDArray[np.float64],  # alpha_e, thermal expansion coeff - KpT only; NaN sentinel when is_fracture
]:
    """Resolve per-region thermal conductivity, capacity, and expansion data."""
    # Fracture case
    if is_fracture:
        lambda_e = mat["fluid"]["therm_cond"]
        w = mat["fracture"]["aperture"][r]
        nusselt = mat["fracture"]["nusselt"][r]
        h = nusselt * lambda_e / 2 / w  # interface heat transfer coefficient, coupling only
        m_e = mat["fluid"]["density"] * mat["fluid"]["specific_heat"] * w
        return lambda_e, True, h, m_e, np.full((1, 1), np.nan)

    # Bulk case
    lambda_e = mat["solid"]["therm_cond_eff"][r]
    isotropic_cond = bool(isinstance(lambda_e, (float, int)))
    m_e = mat["solid"]["volumetric_heat_capacity_eff"][r]
    alpha_e = np.asarray(mat["solid"]["therm_expansion_eff"][r])
    return lambda_e, isotropic_cond, float("nan"), m_e, alpha_e


_YELLOW = "\033[93m"
_RESET = "\033[0m"


def _warn_nonmonotone_pressure_diffusion(Kpp: csc_matrix[Any], rel_tol: float = 0.7) -> None:
    """Warn if the pressure stiffness matrix severely violates the discrete maximum principle.

    A physically well-behaved diffusion stiffness matrix is an M-matrix: every
    off-diagonal entry must be <= 0, so increasing pressure at a neighboring node
    can only ever pull pressure down at a node via diffusion, never push it up.
    Positive off-diagonal entries typically arise from a region's permeability
    tensor being strongly anisotropic and/or rotated relative to the mesh (i.e.
    not aligned with the local triangulation).

    Mild violations are common and mostly harmless, so `rel_tol` is calibrated high.
    """
    Kpp_coo = Kpp.tocoo()
    diag = np.asarray(Kpp.diagonal())

    off_diag = Kpp_coo.row != Kpp_coo.col
    positive = Kpp_coo.data > 0
    mask = off_diag & positive
    rows, cols, vals = Kpp_coo.row[mask], Kpp_coo.col[mask], Kpp_coo.data[mask]

    row_diag = diag[rows]
    significant = (row_diag > 0) & (vals > rel_tol * row_diag)
    rows, cols, vals = rows[significant], cols[significant], vals[significant]
    if rows.size == 0:
        return

    ratio = vals / diag[rows]
    order = np.argsort(-ratio)
    n_nodes = np.unique(rows).size
    worst = [(int(rows[i]), int(cols[i]), round(float(ratio[i]), 3)) for i in order[:5]]

    warnings.warn(
        f"{_YELLOW}Pressure stiffness matrix (Kpp) has {rows.size} strongly positive "
        f"off-diagonal entries (>{rel_tol:.0%} of the diagonal) across {n_nodes} node(s)."
        "This usually means a region's permeability tensor is too anisotropic and/or "
        "rotated relative to the mesh for the linear FEM discretization to remain "
        "monotonic. Spurious pressure/temperature oscillations may occur in that area."
        f"Worst offenders as (node, neighbor, off-diag/diag ratio): {worst}."
        "If this occurs, reduce the anisotropic ratio and/or align and stretch"
        f"the mesh to the anisotropy ratio.{_RESET}",
        stacklevel=2,
    )


def integrate_linear_matrices(  # noqa
    mesh: dict[str, Any],
    mat: dict[str, Any],
    controls: dict[str, Any],
    *,
    is_fracture: bool = False,
) -> tuple[dict[str, csc_matrix[Any]], dict[str, Any], dict[str, Any]]:
    """Integrate linear matrices outside of time integration loop.

    Matrices assembled:
        Kpp = Pressure conductance matrix
        KbT = Pressure buoyancy matrix
        KM =  Pressure storage matrix
        KpT =  Pressurization matrix (matrix only)
        C = Thermal capacity matrix
        KTT =  Thermal conductance matrix
        Kpp_fm, KTT_fm, Kpp_mm, KTT_mm = fracture-matrix coupling terms (fracture pass only)

    When called normally (`is_fracture=False`) and `fracture_on`, this recurses once onto
    the fracture mesh (`is_fracture=True`) and stacks the two systems together.
    """
    mu_linear = not bool(controls["viscosity_nonconstant"])
    axisymmetry = bool(controls["axisymmetric_on"])
    fracture_on = bool(controls["fracture_on"])

    # ----- fluid properties, universal
    if mu_linear:
        mu = mat["fluid"]["viscosity"]
        alpha_f = mat["fluid"]["therm_expansion"]
    rho_f = mat["fluid"]["density"]
    g, ng = _get_gravity(mesh["nsd"])

    # ----- mesh data: bulk (matrix) mesh, or the fracture mesh on the recursive pass
    (
        nn,
        nsd,
        conn,
        nne_list,
        etype_list,
        elem_indx,
        nq_list,
        region_lists,
        num_regions,
        fmesh,
        nodes_f,
    ) = _get_region_mesh_data(mesh, is_fracture)

    if is_fracture:
        conn_fm = fmesh["matrix_conn"]

        list_of_matrices = ["KM", "C", "KTT"] + (["KbT", "Kpp"] if mu_linear else [])
        list_of_ff_matrices = ["KTT_ff"] + (["Kpp_ff"] if mu_linear else [])
        list_of_fm_matrices = ["KTT_fm"] + (["Kpp_fm"] if mu_linear else [])
        list_of_mm_matrices = ["KTT_mm"] + (["Kpp_mm"] if mu_linear else [])
        (
            vec_count_ff,
            row_vec_ff,
            col_vec_ff,
            vec_count_fm,
            row_vec_fm,
            col_vec_fm,
            vec_count_mm,
            row_vec_mm,
            col_vec_mm,
        ) = _create_fracture_coupling_vectors(conn, conn_fm, mesh["conn"])
        # Cache for reuse by integrate_nonlinear_matrices / integrate_NLpressure_matrices,
        # rather than re-deriving this connectivity-only (mesh-static) data every timestep.
        fmesh["vec_mat_asmbl"] = _CachedCouplingVectors(
            row_vec_ff=row_vec_ff,
            col_vec_ff=col_vec_ff,
            row_vec_fm=row_vec_fm,
            col_vec_fm=col_vec_fm,
            row_vec_mm=row_vec_mm,
            col_vec_mm=col_vec_mm,
        )

        vec_ff = {name: np.zeros(vec_count_ff) for name in list_of_ff_matrices}
        vec_fm = {name: np.zeros(vec_count_fm) for name in list_of_fm_matrices}
        vec_mm = {name: np.zeros(vec_count_mm) for name in list_of_mm_matrices}
        vec_counter_ff = 0
        vec_counter_fm = 0
        vec_counter_mm = 0
    else:
        list_of_matrices = ["KM", "KpT", "C", "KTT"] + (["Kpp", "KbT"] if mu_linear else [])

    vec = _initiate_vectors(list_of_matrices, nne_list, elem_indx)
    vec_counter = 0  # vectorized storage index counter

    # ---- Matrix integration element loop
    for r in range(num_regions):
        kf, isotropic_perm, nodes_r, w, u, kf_p, comp_f = _get_region_flow_data(
            mesh, mat, fmesh, nodes_f, is_fracture, r
        )
        lambda_e, isotropic_cond, h, m_e, alpha_e = _get_region_thermal_data(mat, is_fracture, r)

        if is_fracture:
            if mu_linear:
                kf_b = ng.T @ np.dot(kf, ng) / mu * alpha_f * rho_f @ g.T @ u  # local gravity component
        else:
            if mu_linear:
                kf_b = ng.T @ np.dot(kf, ng) / mu * alpha_f * rho_f @ g.T  # kf component in gravity direction

        # find element type of first element in region
        elem_type_counter = -1  # force an update on the first element

        for e in region_lists[r]:
            new_type_idx = bisect.bisect_left(elem_indx, e)
            if new_type_idx != elem_type_counter:
                # Update for different element type
                # Evaluate quadrature points and get weights and shape functions.
                W, nq, N_q, GN_q = _get_quad_points_and_shape_functions(
                    nq_list[new_type_idx], nsd, etype_list[new_type_idx]
                )
                nne = nne_list[new_type_idx]
                elem_type_counter = new_type_idx

            enodes = conn[e]  # local element nodes
            xI = nodes_r[enodes, :]
            B_q, detJe_q = getB(GN_q, xI, nsd)
            wJe_q = W * detJe_q  # Weight * det(Jacobian), used in every integrand
            if axisymmetry:
                # Add radial component to integrand for axisymmetry
                r_coord = N_q @ xI[:, 0]
                wJe_q = wJe_q * r_coord.T

            # Element matrices
            element_matrices = {
                "KM": _vectorized_quad_point_integration(N_q, comp_f, N_q, wJe_q, True),
                "C": _vectorized_quad_point_integration(N_q, m_e, N_q, wJe_q, True),
                "KTT": _vectorized_quad_point_integration(B_q, lambda_e, B_q, wJe_q, isotropic_cond),
            }
            if not is_fracture:
                element_matrices["KpT"] = _vectorized_quad_point_integration(N_q, alpha_e, N_q, wJe_q, True)
            if mu_linear:
                element_matrices["Kpp"] = _vectorized_quad_point_integration(
                    B_q, kf / mu, B_q, wJe_q, isotropic_perm
                )
                # KbT is a scalar product once projected onto the fracture's tangent, a matmul
                # product for the bulk mesh - `is_fracture` doubles as that scalar/matmul flag.
                element_matrices["KbT"] = _vectorized_quad_point_integration(
                    N_q, kf_b, B_q, wJe_q, is_fracture
                )

            # ---- Assemble element contributions into global matrices
            stop = vec_counter + nne * nne  # vectorized index count end
            _store_vectorized_element_matrix(vec, element_matrices, vec_counter, stop)
            vec_counter = stop

            # ---- Fractures
            if is_fracture:
                # Coupling matrices
                coupling_entries = _integrate_fracture_matrix_coupling(
                    e,
                    enodes,
                    nodes_f,
                    N_q,
                    wJe_q,
                    mesh,
                    kf_p=kf_p,
                    mu_q=(mu if mu_linear else None),
                    h=h,
                )

                for nne_m, ff_matrices, fm_matrices, mm_matrices in coupling_entries:
                    stop = vec_counter_ff + nne * nne  # vectorized index count end
                    _store_vectorized_element_matrix(vec_ff, ff_matrices, vec_counter_ff, stop)
                    vec_counter_ff = stop
                    stop = vec_counter_fm + nne_m * nne  # vectorized index count end
                    _store_vectorized_element_matrix(vec_fm, fm_matrices, vec_counter_fm, stop)
                    vec_counter_fm = stop
                    stop = vec_counter_mm + nne_m * nne_m  # vectorized index count end
                    _store_vectorized_element_matrix(vec_mm, mm_matrices, vec_counter_mm, stop)
                    vec_counter_mm = stop

    integrated_matrices = _assemble_global_matrices(
        vec, conn, nn, nne_list, elem_indx, np.concatenate(region_lists)
    )

    # ----- Mass lumping: sum rows of KM and C, store as diagonal (nn, nn) csc matrices
    for name in ("KM", "C"):
        integrated_matrices[name] = _lump_diagonal(integrated_matrices[name])

    if is_fracture:
        # Assemble the coupling matrices
        nn_m = mesh["nn"]
        for name in vec_ff:
            integrated_matrices[name] = _make_sparse(vec_ff[name], row_vec_ff, col_vec_ff, nn, nn)
        for name in vec_fm:
            integrated_matrices[name] = _make_sparse(vec_fm[name], row_vec_fm, col_vec_fm, nn_m, nn)
        for name in vec_mm:
            integrated_matrices[name] = _make_sparse(vec_mm[name], row_vec_mm, col_vec_mm, nn_m, nn_m)
        return integrated_matrices, mat, mesh

    if fracture_on:
        # Recursion call
        integrated_fracture_matrices, _, _ = integrate_linear_matrices(mesh, mat, controls, is_fracture=True)
        # Combine fracture and matrix matrices into one
        integrated_matrices = _stack_fracture_matrices(
            integrated_matrices, integrated_fracture_matrices, nn, mesh["fracture_mesh"]["nn"]
        )

    if "Kpp" in integrated_matrices:
        _warn_nonmonotone_pressure_diffusion(integrated_matrices["Kpp"])

    return (
        integrated_matrices,
        mat,
        mesh,
    )


@dataclass
class _SUPGQuantities:
    """Flux, Peclet number, and SUPG stabilization terms at an element's quadrature points.

    Computed once per element by `_compute_supg_quantities` and shared by every
    physics block that needs them (base advection, viscosity correction, and the
    fully-coupled pressure/temperature off-diagonal terms), instead of recomputing
    or re-threading them through each block individually.
    """

    gradp: npt.NDArray[np.float64]  # nq x nsd x 1
    flux: npt.NDArray[np.float64]  # nq x nsd x 1
    flux_T: npt.NDArray[np.float64]  # nq x 1 x nsd # noqa
    flux_mag: npt.NDArray[np.float64]  # nq,
    tau: npt.NDArray[np.float64]  # nq x 1 x 1
    Pe: npt.NDArray[np.float64]  # Peclet number
    integrand_Kstab: npt.NDArray[np.float64]  # tau * flux @ flux_T, used for K_stab_TT # noqa


def _compute_supg_quantities(
    B_q: npt.NDArray[np.float64],
    pe: npt.NDArray[np.float64],
    advec_conduct: npt.NDArray[np.float64],
    isotropic_perm: bool,
    isotropic_cond: bool,
    lambda_e: npt.NDArray[np.float64] | float,
    L_vec: npt.NDArray[np.float64],
    nq: int,
    nsd: int,
) -> _SUPGQuantities:
    """Compute the Darcy flux and SUPG (Shakib, Hughes, & Zohan (1991)) stabilization terms."""
    gradp = B_q @ pe  # nq x nsd x 1

    flux = -advec_conduct * gradp if isotropic_perm else -advec_conduct @ gradp
    flux_T = flux.transpose(0, 2, 1)  # quadrature-point-wise transpose, calculate once
    if nsd == 3:
        flux_mag = np.sqrt(flux[:, 0, 0] ** 2 + flux[:, 1, 0] ** 2 + flux[:, 2, 0] ** 2)  # flux magnitude
    elif nsd == 2:
        flux_mag = np.sqrt(flux[:, 0, 0] ** 2 + flux[:, 1, 0] ** 2)  # flux magnitude
    else:
        flux_mag = np.abs(flux)
    indx_noflow = np.nonzero(flux_mag == 0)
    # prevent divide by zero
    flux_mag[indx_noflow] = 1.0e-15
    flux[indx_noflow, :, :] = 1.0e-15

    # Effective length weighted by fluxes
    if nsd == 3:
        h_eff = flux_mag / np.sqrt(
            (flux[:, 0, 0] / L_vec[0, 0]) ** 2
            + (flux[:, 1, 0] / L_vec[1, 0]) ** 2
            + (flux[:, 2, 0] / L_vec[2, 0]) ** 2
        )
    elif nsd == 2:
        h_eff = flux_mag / np.sqrt((flux[:, 0, 0] / L_vec[0, 0]) ** 2 + (flux[:, 1, 0] / L_vec[1, 0]) ** 2)
    else:
        h_eff = L_vec * np.ones((1, 1, nq))
        flux_mag = flux_mag.reshape(1, 1, -1)  # reshape for 1D broadcasting

    if isotropic_cond:
        lambda_e_projected = lambda_e * np.ones((nq, 1, 1))
    else:
        # anisotropic case - project conductivity along flowlines
        n_stream = flux / flux_mag.reshape(-1, 1, 1)
        n_stream_T = n_stream.transpose(0, 2, 1)
        lambda_e_projected = (n_stream_T @ lambda_e) @ n_stream  # nq x 1 x 1

    # Peclet number
    Pe = h_eff * flux_mag / lambda_e_projected.reshape(1, 1, -1)

    # SUPG stabilization coefficient
    tau = h_eff / (2 * flux_mag) * (1 + 9 / Pe**2) ** (-0.5)
    tau[:, :, indx_noflow] = 0
    tau = tau.reshape(-1, 1, 1)

    integrand_Kstab = tau * flux @ flux_T

    return _SUPGQuantities(
        gradp=gradp,
        flux=flux,
        flux_T=flux_T,
        flux_mag=flux_mag,
        tau=tau,
        Pe=Pe,
        integrand_Kstab=integrand_Kstab,
    )


def _integrate_pressure_flow_terms(
    N_q: npt.NDArray[np.float64],
    B_q: npt.NDArray[np.float64],
    wJe_q: npt.NDArray[np.float64],
    kf: npt.NDArray[np.float64] | float,
    mu_q: npt.NDArray[np.float64],
    kf_b: npt.NDArray[np.float64],
    isotropic_perm: bool,
    kf_b_scalar: bool = False,
) -> dict[str, npt.NDArray[np.float64]]:
    """Integrate the viscosity-dependent pressure conductance (Kpp) and buoyancy (KbT) matrices.

    `kf_b_scalar` distinguishes the bulk mesh (KbT is a matmul product, the default) from
    the fracture mesh (KbT reduces to a scalar product once projected onto the tangent).
    """
    return {
        "Kpp": _vectorized_quad_point_integration(B_q, kf / mu_q, B_q, wJe_q, isotropic_perm),
        "KbT": _vectorized_quad_point_integration(N_q, kf_b, B_q, wJe_q, kf_b_scalar),
    }


def _integrate_viscosity_temperature_correction(
    N_q: npt.NDArray[np.float64],
    B_q: npt.NDArray[np.float64],
    wJe_q: npt.NDArray[np.float64],
    gradT: npt.NDArray[np.float64],
    supg: _SUPGQuantities,
    mu_q: npt.NDArray[np.float64],
    dmu_dT_q: npt.NDArray[np.float64],
) -> dict[str, npt.NDArray[np.float64]]:
    """Integrate the viscosity-derivative correction to the temperature advection/stabilization matrices."""
    FA = supg.flux_T @ gradT
    integrand_KATT_mu = -1 / mu_q @ FA * dmu_dT_q

    Fstab = (supg.tau * supg.flux) @ (supg.flux_T @ gradT)

    # tau derivative wrt mu(T) correction factor
    correction_factor = 0.5 * (supg.Pe**2 + 18) / (supg.Pe**2 + 9)
    integrand_Kstab_mu = -2 / mu_q * Fstab * dmu_dT_q * correction_factor.reshape(-1, 1, 1)

    return {
        "KATT_mu": _vectorized_quad_point_integration(N_q, integrand_KATT_mu, N_q, wJe_q, False),
        "K_stab_TT_mu": _vectorized_quad_point_integration(B_q, integrand_Kstab_mu, N_q, wJe_q, False),
    }


def _integrate_coupled_pressure_terms(
    N_q: npt.NDArray[np.float64],
    B_q: npt.NDArray[np.float64],
    wJe_q: npt.NDArray[np.float64],
    gradT: npt.NDArray[np.float64],
    supg: _SUPGQuantities,
    advec_conduct: npt.NDArray[np.float64],
    isotropic_perm: bool,
    nsd: int,
) -> dict[str, npt.NDArray[np.float64]]:
    """Integrate the fully-coupled temperature-wrt-pressure off-diagonal matrices (KATp, K_stab_Tp)."""
    Rth_dp = -advec_conduct * gradT if isotropic_perm else -advec_conduct @ gradT
    integrand_KATp = Rth_dp.transpose(0, 2, 1)  # nq x 1 x nsd

    # Stabilization wrt p
    Rth = supg.flux_T @ gradT
    inner_term = Rth * np.eye(nsd) + supg.flux @ gradT.transpose(0, 2, 1)

    flux_mag_sq = supg.flux_mag.reshape(-1, 1, 1) ** 2
    pe_factor = supg.Pe**2 / (supg.Pe**2 + 9)
    tau_correction_Tp = pe_factor.reshape(-1, 1, 1) * (Rth / flux_mag_sq) * (supg.flux @ supg.flux_T)
    inner_term = inner_term - tau_correction_Tp

    integrand_Kstab_Tp = (
        -supg.tau * advec_conduct * inner_term if isotropic_perm else -supg.tau * (inner_term @ advec_conduct)
    )

    return {
        "KATp": _vectorized_quad_point_integration(N_q, integrand_KATp, B_q, wJe_q, False),
        "K_stab_Tp": _vectorized_quad_point_integration(B_q, integrand_Kstab_Tp, B_q, wJe_q, False),
    }


def _integrate_viscosity_pressure_correction(
    N_q: npt.NDArray[np.float64],
    B_q: npt.NDArray[np.float64],
    wJe_q: npt.NDArray[np.float64],
    supg: _SUPGQuantities,
    kf: npt.NDArray[np.float64] | float,
    mu_q: npt.NDArray[np.float64],
    dmu_dT_q: npt.NDArray[np.float64],
    kf_b: npt.NDArray[np.float64],
    del_T: npt.NDArray[np.float64],
    isotropic_perm: bool,
) -> dict[str, npt.NDArray[np.float64]]:
    """Integrate the viscosity-wrt-temperature correction to the pressure/buoyancy matrices."""
    # contribution of viscosity to pressure terms
    Fk = kf / mu_q * supg.gradp if isotropic_perm else kf / mu_q @ supg.gradp  # (nq, nsd, 1)
    Fb = kf_b @ B_q @ del_T  # (nq, 1, 1)

    #   Darcy's law component
    integrand_KpT_mu_p = -(dmu_dT_q / mu_q) * Fk  # (nq, nsd, 1), elementwise
    #   Buoyancy component
    integrand_KpT_mu_b = -1.0 / mu_q * Fb @ dmu_dT_q

    return {
        "KpT_mu_p": _vectorized_quad_point_integration(B_q, integrand_KpT_mu_p, N_q, wJe_q, False),
        "KpT_mu_b": _vectorized_quad_point_integration(N_q, integrand_KpT_mu_b, N_q, wJe_q, False),
    }


def integrate_nonlinear_matrices(  # noqa
    mesh: dict[str, Any],
    mat: dict[str, Any],
    primary_variables: dict[str, Any],
    controls: dict[str, Any],
    *,
    is_fracture: bool = False,
) -> dict[str, Any]:
    """Integrate non-linear matrices inside time integration loop.

    KATT = Temperature advection tangent matrix wrt temperature
    K_stab_TT = Advection stabilization wrt temperature
    KATp = Temperature advection tangent matrix wrt pressure
    K_stab_Tp = Advection stabilization wrt pressure
    Kpp, KbT, KpT_mu_p, KpT_mu_b, KATT_mu, K_stab_TT_mu = viscosity-nonlinear corrections, see helpers below.

    When called normally (`is_fracture=False`) and `fracture_on`, this recurses once onto
    the fracture mesh (`is_fracture=True`).
    """
    p = primary_variables["p"][:, 0]
    T = primary_variables["T"][:, 0]
    mu_nonlinear = bool(controls["viscosity_nonconstant"])
    mu_linear = not mu_nonlinear
    fully_coupled = bool(controls["fully_coupled"])
    axisymmetry = bool(controls["axisymmetric_on"])
    fracture_on = bool(controls["fracture_on"])

    mu = mat["fluid"]["viscosity"]
    cp_f = mat["fluid"]["specific_heat"]
    rho_f = mat["fluid"]["density"]
    if mu_nonlinear:
        alpha_f = mat["fluid"]["therm_expansion"]

    list_of_matrices = ["KATT", "K_stab_TT"]
    if fully_coupled:
        list_of_matrices.extend(["KATp", "K_stab_Tp"])
    if mu_nonlinear:
        list_of_matrices.extend(["Kpp", "KbT", "KpT_mu_b", "KpT_mu_p", "KATT_mu", "K_stab_TT_mu"])

    # ----- mesh data: bulk (matrix) mesh, or the fracture mesh on the recursive pass
    (
        nn,
        nsd,
        conn,
        nne_list,
        etype_list,
        elem_indx,
        nq_list,
        region_lists,
        num_regions,
        fmesh,
        nodes_f,
    ) = _get_region_mesh_data(mesh, is_fracture)

    if is_fracture:
        conn_global = fmesh["conn_global"]

        if mu_nonlinear:
            # Connectivity-only (mesh-static) data, cached by integrate_linear_matrices's
            # fracture pass rather than re-derived here every timestep/Newton iteration.
            cv = fmesh["vec_mat_asmbl"]
            row_vec_ff, col_vec_ff = cv.row_vec_ff, cv.col_vec_ff
            row_vec_fm, col_vec_fm = cv.row_vec_fm, cv.col_vec_fm
            row_vec_mm, col_vec_mm = cv.row_vec_mm, cv.col_vec_mm

            vec_ff = {"Kpp_ff": np.zeros(row_vec_ff.size)}
            vec_fm = {"Kpp_fm": np.zeros(row_vec_fm.size)}
            vec_mm = {"Kpp_mm": np.zeros(row_vec_mm.size)}

            vec_counter_ff = 0
            vec_counter_fm = 0
            vec_counter_mm = 0

    # ----- gravity vector
    g, ng = _get_gravity(mesh["nsd"])

    vec = _initiate_vectors(list_of_matrices, nne_list, elem_indx)
    vec_counter = 0  # vectorized storage index counter

    # ----- Matrix integration element loop
    for r in range(num_regions):
        kf, isotropic_perm, nodes_r, w, u, kf_p, _ = _get_region_flow_data(
            mesh, mat, fmesh, nodes_f, is_fracture, r
        )
        lambda_e, isotropic_cond, _, _, _ = _get_region_thermal_data(mat, is_fracture, r)

        # advection velocity = advection_conductivity * pressure gradient
        if mu_linear:
            advec_conduct = cp_f * rho_f * kf / mu  # advection conductivity

        # find element type of first element in region
        elem_type_counter = -1  # force an update on the first element

        for e in region_lists[r]:
            new_type_idx = bisect.bisect_left(elem_indx, e)
            if new_type_idx != elem_type_counter:
                # Update for different element type
                # Evaluate quadrature points and get weights and shape functions.
                W, nq, N_q, GN_q = _get_quad_points_and_shape_functions(
                    nq_list[new_type_idx], nsd, etype_list[new_type_idx]
                )
                nne = nne_list[new_type_idx]
                elem_type_counter = new_type_idx

            enodes = conn[e]  # local element nodes
            enodes_dof = conn_global[e] if is_fracture else enodes  # global DoF index, for T/p
            xI = nodes_r[enodes, :]
            B_q, detJe_q = getB(GN_q, xI, nsd)
            wJe_q = W * detJe_q  # Weight * det(Jacobian), used in every integrand
            if axisymmetry:
                # Add radial component to integrand for axisymmetry
                r_coord = N_q @ xI[:, 0]
                wJe_q = wJe_q * r_coord.T

            # Element matrices initialize
            element_matrices: dict[str, npt.NDArray[np.float64]] = {}

            Te = T[enodes_dof].reshape(-1, 1)  # element nodal temperatures
            pe = p[enodes_dof].reshape(-1, 1)  # element nodal pressures
            L_vec = (xI.max(axis=0) - xI.min(axis=0)).reshape(-1, 1)  # element lengths

            # ------------------------------------ Pressure and fluid flow
            if mu_nonlinear:
                # If mu is nonlinear, the pressure diffusion is also non-linear and must
                # be re-integrated at each timestep.
                T_q = N_q @ Te  # quadrature point temperatures
                mu_q, dmu_dT_q = get_viscosity(T_q, mu)  # quadrature point viscosities

                advec_conduct = cp_f * rho_f * kf / mu_q  # advection conductivity
                kf_b = (
                    ng.T @ np.dot(kf, ng) / mu_q * alpha_f * rho_f @ (g.T @ u if is_fracture else g.T)
                )  # kf component in gravity direction

                # Integrate the tangent matrices
                # Kpp = dFk / dp - Darcy's Law
                # KbT = dFb / dT - Buoyancy
                element_matrices.update(
                    _integrate_pressure_flow_terms(
                        N_q, B_q, wJe_q, kf, mu_q, kf_b, isotropic_perm, is_fracture
                    )
                )

            # ----------------------------------- Advection and stabilization
            # Calculate fluxes, stabilization coefficicent, pressure gradients and store in supg dataclass
            supg = _compute_supg_quantities(
                B_q, pe, advec_conduct, isotropic_perm, isotropic_cond, lambda_e, L_vec, nq, nsd
            )

            # advection matrix
            element_matrices["KATT"] = _vectorized_quad_point_integration(N_q, supg.flux_T, B_q, wJe_q, False)
            # advection stabilization
            element_matrices["K_stab_TT"] = _vectorized_quad_point_integration(
                B_q, supg.integrand_Kstab, B_q, wJe_q, False
            )

            gradT = B_q @ Te

            # ---------------------------------- Viscosity correction
            if mu_nonlinear:
                # Integrate tangent matrices:
                # KATT_mu = dFA / dmu * dmu / dT
                # K_stab_TT_mu = dF_stab / dmu * dmu / dT
                element_matrices.update(
                    _integrate_viscosity_temperature_correction(N_q, B_q, wJe_q, gradT, supg, mu_q, dmu_dT_q)
                )

            # ----------------------------------- Fully coupled off diagonal terms
            if fully_coupled:
                # Integrate the off-diagonal tangents wrt pressure
                # KATp = dFA / dp
                # K_stab_Tp = dF_stab / dp
                element_matrices.update(
                    _integrate_coupled_pressure_terms(
                        N_q, B_q, wJe_q, gradT, supg, advec_conduct, isotropic_perm, nsd
                    )
                )

                # viscosity wrt T
                if mu_nonlinear:
                    # If non-linear viscosity, integrate the viscosity tangents
                    # KpT_mu_p = dFk / dmu * dmu / dT - for Darcy's law
                    # KpT_mu_b = dFb / dmu * dmu / dT - for buoyancy
                    T_0e = mat["T_0"][enodes_dof]
                    del_T = Te - T_0e.reshape(-1, 1)
                    element_matrices.update(
                        _integrate_viscosity_pressure_correction(
                            N_q, B_q, wJe_q, supg, kf, mu_q, dmu_dT_q, kf_b, del_T, isotropic_perm
                        )
                    )

            # ----- Fracture-matrix pressure coupling
            if is_fracture and mu_nonlinear:
                # Coupling matrices
                coupling_entries = _integrate_fracture_matrix_coupling(
                    e,
                    enodes,
                    nodes_f,
                    N_q,
                    wJe_q,
                    mesh,
                    kf_p=kf_p,
                    mu_q=mu_q,
                )

                for nne_m, ff_matrices, fm_matrices, mm_matrices in coupling_entries:
                    stop = vec_counter_ff + nne * nne  # vectorized index count end
                    _store_vectorized_element_matrix(vec_ff, ff_matrices, vec_counter_ff, stop)
                    vec_counter_ff = stop
                    stop = vec_counter_fm + nne_m * nne  # vectorized index count end
                    _store_vectorized_element_matrix(vec_fm, fm_matrices, vec_counter_fm, stop)
                    vec_counter_fm = stop
                    stop = vec_counter_mm + nne_m * nne_m  # vectorized index count end
                    _store_vectorized_element_matrix(vec_mm, mm_matrices, vec_counter_mm, stop)
                    vec_counter_mm = stop

            # ----- Assemble element contributions into global matrices
            stop = vec_counter + nne * nne  # vectorized index count end
            _store_vectorized_element_matrix(vec, element_matrices, vec_counter, stop)
            vec_counter = stop

    # ----- Return and convert to sparse matrix
    integrated_matrices = _assemble_global_matrices(
        vec, conn, nn, nne_list, elem_indx, np.concatenate(region_lists)
    )

    if is_fracture:
        if mu_nonlinear:
            # Assemble the coupling matrices
            nn_m = mesh["nn"]
            for name in vec_ff:
                integrated_matrices[name] = _make_sparse(vec_ff[name], row_vec_ff, col_vec_ff, nn, nn)
            for name in vec_fm:
                integrated_matrices[name] = _make_sparse(vec_fm[name], row_vec_fm, col_vec_fm, nn_m, nn)
            for name in vec_mm:
                integrated_matrices[name] = _make_sparse(vec_mm[name], row_vec_mm, col_vec_mm, nn_m, nn_m)
        return integrated_matrices

    if fracture_on:
        # Recursion call
        integrated_fracture_matrices = integrate_nonlinear_matrices(
            mesh, mat, primary_variables, controls, is_fracture=True
        )
        # Combine fracture and matrix matrices into one
        integrated_matrices = _stack_fracture_matrices(
            integrated_matrices, integrated_fracture_matrices, nn, mesh["fracture_mesh"]["nn"]
        )

    return integrated_matrices


def integrate_NLpressure_matrices(  # noqa
    mesh: dict[str, Any],
    mat: dict[str, Any],
    primary_variables: dict[str, Any],
    controls: dict[str, Any],
    *,
    is_fracture: bool = False,
) -> dict[str, Any]:
    """Integrate the non-linear-viscosity pressure/buoyancy matrices (Kpp, KbT) inside
    the time integration loop - a lighter-weight alternative to `integrate_nonlinear_matrices`
    for solvers that only need the pressure system.

    When called normally (`is_fracture=False`) and `fracture_on`, this recurses once onto
    the fracture mesh (`is_fracture=True`), integrating the `Kpp_fm`/`Kpp_mm` pressure
    coupling (responsive to the same nonlinear viscosity) in the same pass.
    """
    T = primary_variables["T"][:, 0]
    mu_coeff = mat["fluid"]["viscosity"]
    alpha_f = mat["fluid"]["therm_expansion"]
    rho_f = mat["fluid"]["density"]
    axisymmetry = bool(controls["axisymmetric_on"])
    fracture_on = bool(controls["fracture_on"])
    g, ng = _get_gravity(mesh["nsd"])

    list_of_matrices = ["Kpp", "KbT"]

    # ----- mesh data: bulk (matrix) mesh, or the fracture mesh on the recursive pass
    (
        nn,
        nsd,
        conn,
        nne_list,
        etype_list,
        elem_indx,
        nq_list,
        region_lists,
        num_regions,
        fmesh,
        nodes_f,
    ) = _get_region_mesh_data(mesh, is_fracture)

    if is_fracture:
        conn_global = fmesh["conn_global"]

        # Connectivity-only (mesh-static) data, cached by integrate_linear_matrices's
        # fracture pass rather than re-derived here every timestep. `integrate_NLpressure_matrices`
        # is only ever called for nonlinear viscosity, so this cache is unconditionally available.
        cv = fmesh["vec_mat_asmbl"]
        row_vec_ff, col_vec_ff = cv.row_vec_ff, cv.col_vec_ff
        row_vec_fm, col_vec_fm = cv.row_vec_fm, cv.col_vec_fm
        row_vec_mm, col_vec_mm = cv.row_vec_mm, cv.col_vec_mm

        vec_ff = {"Kpp_ff": np.zeros(row_vec_ff.size)}
        vec_fm = {"Kpp_fm": np.zeros(row_vec_fm.size)}
        vec_mm = {"Kpp_mm": np.zeros(row_vec_mm.size)}

        vec_counter_ff = 0
        vec_counter_fm = 0
        vec_counter_mm = 0

    vec = _initiate_vectors(list_of_matrices, nne_list, elem_indx)
    vec_counter = 0  # vectorized storage index counter

    # ----- Matrix integration element loop
    for r in range(num_regions):
        kf, isotropic_perm, nodes_r, w, u, kf_p, _ = _get_region_flow_data(
            mesh, mat, fmesh, nodes_f, is_fracture, r
        )

        # find element type of first element in region
        elem_type_counter = -1  # force an update on the first element

        for e in region_lists[r]:
            new_type_idx = bisect.bisect_left(elem_indx, e)
            if new_type_idx != elem_type_counter:
                # Update for different element type
                # Evaluate quadrature points and get weights and shape functions.
                W, nq, N_q, GN_q = _get_quad_points_and_shape_functions(
                    nq_list[new_type_idx], nsd, etype_list[new_type_idx]
                )
                nne = nne_list[new_type_idx]
                elem_type_counter = new_type_idx

            enodes = conn[e]  # local element nodes
            enodes_dof = conn_global[e] if is_fracture else enodes  # global DoF index, for T
            xI = nodes_r[enodes, :]
            B_q, detJe_q = getB(GN_q, xI, nsd)
            wJe_q = W * detJe_q  # Weight * det(Jacobian), used in every integrand
            if axisymmetry:
                # Add radial component to integrand for axisymmetry
                r_coord = N_q @ xI[:, 0]
                wJe_q = wJe_q * r_coord.T

            # vectorized integration
            Te = T[enodes_dof].reshape(-1, 1)  # element nodal temperatures
            T_q = N_q @ Te  # quadrature point temperatures
            mu_q, _ = get_viscosity(T_q, mu_coeff)  # quadrature point viscosities

            kf_b = (
                ng.T @ np.dot(kf, ng) / mu_q * alpha_f * rho_f @ (g.T @ u if is_fracture else g.T)
            )  # kf component in gravity direction

            element_matrices = _integrate_pressure_flow_terms(
                N_q, B_q, wJe_q, kf, mu_q, kf_b, isotropic_perm, is_fracture
            )

            # ----- Fracture-matrix pressure coupling
            if is_fracture:
                # Coupling matrices
                coupling_entries = _integrate_fracture_matrix_coupling(
                    e,
                    enodes,
                    nodes_f,
                    N_q,
                    wJe_q,
                    mesh,
                    kf_p=kf_p,
                    mu_q=mu_q,
                )
                for nne_m, ff_matrices, fm_matrices, mm_matrices in coupling_entries:
                    stop = vec_counter_ff + nne * nne  # vectorized index count end
                    _store_vectorized_element_matrix(vec_ff, ff_matrices, vec_counter_ff, stop)
                    vec_counter_ff = stop
                    stop = vec_counter_fm + nne_m * nne  # vectorized index count end
                    _store_vectorized_element_matrix(vec_fm, fm_matrices, vec_counter_fm, stop)
                    vec_counter_fm = stop
                    stop = vec_counter_mm + nne_m * nne_m  # vectorized index count end
                    _store_vectorized_element_matrix(vec_mm, mm_matrices, vec_counter_mm, stop)
                    vec_counter_mm = stop

            # ----- Assemble element contributions into global matrices
            stop = vec_counter + nne * nne  # vectorized index count end
            _store_vectorized_element_matrix(vec, element_matrices, vec_counter, stop)
            vec_counter = stop

    # ----- Return and convert to sparse matrix
    integrated_matrices = _assemble_global_matrices(
        vec, conn, nn, nne_list, elem_indx, np.concatenate(region_lists)
    )

    if is_fracture:
        # Assemble the coupling matrices
        nn_m = mesh["nn"]
        for name in vec_ff:
            integrated_matrices[name] = _make_sparse(vec_ff[name], row_vec_ff, col_vec_ff, nn, nn)
        for name in vec_fm:
            integrated_matrices[name] = _make_sparse(vec_fm[name], row_vec_fm, col_vec_fm, nn_m, nn)
        for name in vec_mm:
            integrated_matrices[name] = _make_sparse(vec_mm[name], row_vec_mm, col_vec_mm, nn_m, nn_m)
        return integrated_matrices

    if fracture_on:
        # Recursion call
        integrated_fracture_matrices = integrate_NLpressure_matrices(
            mesh, mat, primary_variables, controls, is_fracture=True
        )
        # Combine fracture and matrix matrices into one
        integrated_matrices = _stack_fracture_matrices(
            integrated_matrices, integrated_fracture_matrices, nn, mesh["fracture_mesh"]["nn"]
        )

    return integrated_matrices


def integrate_boundary_flux_matrices(
    mesh: dict[str, Any], conn_face: list[list[int]], node_order: Sequence[int], controls: dict[str, Any]
) -> dict[str, csc_matrix[Any]]:
    """Integrate lower dimensional boundary integrals for flux BCs."""
    ne = np.size(conn_face, 0)
    nne = np.size(conn_face, 1)
    nsd = mesh["nsd"] - 1
    nodes = mesh["nodes"]  # (nn,  nsd) - nodal degrees of freedom
    etype = mesh["element_type"]
    axisymmetry = bool(controls["axisymmetric_on"])
    elem_indx = [ne - 1]
    nne_list = [nne]

    # ----- Local face numbering
    # Map each global node id in conn_face to a local index.
    # Used for assembly.
    face_node_ids = np.unique(np.asarray(conn_face, dtype=np.int32))
    global_to_local = {int(g): i for i, g in enumerate(face_node_ids)}
    nn_face = len(face_node_ids)

    conn_face_local = [[global_to_local[int(n)] for n in enodes] for enodes in conn_face]
    # conn_face_local is used for assembly then mapped back to global

    match etype[-1]:
        case "B8":
            etype = "Q4"
            nq = 2
        case "T4":
            etype = "T3"
            nq = 1
        case "T3":
            etype = "L2"
            nq = 2
        case "Q4":
            etype = "L2"
            nq = 2
        case _:
            raise NotImplementedError(f"Face elements of {etype}.")

    # ----- Vectorized matrix assembly
    # Assemble lists of rows, columns, and entries and then assemble the sparse
    # matrix all at once.
    # For the scalar pressure and temperature, there are nne * nne entries
    # per element.
    list_of_matrices = ["A"]
    vec = _initiate_vectors(list_of_matrices, nne_list, elem_indx)
    vec_counter = 0

    # Matrices to assemble:
    # A = area weight for flux distribution over the face
    integrand = np.asarray(1.0)

    # Evaluate quadrature points and get weights and shape functions.
    W, nq, N_q, GN_q = _get_quad_points_and_shape_functions(nq, nsd, etype)

    # ----- Matrix integration element loop
    for e in range(ne):
        enodes = conn_face[e]  # global node ids, used for geometry

        # calculate face normal
        xI = nodes[enodes, :]
        if nsd == 2:
            n = np.cross(xI[1, :] - xI[0, :], xI[2, :] - xI[0, :])
        else:
            n = np.array([-(xI[1, 1] - xI[0, 1]), xI[1, 0] - xI[0, 0]])

        # Find the in-plane coordinates - assumes faces are aligned with cartesian axes
        ax = np.argmax(abs(n))
        xI = np.delete(xI, ax, 1)

        detJe_q = getB(GN_q, xI, nsd)[1]
        wJe_q = W * detJe_q  # Weight * det(Jacobian), used in every integrand
        if axisymmetry:
            # Add radial component to integrand for axisymmetry
            r_coord = N_q @ nodes[enodes, 0]
            wJe_q = wJe_q * r_coord.T

        # Element matrices
        element_matrices = _initialize_element_matrices(list_of_matrices, nne, nne)
        # Boundary faces may be misordered relative to their global position,
        # relative in negative jacobians.
        # This shouldn't be an issue because all elements are defined with the same
        # relative numbering. Flux directions can then be controlled via
        # the sign of the applied traction.
        element_matrices["A"] = _vectorized_quad_point_integration(N_q, integrand, N_q, abs(wJe_q), True)

        # ----- Assemble element contributions into global matrices
        stop = vec_counter + nne * nne  # vectorized index count end
        _store_vectorized_element_matrix(vec, element_matrices, vec_counter, stop)
        vec_counter = stop

    matrices = _assemble_global_matrices(
        vec, conn_face_local, nn_face, nne_list, elem_indx, np.arange(0, ne, dtype=np.int32)
    )
    # Map local ordering back to global ordering using the input node_order
    local_idx = np.searchsorted(face_node_ids, np.asarray(node_order, dtype=np.int32))
    return {name: mat[local_idx, :][:, local_idx] for name, mat in matrices.items()}
