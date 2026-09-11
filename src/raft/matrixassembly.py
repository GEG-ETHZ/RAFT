"""Assembles global linear FEM matrices."""

from __future__ import annotations

import bisect
import warnings
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.sparse import csc_matrix, hstack, vstack

from .matprops import (
    EnthalpyResult,
    PropertyResult,
    get_density,
    get_effective_permeability,
    get_enthalpy,
    get_internal_energy,
    get_porosity_lagrangian,
    get_viscosity,
)

# from .parseboundaryconditions import allocate_WB_BCs
from .shapefunctions import GNMATRIX_FUNCS, NMATRIX_FUNCS, getB, getquadrature


def _get_gravity(nsd: int, magnitude: float) -> npt.NDArray[np.float64]:
    ng = np.zeros((nsd, 1))
    ng[nsd - 1] = 1.0  # gravity normal
    return magnitude * ng


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


def _get_region_value(val: Any, r: int) -> Any:
    """Return value for region, handling both scalar and list cases.

    Fallback for homogeneous cases defined as scalars instead of lists.
    """
    return val[r] if isinstance(val, list) else val


def _assemble_global_matrices(
    vec: dict[str, npt.NDArray[np.float64]],
    conn: list[list[int]],
    nn: int,
    nne_list: list[int],
    elem_indx: list[int],
    elem_order: npt.NDArray[np.int32],  # sequence the values were written
    isvector: bool = False,
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

            if isvector:
                # Vector assembly: 1 entry per node per element
                row_chunks.append(conn_run.ravel())
                col_chunks.append(np.zeros(conn_run.size, dtype=np.int32))
            else:
                # Matrix assembly: nne^2 entries per element
                row_chunks.append(np.repeat(conn_run, nne, axis=1).ravel())
                col_chunks.append(np.tile(conn_run, (1, nne)).ravel())

            start = i

    row_vec = np.concatenate(row_chunks)
    col_vec = np.concatenate(col_chunks)
    nn2 = 1 if isvector else nn

    return {name: _make_sparse(vec[name], row_vec, col_vec, nn, nn2) for name in vec}


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
    list_of_matrices: list[str], nne: list[int], elem_indx: list[int], ismatrix: bool = True
) -> dict[str, npt.NDArray[np.float64]]:
    """Create the dictionary of vectorized matrix entries."""
    # calculate the number of entries
    prev = -1
    total_entries = 0
    exponent = 2 if ismatrix else 1

    for n, idx in zip(nne, elem_indx):
        ne_i = idx - prev
        total_entries += n**exponent * ne_i
        prev = idx
    return {name: np.zeros(total_entries) for name in list_of_matrices}


def _accumulate_element_matrices(
    element_matrices: dict[str, npt.NDArray[np.float64]],
    new_terms: dict[str, npt.NDArray[np.float64]],
) -> None:
    """Sum each term onto `element_matrices` in place instead of overwriting (unlike `dict.update`)."""
    for name, term in new_terms.items():
        if name in element_matrices:
            element_matrices[name] = element_matrices[name] + term
        else:
            element_matrices[name] = term


def _store_vctrzd_element_matrix(
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


def _vctrzd_quad_point_integration(
    left_matrix: npt.NDArray[np.float64],
    integrand: npt.NDArray[np.float64] | float,
    right_matrix: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    scalar: bool,
) -> npt.NDArray[np.float64]:
    """Perform vectorized matrix integration over quadrature points.

    left_matrix, right_matrix: nq x a/b x nne. integrand: a x b or nq x a x b.
    """
    if scalar:
        integrand = np.asarray(integrand, dtype=np.float64)
    left_T = left_matrix.transpose(0, 2, 1)

    # choose between multiply types for scalar vs. non-scalar material properties
    middle = integrand * right_matrix if scalar else integrand @ right_matrix

    K = left_T @ middle * weights.reshape(-1, 1, 1)

    result: npt.NDArray[np.float64] = K.sum(0)
    return result


def _vctrzd_qp_vctr_integrate(
    left_matrix: npt.NDArray[np.float64],
    integrand: npt.NDArray[np.float64] | float,
    weights: npt.NDArray[np.float64],
    scalar: bool,
) -> npt.NDArray[np.float64]:
    """Perform vectorized vector integration over quadrature points.

    left_matrix: nq x a x nne. integrand: a x 1 or nq x a x 1.
    """
    if scalar:
        integrand = np.asarray(integrand, dtype=np.float64)
    left_T = left_matrix.transpose(0, 2, 1)

    K = left_T @ integrand * weights.reshape(-1, 1, 1)

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
    int,
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    int,
    npt.NDArray[np.int32],
]:
    """Create the row and column vectors for vectorized fracture coupling matrix assembly."""
    count_vec_ff = 0  # Fracture-fracture on-diagonal counts
    row_list_ff = []
    col_list_ff = []

    count_vec_fm = 0  # Fracture-matrix off-diagonal counts (row=matrix, col=fracture)
    row_list_fm = []
    col_list_fm = []

    count_vec_mf = 0  # Matrix-fracture off-diagonal counts (row=fracture, col=matrix)
    row_list_mf = []
    col_list_mf = []

    count_vec_mm = 0  # Matrix-matrix on-diagonal counts
    row_list_mm = []
    col_list_mm = []

    count_vec_flux_m = 0  # Matrix-side scatter target for the coupling *residual* (a vector, so
    row_list_flux_m = []  # one row index per matrix node per match - no column/nne_m^2 term)

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
            # mf
            row_list_mf.append(
                np.repeat(np.array(e_nodes_f, dtype=np.int32).reshape(-1, 1), nne_m, axis=1).ravel()
            )
            col_list_mf.append(np.tile(np.array(e_nodes_m, dtype=np.int32), (1, nne_f)).ravel())
            count_vec_mf += nne_f * nne_m
            # mm
            row_list_mm.append(
                np.repeat(np.array(e_nodes_m, dtype=np.int32).reshape(-1, 1), nne_m, axis=1).ravel()
            )
            col_list_mm.append(np.tile(np.array(e_nodes_m, dtype=np.int32), (1, nne_m)).ravel())
            count_vec_mm += nne_m * nne_m
            # flux (residual, matrix-side)
            row_list_flux_m.append(np.array(e_nodes_m, dtype=np.int32))
            count_vec_flux_m += nne_m

    row_vec_ff = np.concatenate(row_list_ff).astype(np.int32)
    col_vec_ff = np.concatenate(col_list_ff).astype(np.int32)
    row_vec_fm = np.concatenate(row_list_fm).astype(np.int32)
    col_vec_fm = np.concatenate(col_list_fm).astype(np.int32)
    row_vec_mf = np.concatenate(row_list_mf).astype(np.int32)
    col_vec_mf = np.concatenate(col_list_mf).astype(np.int32)
    row_vec_mm = np.concatenate(row_list_mm).astype(np.int32)
    col_vec_mm = np.concatenate(col_list_mm).astype(np.int32)
    row_vec_flux_m = np.concatenate(row_list_flux_m).astype(np.int32)

    return (
        count_vec_ff,
        row_vec_ff,
        col_vec_ff,
        count_vec_fm,
        row_vec_fm,
        col_vec_fm,
        count_vec_mf,
        row_vec_mf,
        col_vec_mf,
        count_vec_mm,
        row_vec_mm,
        col_vec_mm,
        count_vec_flux_m,
        row_vec_flux_m,
    )


def _iter_fracture_matrix_matches(
    e: int,
    enodes_f: list[int],
    nodes_f: npt.NDArray[np.float64],
    N_q: npt.NDArray[np.float64],
    mesh: dict[str, Any],
) -> Iterator[tuple[int, list[int], int, npt.NDArray[np.float64]]]:
    """Yield `(e_m, enodes_m, nne_m, N_qm)` for each matrix element matched to fracture element `e`.

    `N_qm` is the matrix element's shape functions evaluated at the fracture quadrature points.
    """
    conn_m = mesh["conn"]
    nodes_m = mesh["nodes"]
    etype_list_m = mesh["element_type"]
    elem_indx_m = mesh["elem_indx"]
    conn_fm = mesh["fracture_mesh"]["matrix_conn"]

    for e_m in conn_fm[e]:
        enodes_m = conn_m[e_m]
        nne_m = len(enodes_m)

        etype = etype_list_m[bisect.bisect_left(elem_indx_m, e_m)]
        getN_m = NMATRIX_FUNCS[etype]
        Q = _get_parent_coordinates(etype, nodes_m[enodes_m, :], N_q @ nodes_f[enodes_f, :])
        N_qm = np.array([getN_m(Q[q, :].flatten()) for q in range(np.size(Q, -1))])

        yield e_m, enodes_m, nne_m, N_qm


def _integrate_fracture_matrix_coupling_tangent(
    e: int,
    enodes_f: list[int],
    nodes_f: npt.NDArray[np.float64],
    N_q: npt.NDArray[np.float64],
    wJe_q: npt.NDArray[np.float64],
    mesh: dict[str, Any],
    fluid: dict[str, Any],
    pe_f: npt.NDArray[np.float64],  # Fracture pressure at quad points
    pe_f_lag: npt.NDArray[np.float64],  # Fracture pressure, lagged - upwind decision only
    Te_f: npt.NDArray[np.float64],  # Fracture temperature at quad points
    Te_flow_f: npt.NDArray[np.float64],  # Fracture flow-property temperature, pre-sliced by caller
    p: npt.NDArray[np.float64],  # Full pressure vector
    p_lag: npt.NDArray[np.float64],  # Full pressure vector, lagged - upwind decision only
    T: npt.NDArray[np.float64],  # Full temperature vector
    T_flow: npt.NDArray[np.float64],  # Temperature vector for property evaluation
    kf_perp: PropertyResult,
    mu_q: PropertyResult,  # viscosity at quad points
    rhof_q: PropertyResult,  # density at quad points
    hc: float,
    *,
    pressure_coupling: bool,
    thermal_coupling: bool,
    fully_coupled: bool,
    advection_on: bool,
    T_flow_lagged: bool,
) -> list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Integrate one fracture element's coupling tangent contribution to its neighboring matrix elements."""
    entries: list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]] = []

    p_f_q = N_q @ pe_f  # fracture pressure
    p_f_q_lag = N_q @ pe_f_lag  # lagged fracture pressure, upwind decision only

    for _e_m, enodes_m, nne_m, N_qm in _iter_fracture_matrix_matches(e, enodes_f, nodes_f, N_q, mesh):
        ff_matrices: dict[str, npt.NDArray[np.float64]] = {}
        fm_matrices: dict[str, npt.NDArray[np.float64]] = {}
        mf_matrices: dict[str, npt.NDArray[np.float64]] = {}
        mm_matrices: dict[str, npt.NDArray[np.float64]] = {}

        if pressure_coupling or advection_on:
            p_m_q = N_qm @ p[enodes_m]  # matrix pressure
            flux_p_q, dfluxp_dpf, dfluxp_dpm = _evaluate_fracture_leakoff_flux(
                rhof_q, kf_perp, mu_q, p_f_q, p_m_q, want_tangent_dp=True
            )

        if pressure_coupling:
            ff_matrices["Kpp_ff"] = _vctrzd_quad_point_integration(N_q, -dfluxp_dpf, N_q, wJe_q, True)
            fm_matrices["Kpp_fm"] = _vctrzd_quad_point_integration(N_qm, -dfluxp_dpm, N_q, wJe_q, True)
            mf_matrices["Kpp_mf"] = _vctrzd_quad_point_integration(N_q, dfluxp_dpf, N_qm, wJe_q, True)
            mm_matrices["Kpp_mm"] = _vctrzd_quad_point_integration(N_qm, dfluxp_dpm, N_qm, wJe_q, True)

        if thermal_coupling:
            T_f_q = N_q @ Te_f
            T_m_q = N_qm @ T[enodes_m]
            _, dfluxc_dTf, dfluxc_dTm = _evaluate_fracture_conduction_flux(
                hc, T_f_q, T_m_q, want_tangent_dT=True
            )

            ff_matrices["KTT_ff"] = _vctrzd_quad_point_integration(N_q, dfluxc_dTf, N_q, wJe_q, True)
            fm_matrices["KTT_fm"] = _vctrzd_quad_point_integration(N_qm, dfluxc_dTm, N_q, wJe_q, True)
            mf_matrices["KTT_mf"] = _vctrzd_quad_point_integration(N_q, -dfluxc_dTf, N_qm, wJe_q, True)
            mm_matrices["KTT_mm"] = _vctrzd_quad_point_integration(N_qm, -dfluxc_dTm, N_qm, wJe_q, True)

            if advection_on:
                T_flow_fq = N_q @ Te_flow_f
                T_flow_mq = N_qm @ T_flow[enodes_m]
                p_m_q_lag = N_qm @ p_lag[enodes_m]  # upwind decision only

                _, dqadv_dTf, dqadv_dTm, dqadv_dpf, dqadv_dpm = _evaluate_fracture_advection_flux(
                    p_f_q,
                    p_m_q,
                    p_f_q_lag,
                    p_m_q_lag,
                    T_f_q,
                    T_m_q,
                    T_flow_fq,
                    T_flow_mq,
                    flux_p_q,
                    fluid,
                    kf_perp,
                    mu_q,
                    rhof_q,
                    dfluxp_dpf,
                    dfluxp_dpm,
                    want_tangent=True,
                    T_flow_lagged=T_flow_lagged,
                )

                # fm/mf coefficients use the derivative wrt the column's temperature (Tf for
                # fm, Tm for mf), not the row's own.
                ff_matrices["KTT_ff"] = ff_matrices["KTT_ff"] + _vctrzd_quad_point_integration(
                    N_q, -dqadv_dTf, N_q, wJe_q, True
                )
                fm_matrices["KTT_fm"] = fm_matrices["KTT_fm"] + _vctrzd_quad_point_integration(
                    N_qm, dqadv_dTf, N_q, wJe_q, True
                )
                mf_matrices["KTT_mf"] = mf_matrices["KTT_mf"] + _vctrzd_quad_point_integration(
                    N_q, -dqadv_dTm, N_qm, wJe_q, True
                )
                mm_matrices["KTT_mm"] = mm_matrices["KTT_mm"] + _vctrzd_quad_point_integration(
                    N_qm, dqadv_dTm, N_qm, wJe_q, True
                )

                if fully_coupled:
                    ff_matrices["KTp_ff"] = _vctrzd_quad_point_integration(N_q, -dqadv_dpf, N_q, wJe_q, True)
                    fm_matrices["KTp_fm"] = _vctrzd_quad_point_integration(N_qm, dqadv_dpf, N_q, wJe_q, True)
                    mf_matrices["KTp_mf"] = _vctrzd_quad_point_integration(N_q, -dqadv_dpm, N_qm, wJe_q, True)
                    mm_matrices["KTp_mm"] = _vctrzd_quad_point_integration(N_qm, dqadv_dpm, N_qm, wJe_q, True)

        entries.append((nne_m, ff_matrices, fm_matrices, mf_matrices, mm_matrices))

    return entries


def _integrate_fracture_matrix_coupling_residual(  # noqa
    e: int,  # fracture element
    enodes_f: list[int],  # fracture element connectivity
    nodes_f: npt.NDArray[np.float64],  # fracture element nodes
    N_q: npt.NDArray[np.float64],  # fracture shape function
    wJe_q: npt.NDArray[np.float64],
    mesh: dict[str, Any],
    fluid: dict[str, Any],
    pe_f: npt.NDArray[np.float64],  # Fracture pressure at quad points
    pe_f_lag: npt.NDArray[np.float64],  # Fracture pressure, lagged - upwind decision only
    Te_f: npt.NDArray[np.float64],  # Fracture temperature at quad points
    Te_flow_f: npt.NDArray[np.float64],  # Fracture flow-property temperature, pre-sliced by caller
    p: npt.NDArray[np.float64],  # Full pressure vector
    p_lag: npt.NDArray[np.float64],  # Full pressure vector, lagged - upwind decision only
    T: npt.NDArray[np.float64],  # Full temperature vector
    T_flow: npt.NDArray[np.float64],  # Temperature vector for property evaluation
    kf_perp: PropertyResult,  # fracture transverse permeability at quad points
    mu_q: PropertyResult,  # viscosity at quad points
    rhof_q: PropertyResult,  # density at quadrature points
    hc: float,  # convective heat transfer coefficient
    *,
    pressure_coupling: bool,
    thermal_coupling: bool,
    advection_on: bool,
) -> tuple[dict[str, npt.NDArray[np.float64]], list[tuple[int, dict[str, npt.NDArray[np.float64]]]]]:
    """Integrate the fracture-matrix leak-off/convective-exchange flux at the current state into
    nodal residual vectors.

    Returns `(fracture_vectors, matches)`: `fracture_vectors` accumulates onto this fracture
    element's own residual, and `matches` is a list of `(nne_m, matrix_vectors)`, one entry per
    matched matrix element.
    """
    fracture_vectors: dict[str, npt.NDArray[np.float64]] = {}
    matches: list[tuple[int, dict[str, npt.NDArray[np.float64]]]] = []
    if not pressure_coupling and not thermal_coupling:
        return fracture_vectors, matches

    p_f_q = N_q @ pe_f
    p_f_q_lag = N_q @ pe_f_lag  # lagged fracture pressure, upwind decision only
    T_f_q = N_q @ Te_f
    Rf_p = np.zeros((len(enodes_f), 1))
    Rf_T = np.zeros((len(enodes_f), 1))
    Rf_Tadv = np.zeros((len(enodes_f), 1))

    for _e_m, enodes_m, nne_m, N_qm in _iter_fracture_matrix_matches(e, enodes_f, nodes_f, N_q, mesh):
        matrix_vectors: dict[str, npt.NDArray[np.float64]] = {}
        if pressure_coupling or advection_on:
            p_m_q = N_qm @ p[enodes_m]  # matrix pressure
            flux_p_q, _, _ = _evaluate_fracture_leakoff_flux(rhof_q, kf_perp, mu_q, p_f_q, p_m_q)

        if pressure_coupling:
            Rf_p = Rf_p + _vctrzd_qp_vctr_integrate(N_q, flux_p_q, wJe_q, True)
            matrix_vectors["F_Pdiffusion_mm"] = _vctrzd_qp_vctr_integrate(N_qm, -flux_p_q, wJe_q, True)
        if thermal_coupling:
            T_m_q = N_qm @ T[enodes_m]  # matrix temperature
            flux_hc_T_q, _, _ = _evaluate_fracture_conduction_flux(hc, T_f_q, T_m_q)
            Rf_T = Rf_T + _vctrzd_qp_vctr_integrate(N_q, flux_hc_T_q, wJe_q, True)
            matrix_vectors["F_Tdiffusion_mm"] = _vctrzd_qp_vctr_integrate(N_qm, -flux_hc_T_q, wJe_q, True)

            if advection_on:
                T_flow_fq = N_q @ Te_flow_f
                T_flow_mq = N_qm @ T_flow[enodes_m]
                p_m_q_lag = N_qm @ p_lag[enodes_m]  # upwind decision only

                flux_adv_T_q, _, _, _, _ = _evaluate_fracture_advection_flux(
                    p_f_q,
                    p_m_q,
                    p_f_q_lag,
                    p_m_q_lag,
                    T_f_q,
                    T_m_q,
                    T_flow_fq,
                    T_flow_mq,
                    flux_p_q,
                    fluid,
                    kf_perp,
                    mu_q,
                    rhof_q,
                )
                Rf_Tadv = Rf_Tadv + _vctrzd_qp_vctr_integrate(N_q, flux_adv_T_q, wJe_q, True)
                matrix_vectors["F_Tadvection_mm"] = _vctrzd_qp_vctr_integrate(
                    N_qm, -flux_adv_T_q, wJe_q, True
                )

        matches.append((nne_m, matrix_vectors))

    if pressure_coupling:
        fracture_vectors["F_Pdiffusion"] = Rf_p
    if thermal_coupling:
        fracture_vectors["F_Tdiffusion"] = Rf_T
        if advection_on:
            fracture_vectors["F_Tadvection"] = Rf_Tadv
    return fracture_vectors, matches


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

        # Off-diagonal coupling matrix (row=matrix, col=fracture)
        if name + "_fm" in fracture_matrices:
            fm_matrix = fracture_matrices[name + "_fm"].tocoo()
        else:
            fm_matrix = csc_matrix((nn_m, nn_f)).tocoo()

        # Off-diagonal coupling matrix (row=fracture, col=matrix)
        if name + "_mf" in fracture_matrices:
            mf_matrix = fracture_matrices[name + "_mf"].tocoo()
        else:
            mf_matrix = csc_matrix((nn_f, nn_m)).tocoo()

        # on-diagonal correction to the matrix's own block
        matrix_matrix = body_matrices[name].tocoo()
        if name + "_mm" in fracture_matrices:
            matrix_matrix = matrix_matrix + fracture_matrices[name + "_mm"].tocoo()

        row1 = hstack([matrix_matrix.tocoo(), fm_matrix])
        row2 = hstack([mf_matrix, fracture_matrix])
        body_matrices[name] = vstack([row1, row2]).tocsc()
    return body_matrices


def _stack_fracture_vectors(
    body_vectors: dict[str, csc_matrix[Any]],
    fracture_vectors: dict[str, csc_matrix[Any]],
) -> dict[str, csc_matrix[Any]]:
    """Stack matrix-mesh and fracture-mesh residual vectors into one `(nn_m + nn_f, 1)` vector."""
    for name in body_vectors:
        body_vectors[name] = vstack([body_vectors[name], fracture_vectors[name]]).tocsc()
    return body_vectors


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


def _get_region_flow_data(
    mesh: dict[str, Any],
    material: dict[str, Any],
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
    PropertyResult,  # kf_perp, transverse permeability, coupling only; NaN sentinel when not is_fracture
]:
    """Resolve per-region permeability and node-projection data."""
    # Fracture case
    if is_fracture:
        w = material["fracture"]["aperture"][r]
        kf = w**3 / 12  # longitudinal permeability, cubic law
        kf_perp = w / 12  # transverse permeability, coupling only; constant, so zero derivatives
        u = fmesh["fracture_tangent"][r].reshape(-1, 1)
        nodes_r = nodes_f @ u  # project mesh nodes into local coordinates
        return kf, True, nodes_r, w, u, PropertyResult(kf_perp, 0.0, 0.0)

    # Bulk case
    kf = material["solid"]["permeability_eff"][r]
    isotropic_perm = bool(isinstance(kf, (float, int)))
    kf_perp_sentinel = PropertyResult(float("nan"), float("nan"), float("nan"))
    return kf, isotropic_perm, mesh["nodes"], float("nan"), np.full((1, 1), np.nan), kf_perp_sentinel


def _get_region_thermal_data(material: dict[str, Any], is_fracture: bool, r: int) -> tuple[bool, float]:
    """Resolve the region's isotropic-conductivity flag and interface heat transfer coefficient.

    `hc` is a NaN sentinel when not `is_fracture` (coupling-only quantity).
    """
    # Fracture case
    if is_fracture:
        w = material["fracture"]["aperture"][r]
        lambda_f = material["fluid"]["therm_cond"]
        nusselt = material["fracture"]["nusselt"][r]
        hc = nusselt * lambda_f / 2 / w  # interface heat transfer coefficient, coupling only
        return True, hc

    # Bulk case
    lambda_e = material["solid"]["therm_cond_eff"][r]
    isotropic_cond = bool(isinstance(lambda_e, (float, int)))
    return isotropic_cond, float("nan")


def _get_region_solid_data(
    material: dict[str, Any], r: int
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    """Resolve per-region solid properties."""
    K_d = _get_region_value(material["solid"]["drained_bulk_mod"], r)
    phi_0 = _get_region_value(material["solid"]["porosity"], r)
    biot = _get_region_value(material["solid"]["biot_coeff"], r)
    rho_s = _get_region_value(material["solid"]["density"], r)
    cp_s = _get_region_value(material["solid"]["specific_heat"], r)
    alpha_s = _get_region_value(material["solid"]["therm_expansion"], r)
    lambda_s = _get_region_value(material["solid"]["therm_cond"], r)

    return K_d, phi_0, biot, rho_s, cp_s, alpha_s, lambda_s


_YELLOW = "\033[93m"
_RESET = "\033[0m"


def warn_nonmonotone_pressure_diffusion(Kpp: csc_matrix[Any], rel_tol: float = 0.7) -> None:
    """Warn if the pressure stiffness matrix severely violates the discrete maximum principle."""
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


# ----------------------------------------------------------------------------------------
# Per-element physics evaluators shared between integrate_residuals and
# integrate_tangent_matrices. Each one is called identically by both - the tangent function
# just also keeps the derivative outputs the residual discards - except where noted.
# ----------------------------------------------------------------------------------------

# Sentinel for a tangent-only input/output not needed when want_tangent_*=False.
_EMPTY = np.empty((0, 0))


def _evaluate_fracture_leakoff_flux(
    rhof_q: PropertyResult,
    kf_perp: PropertyResult,
    mu_q: PropertyResult,
    p_f_q: npt.NDArray[np.float64],
    p_m_q: npt.NDArray[np.float64],
    *,
    want_tangent_dp: bool = False,
) -> tuple[
    npt.NDArray[np.float64],  # flux_p_q
    npt.NDArray[np.float64],  # dfluxp_dpf, `_EMPTY` unless want_tangent_dp
    npt.NDArray[np.float64],  # dfluxp_dpm, `_EMPTY` unless want_tangent_dp
]:
    """Evaluate the fracture-matrix leak-off mass flux and optionally its pressure tangent."""
    flux_coeff = rhof_q.value * kf_perp.value / mu_q.value  # leak-off conductance
    flux_p_q = flux_coeff * (p_f_q - p_m_q)

    dfluxp_dpf = dfluxp_dpm = _EMPTY
    if want_tangent_dp:
        dcoeff_dp = rhof_q.dp * kf_perp.value + rhof_q.value * kf_perp.dp - flux_coeff * mu_q.dp
        dfluxp_dpf = flux_coeff + dcoeff_dp / mu_q.value * (p_f_q - p_m_q)
        dfluxp_dpm = -flux_coeff

    return flux_p_q, dfluxp_dpf, dfluxp_dpm


def _evaluate_fracture_conduction_flux(
    hc: float,
    T_f_q: npt.NDArray[np.float64],
    T_m_q: npt.NDArray[np.float64],
    *,
    want_tangent_dT: bool = False,
) -> tuple[
    npt.NDArray[np.float64],  # flux_hc_T_q
    npt.NDArray[np.float64] | float,  # dfluxc_dTf, `_EMPTY` unless want_tangent_dT
    npt.NDArray[np.float64] | float,  # dfluxc_dTm, `_EMPTY` unless want_tangent_dT
]:
    """Evaluate the fracture-matrix convective conduction flux and optionally its temperature tangent."""
    flux_hc_T_q = hc * (T_f_q - T_m_q)
    dfluxc_dTf: npt.NDArray[np.float64] | float = _EMPTY
    dfluxc_dTm: npt.NDArray[np.float64] | float = _EMPTY
    if want_tangent_dT:
        dfluxc_dTf = hc
        dfluxc_dTm = -hc
    return flux_hc_T_q, dfluxc_dTf, dfluxc_dTm


def _evaluate_fracture_advection_flux(
    p_f_q: npt.NDArray[np.float64],
    p_m_q: npt.NDArray[np.float64],
    p_f_q_lag: npt.NDArray[np.float64],
    p_m_q_lag: npt.NDArray[np.float64],
    T_f_q: npt.NDArray[np.float64],
    T_m_q: npt.NDArray[np.float64],
    T_flow_fq: npt.NDArray[np.float64],
    T_flow_mq: npt.NDArray[np.float64],
    flux_p_q: npt.NDArray[np.float64],
    fluid: dict[str, Any],
    kf_perp: PropertyResult,
    mu_q: PropertyResult,
    rhof_q: PropertyResult,
    dfluxp_dpf: npt.NDArray[np.float64] = _EMPTY,
    dfluxp_dpm: npt.NDArray[np.float64] = _EMPTY,
    *,
    want_tangent: bool = False,
    T_flow_lagged: bool = False,
) -> tuple[
    npt.NDArray[np.float64],  # flux_adv_T_q
    npt.NDArray[np.float64],  # dqadv_dTf, `_EMPTY` unless want_tangent
    npt.NDArray[np.float64],  # dqadv_dTm, `_EMPTY` unless want_tangent
    npt.NDArray[np.float64],  # dqadv_dpf, `_EMPTY` unless want_tangent
    npt.NDArray[np.float64],  # dqadv_dpm, `_EMPTY` unless want_tangent
]:
    """Evaluate the fracture-matrix advective enthalpy flux and optionally its temperature/pressure tangents.

    Upwind direction is picked from the lagged pressures `p_f_q_lag`/`p_m_q_lag`.
    """
    fracture_upwind = np.mean(p_f_q_lag) > np.mean(p_m_q_lag)
    if fracture_upwind:
        T_eval, T_flow_eval, p_eval = T_f_q, T_flow_fq, p_f_q
    else:
        T_eval, T_flow_eval, p_eval = T_m_q, T_flow_mq, p_m_q

    hf_q = get_enthalpy(p_eval, T_eval, T_flow_eval, fluid, T_flow_lagged)
    flux_adv_T_q = hf_q.value * flux_p_q

    dqadv_dTf = dqadv_dTm = dqadv_dpf = dqadv_dpm = _EMPTY
    if want_tangent:
        flux_coeff = rhof_q.value * kf_perp.value / mu_q.value  # same conductance as leak-off flux
        dcoeff_dT = rhof_q.dT * kf_perp.value + rhof_q.value * kf_perp.dT - flux_coeff * mu_q.dT
        if fracture_upwind:
            dqadv_dTf = hf_q.dT * flux_p_q + hf_q.value * dcoeff_dT / mu_q.value * (p_f_q - p_m_q)
            dqadv_dTm = np.zeros_like(dqadv_dTf)
            dqadv_dpf = hf_q.dp * flux_p_q + hf_q.value * dfluxp_dpf
            dqadv_dpm = hf_q.value * dfluxp_dpm
        else:
            dqadv_dTf = hf_q.value * dcoeff_dT / mu_q.value * (p_f_q - p_m_q)
            dqadv_dTm = hf_q.dT * flux_p_q
            dqadv_dpf = hf_q.value * dfluxp_dpf
            dqadv_dpm = hf_q.dp * flux_p_q + hf_q.value * dfluxp_dpm

    return flux_adv_T_q, dqadv_dTf, dqadv_dTm, dqadv_dpf, dqadv_dpm


def _evaluate_element_kinematics(
    N_q: npt.NDArray[np.float64],
    B_q: npt.NDArray[np.float64],
    xI: npt.NDArray[np.float64],
    T: npt.NDArray[np.float64],
    T_flow: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    T_0: npt.NDArray[np.float64],
    p_0: npt.NDArray[np.float64],
    enodes_dof: npt.NDArray[np.int32] | list[int],
) -> tuple[
    npt.NDArray[np.float64],  # Te
    npt.NDArray[np.float64],  # Te_flow
    npt.NDArray[np.float64],  # pe
    npt.NDArray[np.float64],  # T_0e
    npt.NDArray[np.float64],  # p_0e
    npt.NDArray[np.float64],  # L_vec
    npt.NDArray[np.float64],  # T_q
    npt.NDArray[np.float64],  # T_flow_q
    npt.NDArray[np.float64],  # p_q
    npt.NDArray[np.float64],  # gradT
    npt.NDArray[np.float64],  # gradP
]:
    """Slice this element's nodal primary variables and interpolate them to quadrature points."""
    Te = T[enodes_dof]  # element nodal temperatures
    Te_flow = T_flow[enodes_dof]  # element nodal temperatures, for flow properties
    pe = p[enodes_dof]  # element nodal pressures
    T_0e = T_0[enodes_dof].reshape(-1, 1)  # element reference temperatures
    p_0e = p_0[enodes_dof].reshape(-1, 1)  # element reference pressures
    L_vec = (xI.max(axis=0) - xI.min(axis=0)).reshape(-1, 1)  # element lengths

    T_q = N_q @ Te  # quadrature point temperatures, nq x 1 x (n, n-1, n-2, n-3)
    T_flow_q = N_q @ Te_flow  # quadrature point temperatures, for flow properties
    p_q = N_q @ pe  # quadrature point pressures, nq x 1 x (n, n-1, n-2, n-3)
    gradT = B_q @ Te[:, 0:1]  # qudarature point temperature gradient, nq x nsd x 1
    gradP = B_q @ pe[:, 0:1]  # quadrature point pressure gradient, nq x nsd x 1

    return Te, Te_flow, pe, T_0e, p_0e, L_vec, T_q, T_flow_q, p_q, gradT, gradP


def _evaluate_flow_properties(
    kf_0: npt.NDArray[np.float64] | float,
    T_flow_q_n: npt.NDArray[np.float64],
    p_q_n: npt.NDArray[np.float64],
    T_flow_q: npt.NDArray[np.float64],
    p_q: npt.NDArray[np.float64],
    fluid: dict[str, Any],
    *,
    T_flow_lagged: bool = False,
) -> tuple[PropertyResult, PropertyResult, PropertyResult]:
    """Evaluate permeability, viscosity, density and their derivatives: `(kf, mu_q, rho_f)`.

    Viscosity and permeability are evaluated at the current time. The returned `rho_f.value`
    carries the full time history (for the rate evaluation), while `rho_f.dp`/`rho_f.dT` are
    sliced to the current iterate only.
    """
    kf = get_effective_permeability(kf_0, T_flow_q_n, p_q_n, fluid, T_flow_lagged)
    mu_q = get_viscosity(T_flow_q_n, p_q_n, fluid, T_flow_lagged)
    rho_f = get_density(T_flow_q, p_q, fluid, T_flow_lagged)
    rho_f = PropertyResult(rho_f.value, rho_f.dp[:, :, 0:1], rho_f.dT[:, :, 0:1])
    return kf, mu_q, rho_f


def _evaluate_darcy_flux(
    kf: PropertyResult,
    mu_q: PropertyResult,
    gradP: npt.NDArray[np.float64],
    rho_f_n: PropertyResult,
    g_r: npt.NDArray[np.float64],
    isotropic_perm: bool,
    N_q: npt.NDArray[np.float64],
    B_q: npt.NDArray[np.float64],
    *,
    want_tangent_dp: bool = False,
    want_tangent_dT: bool = False,
) -> tuple[
    npt.NDArray[np.float64],  # darcy_flux
    npt.NDArray[np.float64],  # q_D
    npt.NDArray[np.float64],  # dqD_dp, `_EMPTY` unless want_tangent_dp
    npt.NDArray[np.float64],  # dqD_dT, `_EMPTY` unless want_tangent_dT
]:
    """Evaluate the Darcy flux and derivatives at quadrature points."""
    gradP_mod = gradP - rho_f_n.value * g_r
    kfmu = kf.value / mu_q.value
    darcy_flux = -kfmu * gradP_mod if isotropic_perm else -kfmu @ gradP_mod
    q_D = rho_f_n.value * darcy_flux

    dqD_dp = _EMPTY
    if want_tangent_dp:
        term_2 = (
            kfmu * rho_f_n.dp
            + rho_f_n.value / mu_q.value * kf.dp
            + rho_f_n.value * kf.value / mu_q.value**2 * mu_q.dp
        )
        if isotropic_perm:
            dqD_dp = rho_f_n.value * kfmu * (B_q + rho_f_n.dp * N_q) + term_2 * gradP_mod * N_q
        else:
            dqD_dp = rho_f_n.value * kfmu @ (B_q + rho_f_n.dp * N_q) + term_2 @ gradP_mod * N_q

    dqD_dT = _EMPTY
    if want_tangent_dT:
        term_2 = (
            kfmu * rho_f_n.dT
            + rho_f_n.value / mu_q.value * kf.dT
            + rho_f_n.value * kf.value / mu_q.value**2 * mu_q.dT
        )
        if isotropic_perm:
            dqD_dT = rho_f_n.value * kfmu * (rho_f_n.dT * g_r * N_q) + term_2 * gradP_mod * N_q
        else:
            dqD_dT = rho_f_n.value * kfmu @ (rho_f_n.dT * g_r * N_q) + term_2 @ gradP_mod * N_q

    return darcy_flux, q_D, dqD_dp, dqD_dT


def _evaluate_quad_porosity(
    p_q: npt.NDArray[np.float64],
    p_0e: npt.NDArray[np.float64],
    T_flow_q: npt.NDArray[np.float64],
    T_0e: npt.NDArray[np.float64],
    N_q: npt.NDArray[np.float64],
    phi_0: float,
    K_d: float,
    biot: float,
    alpha_s: float,
    is_fracture: bool,
    w0: float,
    nq: int,
    *,
    T_flow_lagged: bool = False,
    volumetric_strain_on: bool = True,
) -> PropertyResult:
    """Evaluate the Lagrangian porosity and derivatives at quadrature points.

    `.value` carries the full time history; `.dp`/`.dT` are sliced to the current iterate only.
    """
    if is_fracture:
        # Fracture case: phi is replaced by aperture, a fixed property independent of p/T.
        phi_L = w0 * np.ones_like(T_flow_q)
        return PropertyResult(phi_L, np.zeros((nq, 1, 1)), np.zeros((nq, 1, 1)))

    phi_L = get_porosity_lagrangian(
        p_q,
        N_q @ p_0e,
        T_flow_q,
        N_q @ T_0e,
        phi_0,
        K_d,
        biot,
        alpha_s,
        T_flow_lagged,
        volumetric_strain_on,
    )
    return PropertyResult(phi_L.value, phi_L.dp[:, :, 0:1], phi_L.dT[:, :, 0:1])


def _evaluate_nodal_state(
    pe: npt.NDArray[np.float64],
    p_0e: npt.NDArray[np.float64],
    Te: npt.NDArray[np.float64],
    Te_flow: npt.NDArray[np.float64],
    T_0e: npt.NDArray[np.float64],
    phi_0: float,
    K_d: float,
    biot: float,
    alpha_s: float,
    is_fracture: bool,
    w0: float,
    fluid: dict[str, Any],
    *,
    T_flow_lagged: bool = False,
    volumetric_strain_on: bool = True,
) -> tuple[PropertyResult, PropertyResult, PropertyResult]:
    """Evaluate porosity, density, and internal energy at this element's own nodes.

    Returns `(phi_L_e, rho_f_e, u_f_e)`. `u_f_e.dp` is `_EMPTY` — the pressure derivative of
    internal energy is only needed at quadrature points (see `_evaluate_energy_properties`).
    """
    if is_fracture:
        phi_L_e = PropertyResult(w0 * np.ones_like(pe), np.zeros_like(pe), np.zeros_like(pe))
    else:
        phi_L_e = get_porosity_lagrangian(
            pe, p_0e, Te_flow, T_0e, phi_0, K_d, biot, alpha_s, T_flow_lagged, volumetric_strain_on
        )
    rho_f_e = get_density(Te_flow, pe, fluid, T_flow_lagged)
    u_f_e_full = get_internal_energy(Te, pe, fluid)
    u_f_e = PropertyResult(u_f_e_full.value, _EMPTY, u_f_e_full.dT)
    return phi_L_e, rho_f_e, u_f_e


def _evaluate_total_energy(
    T: npt.NDArray[np.float64],
    phi_L: PropertyResult,
    rho_f: PropertyResult,
    u_f: PropertyResult,
    is_fracture: bool,
    rho_s: float,
    cp_s: float,
) -> npt.NDArray[np.float64]:
    """Evaluate the total (solid + fluid) stored energy."""
    return (1 - phi_L.value) * rho_s * cp_s * T * (1 - is_fracture) + phi_L.value * rho_f.value * u_f.value


def _evaluate_mass_storage_integrands(
    phi_L_e: PropertyResult,
    rho_f_e: PropertyResult,
    cv: npt.NDArray[np.float64],
    steady_pres: float,
    phi_L_n: PropertyResult = PropertyResult(_EMPTY, _EMPTY, _EMPTY),
    rho_f_n: PropertyResult = PropertyResult(_EMPTY, _EMPTY, _EMPTY),
    *,
    want_tangent_dp: bool = False,
    want_tangent_dT: bool = False,
) -> tuple[
    npt.NDArray[np.float64],  # mass_dot_e
    npt.NDArray[np.float64],  # integrand_dFsdp_e (Kpp), `_EMPTY` unless want_tangent_dp
    npt.NDArray[np.float64],  # integrand_dFsdT (KpT), `_EMPTY` unless want_tangent_dT
]:
    """Evaluate the mass storage rate (nodal, lumped) and optionally its pressure and temperature tangents.

    `phi_L_e`/`rho_f_e` are nodal (their `.dp` feeds the dp-tangent, matching Kpp's lumped
    discretization); `phi_L_n`/`rho_f_n` are quadrature current-iterate (their `.dT` feeds the
    dT-tangent, matching KpT's consistent discretization).
    """
    f_mass_e = phi_L_e.value * rho_f_e.value
    mass_dot_e = (1 - steady_pres) * (
        cv[0] * f_mass_e[:, 0:1] + cv[1] * f_mass_e[:, 1:2] + cv[2] * f_mass_e[:, 2:3]
    )

    integrand_dFsdp_e = _EMPTY
    if want_tangent_dp:
        integrand_dFsdp_e = (
            (1 - steady_pres) * cv[0] * (phi_L_e.dp * rho_f_e.value + phi_L_e.value * rho_f_e.dp)
        )

    integrand_dFsdT = _EMPTY
    if want_tangent_dT:
        integrand_dFsdT = (
            (1 - steady_pres) * cv[0] * (phi_L_n.dT * rho_f_n.value + phi_L_n.value * rho_f_n.dT)
        )

    return mass_dot_e, integrand_dFsdp_e, integrand_dFsdT


def _evaluate_energy_storage_integrands(
    Te: npt.NDArray[np.float64],
    phi_L_e: PropertyResult,
    rho_f_e: PropertyResult,
    u_f_e: PropertyResult,
    is_fracture: bool,
    rho_s: float,
    cp_s: float,
    cv: npt.NDArray[np.float64],
    steady_temp: float,
    Te_n: npt.NDArray[np.float64] = _EMPTY,
    T_q_n: npt.NDArray[np.float64] = _EMPTY,
    phi_L_n: PropertyResult = PropertyResult(_EMPTY, _EMPTY, _EMPTY),
    rho_f_n: PropertyResult = PropertyResult(_EMPTY, _EMPTY, _EMPTY),
    u_f_n: PropertyResult = PropertyResult(_EMPTY, _EMPTY, _EMPTY),
    *,
    want_tangent_dT: bool = False,
    want_tangent_dp: bool = False,
) -> tuple[
    npt.NDArray[np.float64],  # E_e
    npt.NDArray[np.float64],  # energy_dot_e
    npt.NDArray[np.float64],  # integrand_dFCdT_e (KTT), `_EMPTY` unless want_tangent_dT
    npt.NDArray[np.float64],  # integrand_dFCdp (KTp), `_EMPTY` unless want_tangent_dp
]:
    """Evaluate the nodal (lumped) energy storage rate and optionally its temperature and pressure tangents.

    `phi_L_e`/`rho_f_e`/`u_f_e` are nodal (their `.dT` feeds the dT-tangent, matching KTT's
    lumped discretization); `phi_L_n`/`rho_f_n`/`u_f_n` are quadrature current-iterate (their
    `.dp` feeds the dp-tangent, matching KTp's consistent discretization).
    """
    E_e = _evaluate_total_energy(Te, phi_L_e, rho_f_e, u_f_e, is_fracture, rho_s, cp_s)
    energy_dot_e = (1 - steady_temp) * (cv[0] * E_e[:, 0:1] + cv[1] * E_e[:, 1:2] + cv[2] * E_e[:, 2:3])

    integrand_dFCdT_e = _EMPTY
    if want_tangent_dT:
        integrand_dFCdT_e = (
            (1 - steady_temp)
            * cv[0]
            * (
                (1 - phi_L_e.value) * rho_s * cp_s * (1 - is_fracture)
                + phi_L_e.value * (rho_f_e.dT * u_f_e.value + rho_f_e.value * u_f_e.dT)
                + phi_L_e.dT * (rho_f_e.value * u_f_e.value - rho_s * cp_s * Te_n)
            )
        )

    integrand_dFCdp = _EMPTY
    if want_tangent_dp:
        integrand_dFCdp = (
            (1 - steady_temp)
            * cv[0]
            * (
                phi_L_n.value * (rho_f_n.dp * u_f_n.value + rho_f_n.value * u_f_n.dp)
                + phi_L_n.dp * (rho_f_n.value * u_f_n.value - rho_s * cp_s * T_q_n)
            )
        )

    return E_e, energy_dot_e, integrand_dFCdT_e, integrand_dFCdp


def _evaluate_energy_properties(
    T_q: npt.NDArray[np.float64],
    p_q: npt.NDArray[np.float64],
    T_q_n: npt.NDArray[np.float64],
    p_q_n: npt.NDArray[np.float64],
    T_flow_q_n: npt.NDArray[np.float64],
    phi_L_n: PropertyResult,
    is_fracture: bool,
    w0: float,
    lambda_f: float,
    lambda_s: npt.NDArray[np.float64] | float,
    isotropic_cond: bool,
    nsd: int,
    fluid: dict[str, Any],
    *,
    T_flow_lagged: bool = False,
) -> tuple[PropertyResult, EnthalpyResult, npt.NDArray[np.float64] | float]:
    """Evaluate fluid internal energy and enthalpy at quadrature points: `(u_f, hf_q, lambda_e)`.

    `u_f` carries the full time history in `.value`/`.dp`/`.dT` (not sliced to current
    iterate); `hf_q` is already evaluated at the current iterate only, since it's built from
    the `_n`-sliced inputs. Also calculates the region's effective thermal conductivity.
    """
    u_f = get_internal_energy(T_q, p_q, fluid)  # fluid internal energy, nq x 1 x 1
    hf_q = get_enthalpy(p_q_n, T_q_n, T_flow_q_n, fluid, T_flow_lagged)  # fluid enthalpy, nq x 1 x 1

    lambda_e: npt.NDArray[np.float64] | float
    if is_fracture:
        lambda_e = w0 * lambda_f
    else:
        lambda_e = (
            phi_L_n.value * lambda_f + (1 - phi_L_n.value) * lambda_s
            if isotropic_cond
            else (phi_L_n.value * lambda_f * np.eye(nsd) + (1 - phi_L_n.value) * lambda_s)
        )

    return u_f, hf_q, lambda_e


def _evaluate_advection_terms(
    B_q: npt.NDArray[np.float64],
    N_q: npt.NDArray[np.float64],
    q_D: npt.NDArray[np.float64],  # mass flux
    rho_f_n: PropertyResult,
    rho_f: PropertyResult,
    hf_q: EnthalpyResult,
    T_q: npt.NDArray[np.float64],
    phi_L: PropertyResult,
    u_f: PropertyResult,
    is_fracture: bool,
    rho_s: float,
    cp_s: float,
    cv: npt.NDArray[np.float64],
    steady_temp: float,
    pe_n: npt.NDArray[np.float64],
    Te_n: npt.NDArray[np.float64],
    Te_flow_n: npt.NDArray[np.float64],
    rho_f_e_n: npt.NDArray[np.float64],
    nsd: int,
    nq: int,
    L_vec: npt.NDArray[np.float64],
    lambda_e: npt.NDArray[np.float64] | float,
    isotropic_cond: bool,
    fluid: dict[str, Any],
    dqD_dT: npt.NDArray[np.float64] = _EMPTY,
    T_q_n: npt.NDArray[np.float64] = _EMPTY,
    phi_L_n: PropertyResult = PropertyResult(_EMPTY, _EMPTY, _EMPTY),
    u_f_n: PropertyResult = PropertyResult(_EMPTY, _EMPTY, _EMPTY),
    dqD_dp: npt.NDArray[np.float64] = _EMPTY,
    *,
    want_tangent_dT: bool = False,
    want_tangent_dp: bool = False,
) -> tuple[
    npt.NDArray[np.float64],  # integrand_advection (F_Tadvection)
    npt.NDArray[np.float64],  # integrand_SUPG (F_Tstab)
    npt.NDArray[np.float64],  # integrand_advection_dT (KTT), `_EMPTY` unless want_tangent_dT
    npt.NDArray[np.float64],  # integrand_SUPG_dT (KTT), `_EMPTY` unless want_tangent_dT
    npt.NDArray[np.float64],  # integrand_advection_dp (KTp), `_EMPTY` unless want_tangent_dp
    npt.NDArray[np.float64],  # integrand_SUPG_dp (KTp), `_EMPTY` unless want_tangent_dp
]:
    """Evaluate the SUPG-stabilized advective energy residual integrands and tangents."""
    darcy_flux = q_D / rho_f_n.value
    E_q = _evaluate_total_energy(T_q, phi_L, rho_f, u_f, is_fracture, rho_s, cp_s)
    energy_dot = (1 - steady_temp) * (
        cv[0] * E_q[:, :, 0:1] + cv[1] * E_q[:, :, 1:2] + cv[2] * E_q[:, :, 2:3]
    )

    flux = hf_q.cp * q_D
    if nsd == 3:
        flux_mag = np.sqrt(flux[:, 0, 0] ** 2 + flux[:, 1, 0] ** 2 + flux[:, 2, 0] ** 2)
    elif nsd == 2:
        flux_mag = np.sqrt(flux[:, 0, 0] ** 2 + flux[:, 1, 0] ** 2)
    else:
        flux_mag = np.abs(flux[:, 0, 0])  # keep 1D (nq,), matching the nsd 2/3 branches
    indx_noflow = np.nonzero(flux_mag == 0)
    # prevent divide by zero
    flux_mag[indx_noflow] = 1.0e-15
    flux[indx_noflow, :, :] = 1.0e-15

    # Effective length weighted by fluxes
    if nsd == 3:
        dl_eff = flux_mag / np.sqrt(
            (flux[:, 0, 0] / L_vec[0, 0]) ** 2
            + (flux[:, 1, 0] / L_vec[1, 0]) ** 2
            + (flux[:, 2, 0] / L_vec[2, 0]) ** 2
        )
    elif nsd == 2:
        dl_eff = flux_mag / np.sqrt((flux[:, 0, 0] / L_vec[0, 0]) ** 2 + (flux[:, 1, 0] / L_vec[1, 0]) ** 2)
    else:
        dl_eff = L_vec * np.ones((1, 1, nq))
        flux_mag = flux_mag.reshape(1, 1, -1)  # reshape for 1D broadcasting

    if isotropic_cond:
        lambda_e_projected = lambda_e * np.ones((nq, 1, 1))
    else:
        # anisotropic case - project conductivity along flowlines
        n_stream = flux / flux_mag.reshape(-1, 1, 1)
        n_stream_T = n_stream.transpose(0, 2, 1)
        lambda_e_projected = (n_stream_T @ lambda_e) @ n_stream  # nq x 1 x 1

    # Peclet number
    Pe = dl_eff * flux_mag / lambda_e_projected.reshape(1, 1, -1)

    # SUPG stabilization coefficient
    tau = dl_eff / (2 * flux_mag) * (1 + 9 / Pe**2) ** (-0.5)
    tau[:, :, indx_noflow] = 0
    tau = tau.reshape(-1, 1, 1)

    # Discretized strong form residual at quadrature points
    hf_e = get_enthalpy(pe_n, Te_n, Te_flow_n, fluid)  # fluid enthalpy, nq x 1 x 1
    grad_hf = B_q @ hf_e.value
    grad_rhof = B_q @ rho_f_e_n
    RTh = (
        energy_dot + (grad_hf * rho_f_n.value + hf_q.value * grad_rhof).transpose(0, 2, 1) @ darcy_flux
    )  # discretized residual

    integrand_advection = q_D * hf_q.value
    integrand_SUPG = hf_q.cp * q_D * tau * RTh

    integrand_advection_dT = _EMPTY
    integrand_SUPG_dT = _EMPTY
    if want_tangent_dT:
        D2 = grad_rhof.transpose(0, 2, 1) @ q_D  # d/dTe of (grad_rhof . q_D), nq x 1 x 1
        dqD_dT_direct = -dqD_dT  # +d(q_D)/dT, opposite sign convention from Kpp/KpT's dqD_dT
        dR_dT = (
            (1 - steady_temp)
            * cv[0]
            * (
                (1 - phi_L_n.value) * rho_s * cp_s * (1 - is_fracture)
                + phi_L_n.dT * (rho_f_n.value * u_f_n.value - rho_s * cp_s * T_q_n)
                + phi_L_n.value * (rho_f_n.dT * u_f_n.value + rho_f_n.value * u_f_n.dT)
            )
            * N_q
            + (hf_q.dT + hf_q.value / rho_f_n.value * rho_f_n.dT) * (q_D.transpose(0, 2, 1) @ B_q)
            + (grad_hf + hf_q.value * grad_rhof / rho_f_n.value).transpose(0, 2, 1) @ dqD_dT_direct
            + (hf_q.dT / rho_f_n.value - hf_q.value * rho_f_n.dT / rho_f_n.value**2) * D2 * N_q
        )
        integrand_advection_dT = hf_q.dT * q_D * N_q + hf_q.value * dqD_dT_direct
        integrand_SUPG_dT = tau * hf_q.cp * dR_dT * q_D + tau * hf_q.cp * RTh * dqD_dT_direct

    integrand_advection_dp = _EMPTY
    integrand_SUPG_dp = _EMPTY
    if want_tangent_dp:
        dqD_dp_direct = -dqD_dp  # +d(q_D)/dp, opposite sign convention from Kpp/KpT's dqD_dp
        dR_dp = (
            (1 - steady_temp)
            * cv[0]
            * (
                phi_L_n.dp * (rho_f_n.value * u_f_n.value - rho_s * cp_s * T_q_n)
                + phi_L_n.value * (rho_f_n.dp * u_f_n.value + rho_f_n.value * u_f_n.dp)
            )
            * N_q
            + (hf_q.dp + hf_q.value / rho_f_n.value * rho_f_n.dp) * q_D * B_q
            + (grad_hf + hf_q.value * grad_rhof / rho_f_n.value) * dqD_dp_direct
            + (hf_q.dp * grad_rhof / rho_f_n.value - hf_q.value * grad_rhof / rho_f_n.value**2 * rho_f_n.dp)
        )
        integrand_advection_dp = hf_q.dp * q_D * N_q + hf_q.value * dqD_dp_direct
        integrand_SUPG_dp = tau * hf_q.cp * dR_dp * q_D + tau * hf_q.cp * RTh * dqD_dp_direct

    return (
        integrand_advection,
        integrand_SUPG,
        integrand_advection_dT,
        integrand_SUPG_dT,
        integrand_advection_dp,
        integrand_SUPG_dp,
    )


def integrate_residuals(  # noqa
    mesh: dict[str, Any],
    material: dict[str, Any],
    primary_variables: dict[str, Any],
    controls: dict[str, Any],
    cv: npt.NDArray[np.float64],  # rate coefficients
    T_flow: npt.NDArray[np.float64],  # temperature field used for property evaluations
    *,
    is_fracture: bool = False,
    pressure_only: bool = False,
    temperature_only: bool = False,
) -> dict[str, Any]:
    """Integrate residual vectors for conservation of mass and energy."""
    p = primary_variables["p"]
    T = primary_variables["T"]

    p_0 = material["p_0"]
    T_0 = material["T_0"]

    axisymmetry = bool(controls["axisymmetric_on"])
    fracture_on = bool(controls["fracture_on"])
    advection_on = bool(controls["advection_on"])
    volumetric_strain_on = bool(controls["volumetric_strain_on"])
    steady_pres = controls["steady_pressure"]
    steady_temp = controls["steady_temperature"]

    fluid = material["fluid"]
    lambda_f = fluid["therm_cond"]

    # fixed accumulator buckets; new nonlinear terms just sum onto one
    list_of_matrices = ["F_Pstorage", "F_Pdiffusion"]
    if not pressure_only:
        list_of_matrices = ["F_Tstorage", "F_Tdiffusion", "F_Tadvection", "F_Tstab"] + list_of_matrices

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
        nn_m = mesh["nn"]

        # Connectivity-only fracture/matrix coupling index vectors, cached on fmesh.
        if "vec_mat_asmbl" not in fmesh:
            (
                _vec_count_ff,
                row_vec_ff,
                col_vec_ff,
                _vec_count_fm,
                row_vec_fm,
                col_vec_fm,
                _vec_count_mf,
                row_vec_mf,
                col_vec_mf,
                _vec_count_mm,
                row_vec_mm,
                col_vec_mm,
                _vec_count_flux_m,
                row_vec_flux_m,
            ) = _create_fracture_coupling_vectors(conn, fmesh["matrix_conn"], mesh["conn"])
            fmesh["vec_mat_asmbl"] = {
                "row_vec_ff": row_vec_ff,
                "col_vec_ff": col_vec_ff,
                "row_vec_fm": row_vec_fm,
                "col_vec_fm": col_vec_fm,
                "row_vec_mf": row_vec_mf,
                "col_vec_mf": col_vec_mf,
                "row_vec_mm": row_vec_mm,
                "col_vec_mm": col_vec_mm,
                "row_vec_flux_m": row_vec_flux_m,
            }
        row_vec_flux_m = fmesh["vec_mat_asmbl"]["row_vec_flux_m"]
        coupling_flux_names = (["F_Pdiffusion_mm"] if not temperature_only else []) + (
            ["F_Tdiffusion_mm"] if not pressure_only else []
        )
        if not pressure_only and advection_on:
            coupling_flux_names = coupling_flux_names + ["F_Tadvection_mm"]

        vec_flux_m = {name: np.zeros(row_vec_flux_m.size) for name in coupling_flux_names}
        vec_counter_flux_m = 0

    # ----- gravity vector
    g = _get_gravity(mesh["nsd"], controls["gravity"])

    vec = _initiate_vectors(list_of_matrices, nne_list, elem_indx, ismatrix=False)
    vec_counter = 0  # vectorized storage index counter

    # ----- Matrix integration element loop
    for r in range(num_regions):
        kf_0, isotropic_perm, nodes_r, w0, u, kf_perp = _get_region_flow_data(
            mesh, material, fmesh, nodes_f, is_fracture, r
        )
        # Gravity projected onto the fracture's own local (tangential) coordinate, matching
        # the reduced dimensionality of the fracture's own gradP/B_q.
        g_r = (u.T @ g) if is_fracture else g
        isotropic_cond, hc = _get_region_thermal_data(material, is_fracture, r)
        K_d, phi_0, biot, rho_s, cp_s, alpha_s, lambda_s = _get_region_solid_data(material, r)

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

            # Element vectors initialize
            element_vectors: dict[str, npt.NDArray[np.float64]] = {}

            (
                Te,
                Te_flow,
                pe,
                T_0e,
                p_0e,
                L_vec,
                T_q,
                T_flow_q,
                p_q,
                gradT,
                gradP,
            ) = _evaluate_element_kinematics(N_q, B_q, xI, T, T_flow, p, T_0, p_0, enodes_dof)
            # Current-iterate (column 0) slices, reused throughout below.
            Te_n = Te[:, 0:1]
            Te_flow_n = Te_flow[:, 0:1]
            pe_n = pe[:, 0:1]
            T_q_n = T_q[:, :, 0:1]
            T_flow_q_n = T_flow_q[:, :, 0:1]
            p_q_n = p_q[:, :, 0:1]

            kf, mu_q, rho_f = _evaluate_flow_properties(kf_0, T_flow_q_n, p_q_n, T_flow_q, p_q, fluid)
            rho_f_n = PropertyResult(rho_f.value[:, :, 0:1], rho_f.dp, rho_f.dT)
            _, q_D, _, _ = _evaluate_darcy_flux(kf, mu_q, gradP, rho_f_n, g_r, isotropic_perm, N_q, B_q)

            phi_L = _evaluate_quad_porosity(
                p_q,
                p_0e,
                T_flow_q,
                T_0e,
                N_q,
                phi_0,
                K_d,
                biot,
                alpha_s,
                is_fracture,
                w0,
                nq,
                volumetric_strain_on=volumetric_strain_on,
            )
            phi_L_n = PropertyResult(phi_L.value[:, :, 0:1], phi_L.dp, phi_L.dT)

            # Nodal evaluation for self-consistent lumping of the storage terms.
            phi_L_e, rho_f_e, u_f_e = _evaluate_nodal_state(
                pe,
                p_0e,
                Te,
                Te_flow,
                T_0e,
                phi_0,
                K_d,
                biot,
                alpha_s,
                is_fracture,
                w0,
                fluid,
                volumetric_strain_on=volumetric_strain_on,
            )
            m_lump = _vctrzd_quad_point_integration(N_q, 1.0, N_q, wJe_q, True).sum(axis=1).reshape(-1, 1)

            # ------ Conservation of mass
            if not temperature_only:
                mass_dot_e, _, _ = _evaluate_mass_storage_integrands(phi_L_e, rho_f_e, cv, steady_pres)
                element_vectors["F_Pstorage"] = m_lump * mass_dot_e

                # Diffusion vector
                element_vectors["F_Pdiffusion"] = _vctrzd_qp_vctr_integrate(B_q, q_D, wJe_q, isotropic_perm)

            # ------ Conservation of energy
            if not pressure_only:
                u_f, hf_q, lambda_e = _evaluate_energy_properties(
                    T_q,
                    p_q,
                    T_q_n,
                    p_q_n,
                    T_flow_q_n,
                    phi_L_n,
                    is_fracture,
                    w0,
                    lambda_f,
                    lambda_s,
                    isotropic_cond,
                    nsd,
                    fluid,
                )
                _, energy_dot_e, _, _ = _evaluate_energy_storage_integrands(
                    Te, phi_L_e, rho_f_e, u_f_e, is_fracture, rho_s, cp_s, cv, steady_temp
                )
                element_vectors["F_Tstorage"] = m_lump * energy_dot_e

                # Diffusion vector
                element_vectors["F_Tdiffusion"] = _vctrzd_qp_vctr_integrate(
                    B_q, lambda_e * gradT, wJe_q, isotropic_cond
                )

                # Advection and SUPG stabilization
                if advection_on:
                    integrand_advection, integrand_SUPG, _, _, _, _ = _evaluate_advection_terms(
                        B_q,
                        N_q,
                        q_D,
                        rho_f_n,
                        rho_f,
                        hf_q,
                        T_q,
                        phi_L,
                        u_f,
                        is_fracture,
                        rho_s,
                        cp_s,
                        cv,
                        steady_temp,
                        pe_n,
                        Te_n,
                        Te_flow_n,
                        rho_f_e.value[:, 0:1],
                        nsd,
                        nq,
                        L_vec,
                        lambda_e,
                        isotropic_cond,
                        fluid,
                    )
                    element_vectors["F_Tadvection"] = _vctrzd_qp_vctr_integrate(
                        B_q, integrand_advection, wJe_q, isotropic_perm
                    )
                    element_vectors["F_Tstab"] = _vctrzd_qp_vctr_integrate(
                        B_q, integrand_SUPG, wJe_q, isotropic_perm
                    )

            # ----- Fracture-matrix leak-off/convective-exchange coupling
            if is_fracture:
                fracture_flux_vectors, matches = _integrate_fracture_matrix_coupling_residual(
                    e,
                    enodes,
                    nodes_f,
                    N_q,
                    wJe_q,
                    mesh,
                    fluid,
                    pe_n,
                    pe[:, 1:2],
                    Te_n,
                    Te_flow_n,
                    p[:, 0:1],
                    p[:, 1:2],
                    T[:, 0:1],
                    T_flow[:, 0:1],
                    kf_perp,
                    mu_q,
                    rho_f_n,
                    hc,
                    pressure_coupling=not temperature_only,
                    thermal_coupling=not pressure_only,
                    advection_on=advection_on,
                )
                _accumulate_element_matrices(element_vectors, fracture_flux_vectors)

                for nne_m, matrix_vectors in matches:
                    stop = vec_counter_flux_m + nne_m  # vectorized index count end
                    _store_vctrzd_element_matrix(vec_flux_m, matrix_vectors, vec_counter_flux_m, stop)
                    vec_counter_flux_m = stop

            # ----- Assemble element contributions into global matrices
            stop = vec_counter + nne  # vectorized index count end
            _store_vctrzd_element_matrix(vec, element_vectors, vec_counter, stop)
            vec_counter = stop

    # ----- Return and convert to sparse matrix
    integrated_vectors = _assemble_global_matrices(
        vec, conn, nn, nne_list, elem_indx, np.concatenate(region_lists), isvector=True
    )

    if is_fracture:  # Fracture residuals to be updated
        # Matrix-side correction from the leak-off/convective-exchange coupling, scatter-assembled
        # via the cached row_vec_flux_m into a (nn_m, 1) vector for the caller to fold into its
        # own F_P/Tdiffusion.
        for name in vec_flux_m:
            integrated_vectors[name] = _make_sparse(
                vec_flux_m[name], row_vec_flux_m, np.zeros_like(row_vec_flux_m), nn_m, 1
            )
        return integrated_vectors

    if fracture_on:
        # Recursion call
        integrated_fracture_vectors = integrate_residuals(
            mesh,
            material,
            primary_variables,
            controls,
            cv,
            T_flow,
            is_fracture=True,
            pressure_only=pressure_only,
            temperature_only=temperature_only,
        )
        # Fold the matrix-side "_mm" flux from the fracture pass into the matrix residual.
        for name in ("F_Pdiffusion", "F_Tdiffusion", "F_Tadvection"):
            if name + "_mm" in integrated_fracture_vectors:
                flux_mm = integrated_fracture_vectors[name + "_mm"]
                integrated_vectors[name] = integrated_vectors[name] + flux_mm
        # Combine fracture and matrix vectors into one
        integrated_vectors = _stack_fracture_vectors(integrated_vectors, integrated_fracture_vectors)

    return integrated_vectors


def integrate_enthlpy_bndry_residual(
    mesh: dict[str, Any],
    material: dict[str, Any],
    primary_variables: dict[str, Any],
    controls: dict[str, Any],
    T_flow: npt.NDArray[np.float64],  # temperature field used for flow-property evaluations
    BC_config: list[dict[str, Any]],
) -> dict[str, Any]:
    """Integrate the advective enthalpy flux boundary vector.

    Only applies to fluid BC without an associated temperature BC.
    """
    advection_off = not bool(controls["advection_on"])
    nn_total = mesh["nn_total"]

    if advection_off:
        return {"F_enth_bnd": csc_matrix((nn_total, 1), dtype=float)}

    F_enth_bnd = np.zeros((nn_total, 1))
    fluid = material["fluid"]
    T = primary_variables["T"][:, 0:1]
    p = primary_variables["p"][:, 0:1]
    q = primary_variables["q"][:, 0:1]
    node_value: dict[int, float] = {}  # keyed by node so shared nodes aren't double-counted

    for group in BC_config:
        if "hydraulics" in group and "thermal" not in group and "_conn" in group:
            nodes_i = np.asarray(group["nodes"], dtype=np.int32)
            q_i = q[nodes_i, 0]
            T_i = T[nodes_i, 0]
            T_flow_i = T_flow[nodes_i, 0]
            p_i = p[nodes_i, 0]
            hf_i = get_enthalpy(p_i, T_i, T_flow_i, fluid).value
            node_value.update(zip(nodes_i.tolist(), (q_i * hf_i).tolist()))

    if node_value:
        node_ids = np.fromiter(node_value.keys(), dtype=np.int32)
        F_enth_bnd[node_ids, 0] = np.fromiter(node_value.values(), dtype=np.float64)

    return {"F_enth_bnd": csc_matrix(F_enth_bnd)}


def integrate_enthlpy_bndry_tangent(
    mesh: dict[str, Any],
    material: dict[str, Any],
    primary_variables: dict[str, Any],
    controls: dict[str, Any],
    T_flow: npt.NDArray[np.float64],  # temperature field used for flow-property evaluations
    BC_config: list[dict[str, Any]],
    *,
    fully_coupled: bool = False,
    T_flow_lagged: bool = False,
    Kpp: csc_matrix[Any] = csc_matrix((0, 0)),
    KpT: csc_matrix[Any] = csc_matrix((0, 0)),
) -> dict[str, Any]:
    """Integrate the advective enthalpy flux boundary tangent.

    Only applies to fluid BC without an associated temperature BC. d(F_enth_bnd)/dT is
    q*dhf_dT (diagonal) plus hf_i*dq/dT; the latter is nonzero only for a Dirichlet-pressure
    node under `fully_coupled`, via the Kpp/KpT cross-terms.
    """
    advection_off = not bool(controls["advection_on"])
    nn_total = mesh["nn_total"]

    if advection_off:
        if fully_coupled:
            return {
                "K_enth_bnd": csc_matrix((nn_total, nn_total), dtype=float),
                "K_enth_bnd_Tp": csc_matrix((nn_total, nn_total), dtype=float),
            }
        else:
            return {"K_enth_bnd": csc_matrix((nn_total, nn_total), dtype=float)}

    K_enth_bnd = csc_matrix((nn_total, nn_total), dtype=float)
    K_enth_bnd_Tp = csc_matrix((nn_total, nn_total), dtype=float)
    fluid = material["fluid"]
    T = primary_variables["T"][:, 0:1]
    p = primary_variables["p"][:, 0:1]
    q = primary_variables["q"][:, 0:1]
    node_diag_value: dict[int, float] = {}  # keyed by node so shared nodes aren't double-counted

    for group in BC_config:
        if "hydraulics" in group and "thermal" not in group and "_conn" in group:
            nodes_i = np.asarray(group["nodes"], dtype=np.int32)
            q_i = q[nodes_i, 0]
            T_i = T[nodes_i, 0]
            T_flow_i = T_flow[nodes_i, 0]
            p_i = p[nodes_i, 0]
            hf_result = get_enthalpy(p_i, T_i, T_flow_i, fluid, T_flow_lagged)
            hf_i, dhf_dT_i = hf_result.value, hf_result.dT
            node_diag_value.update(zip(nodes_i.tolist(), (q_i * dhf_dT_i).tolist()))

            # A flux-type node's q is a prescribed BC value (never a function of
            # p/T); only a pressure-type node's q (a mass-residual row) needs the
            # Kpp/KpT cross-terms.
            if fully_coupled and group["hydraulics"]["type"] == "pressure":
                rows = KpT[nodes_i, :].tocoo()
                K_enth_bnd = K_enth_bnd + csc_matrix(
                    (rows.data * hf_i[rows.row], (nodes_i[rows.row], rows.col)),
                    shape=(nn_total, nn_total),
                )
                rows = Kpp[nodes_i, :].tocoo()
                K_enth_bnd_Tp = K_enth_bnd_Tp + csc_matrix(
                    (rows.data * hf_i[rows.row], (nodes_i[rows.row], rows.col)),
                    shape=(nn_total, nn_total),
                )

    if node_diag_value:
        node_ids = np.fromiter(node_diag_value.keys(), dtype=np.int32)
        diag_vals = np.fromiter(node_diag_value.values(), dtype=np.float64)
        K_enth_bnd = K_enth_bnd + csc_matrix((diag_vals, (node_ids, node_ids)), shape=(nn_total, nn_total))

    return {"K_enth_bnd": K_enth_bnd, "K_enth_bnd_Tp": K_enth_bnd_Tp}


def integrate_tangent_matrices(  # noqa
    mesh: dict[str, Any],
    material: dict[str, Any],
    primary_variables: dict[str, Any],
    controls: dict[str, Any],
    cv: npt.NDArray[np.float64],  # rate coefficients
    T_flow: npt.NDArray[np.float64],  # temperature field used for flow-property evaluations
    *,
    is_fracture: bool = False,
    pressure_only: bool = False,
    temperature_only: bool = False,
    fully_coupled: bool = False,
    T_flow_lagged: bool = False,
) -> dict[str, Any]:
    """Integrate tangent matrices.

    `T_flow_lagged` selects whether flow properties' dependence on `T_flow` is treated as
    coupled to the unknown T (False) or as a frozen/extrapolated field (True).
    """
    p = primary_variables["p"]
    T = primary_variables["T"]

    p_0 = material["p_0"]
    T_0 = material["T_0"]

    axisymmetry = bool(controls["axisymmetric_on"])
    fracture_on = bool(controls["fracture_on"])
    advection_on = bool(controls["advection_on"])
    volumetric_strain_on = bool(controls["volumetric_strain_on"])
    steady_pres = controls["steady_pressure"]
    steady_temp = controls["steady_temperature"]

    if fully_coupled:
        # Overwrite bad combination if it occurs
        pressure_only = False
        temperature_only = False

    fluid = material["fluid"]
    lambda_f = fluid["therm_cond"]

    # fixed accumulator buckets; new nonlinear terms just sum onto one
    list_of_matrices = ["Kpp"]
    if not pressure_only:
        list_of_matrices = ["KTT", "KpT", "KTp"] + list_of_matrices
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

        # Connectivity-only index vectors, cached by integrate_residuals.
        cached_cv = fmesh["vec_mat_asmbl"]
        row_vec_ff, col_vec_ff = cached_cv["row_vec_ff"], cached_cv["col_vec_ff"]
        row_vec_fm, col_vec_fm = cached_cv["row_vec_fm"], cached_cv["col_vec_fm"]
        row_vec_mf, col_vec_mf = cached_cv["row_vec_mf"], cached_cv["col_vec_mf"]
        row_vec_mm, col_vec_mm = cached_cv["row_vec_mm"], cached_cv["col_vec_mm"]

        ktp_on = not pressure_only and advection_on and fully_coupled
        coupling_ff_names = (
            (["Kpp_ff"] if not temperature_only else [])
            + (["KTT_ff"] if not pressure_only else [])
            + (["KTp_ff"] if ktp_on else [])
        )
        coupling_fm_names = (
            (["Kpp_fm"] if not temperature_only else [])
            + (["KTT_fm"] if not pressure_only else [])
            + (["KTp_fm"] if ktp_on else [])
        )
        coupling_mf_names = (
            (["Kpp_mf"] if not temperature_only else [])
            + (["KTT_mf"] if not pressure_only else [])
            + (["KTp_mf"] if ktp_on else [])
        )
        coupling_mm_names = (
            (["Kpp_mm"] if not temperature_only else [])
            + (["KTT_mm"] if not pressure_only else [])
            + (["KTp_mm"] if ktp_on else [])
        )
        vec_ff = {name: np.zeros(row_vec_ff.size) for name in coupling_ff_names}
        vec_fm = {name: np.zeros(row_vec_fm.size) for name in coupling_fm_names}
        vec_mf = {name: np.zeros(row_vec_mf.size) for name in coupling_mf_names}
        vec_mm = {name: np.zeros(row_vec_mm.size) for name in coupling_mm_names}

        vec_counter_ff = 0
        vec_counter_fm = 0
        vec_counter_mf = 0
        vec_counter_mm = 0

    # ----- gravity vector
    g = _get_gravity(mesh["nsd"], controls["gravity"])

    vec = _initiate_vectors(list_of_matrices, nne_list, elem_indx)
    vec_counter = 0  # vectorized storage index counter

    # ----- Matrix integration element loop
    for r in range(num_regions):
        kf_0, isotropic_perm, nodes_r, w0, u, kf_perp = _get_region_flow_data(
            mesh, material, fmesh, nodes_f, is_fracture, r
        )
        # Gravity projected onto the fracture's own local (tangential) coordinate, matching
        # the reduced dimensionality of the fracture's own gradP/B_q.
        g_r = (u.T @ g) if is_fracture else g
        isotropic_cond, hc = _get_region_thermal_data(material, is_fracture, r)
        K_d, phi_0, biot, rho_s, cp_s, alpha_s, lambda_s = _get_region_solid_data(material, r)

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

            (
                Te,
                Te_flow,
                pe,
                T_0e,
                p_0e,
                L_vec,
                T_q,
                T_flow_q,
                p_q,
                gradT,
                gradP,
            ) = _evaluate_element_kinematics(N_q, B_q, xI, T, T_flow, p, T_0, p_0, enodes_dof)
            # Current-iterate (column 0) slices, reused throughout below.
            Te_n = Te[:, 0:1]
            Te_flow_n = Te_flow[:, 0:1]
            pe_n = pe[:, 0:1]
            T_q_n = T_q[:, :, 0:1]
            T_flow_q_n = T_flow_q[:, :, 0:1]
            p_q_n = p_q[:, :, 0:1]

            kf, mu_q, rho_f = _evaluate_flow_properties(
                kf_0, T_flow_q_n, p_q_n, T_flow_q, p_q, fluid, T_flow_lagged=T_flow_lagged
            )
            rho_f_n = PropertyResult(rho_f.value[:, :, 0:1], rho_f.dp, rho_f.dT)
            _, q_D, dqD_dp, dqD_dT = _evaluate_darcy_flux(
                kf,
                mu_q,
                gradP,
                rho_f_n,
                g_r,
                isotropic_perm,
                N_q,
                B_q,
                want_tangent_dp=not temperature_only,
                want_tangent_dT=not pressure_only and (advection_on or fully_coupled),
            )

            phi_L = _evaluate_quad_porosity(
                p_q,
                p_0e,
                T_flow_q,
                T_0e,
                N_q,
                phi_0,
                K_d,
                biot,
                alpha_s,
                is_fracture,
                w0,
                nq,
                T_flow_lagged=T_flow_lagged,
                volumetric_strain_on=volumetric_strain_on,
            )
            phi_L_n = PropertyResult(phi_L.value[:, :, 0:1], phi_L.dp, phi_L.dT)

            # Nodal evaluation for self-consistent lumping of the storage terms.
            phi_L_e, rho_f_e, u_f_e = _evaluate_nodal_state(
                pe_n,
                p_0e,
                Te_n,
                Te_flow_n,
                T_0e,
                phi_0,
                K_d,
                biot,
                alpha_s,
                is_fracture,
                w0,
                fluid,
                T_flow_lagged=T_flow_lagged,
                volumetric_strain_on=volumetric_strain_on,
            )
            m_lump = _vctrzd_quad_point_integration(N_q, 1.0, N_q, wJe_q, True).sum(axis=1)

            _, integrand_dFsdp_e, integrand_dFsdT = _evaluate_mass_storage_integrands(
                phi_L_e,
                rho_f_e,
                cv,
                steady_pres,
                phi_L_n,
                rho_f_n,
                want_tangent_dp=not temperature_only,
                want_tangent_dT=fully_coupled,
            )

            # ------------------------------------ Conservation of mass wrt pressure
            if not temperature_only:
                # Storage vector - Fs (lumped: nodal dFs/dp times the lumped mass weight)
                element_matrices["Kpp"] = np.diag(m_lump * integrand_dFsdp_e.ravel())

                # Diffusion vector - Fk
                # dFk/dp
                element_matrices["Kpp"] += _vctrzd_qp_vctr_integrate(B_q, dqD_dp, wJe_q, False)

            if not pressure_only:
                # -------------------------------- Conservation of energy wrt temperature
                u_f, hf_q, lambda_e = _evaluate_energy_properties(
                    T_q,
                    p_q,
                    T_q_n,
                    p_q_n,
                    T_flow_q_n,
                    phi_L_n,
                    is_fracture,
                    w0,
                    lambda_f,
                    lambda_s,
                    isotropic_cond,
                    nsd,
                    fluid,
                    T_flow_lagged=T_flow_lagged,
                )
                u_f_n = PropertyResult(u_f.value[:, :, 0:1], u_f.dp[:, :, 0:1], u_f.dT[:, :, 0:1])

                # Energy storage vector - FC (lumped: nodal dFC/dT times the lumped mass weight)
                _, _, integrand_dFCdT_e, integrand_dFCdp = _evaluate_energy_storage_integrands(
                    Te,
                    phi_L_e,
                    rho_f_e,
                    u_f_e,
                    is_fracture,
                    rho_s,
                    cp_s,
                    cv,
                    steady_temp,
                    Te_n,
                    T_q_n,
                    phi_L_n,
                    rho_f_n,
                    u_f_n,
                    want_tangent_dT=True,
                    want_tangent_dp=fully_coupled,
                )
                element_matrices["KTT"] = np.diag(m_lump * integrand_dFCdT_e.ravel())

                # Themal diffusion vector
                integrand_dFlambdadT = lambda_e * B_q + phi_L_n.dT * (lambda_f - lambda_s) * gradT * N_q
                element_matrices["KTT"] += _vctrzd_qp_vctr_integrate(B_q, integrand_dFlambdadT, wJe_q, False)

                # Advection and SUPG stabilization
                integrand_advection_dp = integrand_SUPG_dp = _EMPTY
                if advection_on:
                    (
                        _,
                        _,
                        integrand_advection_dT,
                        integrand_SUPG_dT,
                        integrand_advection_dp,
                        integrand_SUPG_dp,
                    ) = _evaluate_advection_terms(
                        B_q,
                        N_q,
                        q_D,
                        rho_f_n,
                        rho_f,
                        hf_q,
                        T_q,
                        phi_L,
                        u_f,
                        is_fracture,
                        rho_s,
                        cp_s,
                        cv,
                        steady_temp,
                        pe_n,
                        Te_n,
                        Te_flow_n,
                        rho_f_e.value,
                        nsd,
                        nq,
                        L_vec,
                        lambda_e,
                        isotropic_cond,
                        fluid,
                        dqD_dT,
                        T_q_n,
                        phi_L_n,
                        u_f_n,
                        dqD_dp,
                        want_tangent_dT=True,
                        want_tangent_dp=fully_coupled,
                    )
                    element_matrices["KTT"] -= _vctrzd_qp_vctr_integrate(
                        B_q, integrand_advection_dT, wJe_q, False
                    )
                    element_matrices["KTT"] += _vctrzd_qp_vctr_integrate(B_q, integrand_SUPG_dT, wJe_q, False)

            # Fully coupled - off diagonal matrices
            if fully_coupled:
                # ----------- Conservation of mass wrt temperature
                # Storage vector - Fs
                element_matrices["KpT"] = _vctrzd_quad_point_integration(
                    N_q, integrand_dFsdT, N_q, wJe_q, scalar=True
                )

                # Diffusion vector - Fk
                integrand_dFkdT = dqD_dT
                element_matrices["KpT"] += _vctrzd_qp_vctr_integrate(B_q, integrand_dFkdT, wJe_q, False)

                # ----------- Conservation of energy wrt pressure
                # Energy storage vector - FC
                element_matrices["KTp"] = _vctrzd_quad_point_integration(
                    N_q, integrand_dFCdp, N_q, wJe_q, scalar=True
                )

                # Themal diffusion vector
                integrand_dFlambdadT = phi_L_n.dp * (lambda_f - lambda_s) * gradT * N_q
                element_matrices["KTp"] += _vctrzd_qp_vctr_integrate(B_q, integrand_dFlambdadT, wJe_q, False)

                if advection_on:
                    element_matrices["KTp"] -= _vctrzd_qp_vctr_integrate(
                        B_q, integrand_advection_dp, wJe_q, False
                    )
                    element_matrices["KTp"] += _vctrzd_qp_vctr_integrate(B_q, integrand_SUPG_dp, wJe_q, False)

            # ----- Fracture-matrix leak-off/convective-exchange coupling
            if is_fracture:
                coupling_entries = _integrate_fracture_matrix_coupling_tangent(
                    e,
                    enodes,
                    nodes_f,
                    N_q,
                    wJe_q,
                    mesh,
                    fluid,
                    pe_n,
                    pe[:, 1:2],
                    Te_n,
                    Te_flow_n,
                    p[:, 0:1],
                    p[:, 1:2],
                    T[:, 0:1],
                    T_flow[:, 0:1],
                    kf_perp,
                    mu_q,
                    rho_f_n,
                    hc,
                    pressure_coupling=not temperature_only,
                    thermal_coupling=not pressure_only,
                    fully_coupled=fully_coupled,
                    advection_on=advection_on,
                    T_flow_lagged=T_flow_lagged,
                )

                for nne_m, ff_matrices, fm_matrices, mf_matrices, mm_matrices in coupling_entries:
                    stop = vec_counter_ff + nne * nne  # vectorized index count end
                    _store_vctrzd_element_matrix(vec_ff, ff_matrices, vec_counter_ff, stop)
                    vec_counter_ff = stop
                    stop = vec_counter_fm + nne_m * nne  # vectorized index count end
                    _store_vctrzd_element_matrix(vec_fm, fm_matrices, vec_counter_fm, stop)
                    vec_counter_fm = stop
                    stop = vec_counter_mf + nne * nne_m  # vectorized index count end
                    _store_vctrzd_element_matrix(vec_mf, mf_matrices, vec_counter_mf, stop)
                    vec_counter_mf = stop
                    stop = vec_counter_mm + nne_m * nne_m  # vectorized index count end
                    _store_vctrzd_element_matrix(vec_mm, mm_matrices, vec_counter_mm, stop)
                    vec_counter_mm = stop

            # ----- Assemble element contributions into global matrices
            stop = vec_counter + nne * nne  # vectorized index count end
            _store_vctrzd_element_matrix(vec, element_matrices, vec_counter, stop)
            vec_counter = stop

    # ----- Return and convert to sparse matrix
    integrated_matrices = _assemble_global_matrices(
        vec, conn, nn, nne_list, elem_indx, np.concatenate(region_lists)
    )

    if is_fracture:
        nn_m = mesh["nn"]
        for name in vec_ff:
            integrated_matrices[name] = _make_sparse(vec_ff[name], row_vec_ff, col_vec_ff, nn, nn)
        for name in vec_fm:
            integrated_matrices[name] = _make_sparse(vec_fm[name], row_vec_fm, col_vec_fm, nn_m, nn)
        for name in vec_mf:
            integrated_matrices[name] = _make_sparse(vec_mf[name], row_vec_mf, col_vec_mf, nn, nn_m)
        for name in vec_mm:
            integrated_matrices[name] = _make_sparse(vec_mm[name], row_vec_mm, col_vec_mm, nn_m, nn_m)
        return integrated_matrices

    if fracture_on:
        # Recursion call
        integrated_fracture_matrices = integrate_tangent_matrices(
            mesh,
            material,
            primary_variables,
            controls,
            cv,
            T_flow,
            is_fracture=True,
            pressure_only=pressure_only,
            temperature_only=temperature_only,
            fully_coupled=fully_coupled,
            T_flow_lagged=T_flow_lagged,
        )
        # Combine fracture and matrix matrices into one
        integrated_matrices = _stack_fracture_matrices(
            integrated_matrices, integrated_fracture_matrices, nn, mesh["fracture_mesh"]["nn"]
        )

    return integrated_matrices


def integrate_boundary_flux_matrices(
    mesh: dict[str, Any],
    conn_face: list[list[int]],
    node_order: Sequence[int],
    controls: dict[str, Any],
    intgrl_type: str,
) -> dict[str, csc_matrix[Any]]:
    """Integrate lower dimensional boundary integrals for flux BCs."""
    ne = np.size(conn_face, 0)
    nne = np.size(conn_face, 1)
    nodes = mesh["nodes"]  # (nn,  nsd) - nodal degrees of freedom
    axisymmetry = bool(controls["axisymmetric_on"])
    elem_indx = [ne - 1]
    nne_list = [nne]

    if intgrl_type == "face":
        # Faces of elements (assumed exterior)
        nsd = mesh["nsd"] - 1
        etype = mesh["element_type"]
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
    else:
        # Lines
        nsd = 1
        etype = "L2"
        nq = 2

    # ----- Local face numbering
    # Map each global node id in conn_face to a local index, for assembly.
    face_node_ids = np.unique(np.asarray(conn_face, dtype=np.int32))
    global_to_local = {int(g): i for i, g in enumerate(face_node_ids)}
    nn_face = len(face_node_ids)

    conn_face_local = [[global_to_local[int(n)] for n in enodes] for enodes in conn_face]
    # conn_face_local is used for assembly then mapped back to global

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

        xI = nodes[enodes, :]
        if intgrl_type == "face":
            # calculate face normal
            # Faces
            if nsd == 2:
                n = np.cross(xI[1, :] - xI[0, :], xI[2, :] - xI[0, :])
            else:
                n = np.array([-(xI[1, 1] - xI[0, 1]), xI[1, 0] - xI[0, 0]])

            # Find the in-plane coordinates - assumes faces are aligned with cartesian axes
            ax = np.argmax(abs(n))
            xI = np.delete(xI, ax, 1)
        else:
            # Lines
            # For lines, shift each element edge to its first node
            A = xI[0, :]
            B = xI[1, :]
            xI = np.zeros((2, 1))
            xI[1] = np.linalg.norm(B - A)

        detJe_q = getB(GN_q, xI, nsd)[1]
        wJe_q = W * detJe_q  # Weight * det(Jacobian), used in every integrand
        if axisymmetry:
            # Add radial component to integrand for axisymmetry
            r_coord = N_q @ nodes[enodes, 0]
            wJe_q = wJe_q * r_coord.T

        # Element matrices
        element_matrices = _initialize_element_matrices(list_of_matrices, nne, nne)
        # Boundary faces may be misordered relative to global position (negative jacobians);
        # flux direction is controlled via the sign of the applied traction instead.
        element_matrices["A"] = _vctrzd_quad_point_integration(N_q, integrand, N_q, abs(wJe_q), True)

        # ----- Assemble element contributions into global matrices
        stop = vec_counter + nne * nne  # vectorized index count end
        _store_vctrzd_element_matrix(vec, element_matrices, vec_counter, stop)
        vec_counter = stop

    matrices = _assemble_global_matrices(
        vec, conn_face_local, nn_face, nne_list, elem_indx, np.arange(0, ne, dtype=np.int32)
    )
    # Map local ordering back to global ordering using the input node_order
    local_idx = np.searchsorted(face_node_ids, np.asarray(node_order, dtype=np.int32))
    return {name: mat[local_idx, :][:, local_idx] for name, mat in matrices.items()}
