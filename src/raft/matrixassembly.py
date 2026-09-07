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
    get_density,
    get_effective_permeability,
    get_enthalpy,
    get_internal_energy,
    get_porosity_lagrangian,
    get_viscosity,
)

# from .parseboundaryconditions import allocate_WB_BCs
from .shapefunctions import GNMATRIX_FUNCS, NMATRIX_FUNCS, getB, getquadrature


def _get_gravity(nsd: int, magnitude: float) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    ng = np.zeros((nsd, 1))
    ng[nsd - 1] = 1.0  # gravity normal
    return magnitude * ng, ng


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


def _get_region_value(val: Any, region_idx: int) -> Any:
    """Return value for region, handling both scalar and list cases.

    Fallback for homogeneous cases defined as scalars instead of lists.
    """
    return val[region_idx] if isinstance(val, list) else val


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


def _vctrzd_qp_vctr_integrate(
    left_matrix: npt.NDArray[np.float64],
    integrand: npt.NDArray[np.float64] | float,
    weights: npt.NDArray[np.float64],
    scalar: bool,
) -> npt.NDArray[np.float64]:
    """Perform vectorized vector integration over quadrature points.

    Takes in a left matrix that gets transposed, a middle integrand, and a right matrix.
    The left matrix should have dimensions nq x a x nne.
    The integrand should have dimensions a x 1 or nq x a x 1.
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

    `N_qm` are the matrix element's shape functions evaluated at the physical locations of the
    fracture element's own quadrature points, so `N_q` and `N_qm` sample the same points -
    shared by the tangent and residual fracture-matrix coupling assemblers below.
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
    kf_p: float,
    mu_q: npt.NDArray[np.float64] | float,
    h: float,
    *,
    pressure_coupling: bool,
    thermal_coupling: bool,
) -> list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Integrate one fracture element's coupling *tangent* contribution to its neighboring
    matrix elements - the leak-off/convective-exchange terms are linear in (p_f - p_m) and
    (T_f - T_m), so these coefficient matrices double as both the Jacobian block and (via
    `_integrate_fracture_matrix_coupling_residual`, contracted against the current state
    instead of assembled here) the residual.

    `kf_p`/`mu_q`/`h` are always the caller's real, already-resolved region coefficients;
    `pressure_coupling`/`thermal_coupling` select which of the Kpp/KTT blocks actually get
    assembled this call (a caller not currently using one passes an unread placeholder for it,
    same as the sentinel convention `_get_region_flow_data`/`_get_region_thermal_data` use for
    fields that don't apply).
    """
    entries: list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]] = []

    for _e_m, _enodes_m, nne_m, N_qm in _iter_fracture_matrix_matches(e, enodes_f, nodes_f, N_q, mesh):
        ff_matrices: dict[str, npt.NDArray[np.float64]] = {}
        fm_matrices: dict[str, npt.NDArray[np.float64]] = {}
        mm_matrices: dict[str, npt.NDArray[np.float64]] = {}

        if thermal_coupling:
            ff_matrices["KTT_ff"] = _vctrzd_quad_point_integration(N_q, h, N_q, wJe_q, True)
            fm_matrices["KTT_fm"] = _vctrzd_quad_point_integration(N_qm, -h, N_q, wJe_q, True)
            mm_matrices["KTT_mm"] = _vctrzd_quad_point_integration(N_qm, h, N_qm, wJe_q, True)

        if pressure_coupling:
            kf_p_mu = kf_p / mu_q
            ff_matrices["Kpp_ff"] = _vctrzd_quad_point_integration(N_q, -kf_p_mu, N_q, wJe_q, True)
            fm_matrices["Kpp_fm"] = _vctrzd_quad_point_integration(N_qm, kf_p_mu, N_q, wJe_q, True)
            mm_matrices["Kpp_mm"] = _vctrzd_quad_point_integration(N_qm, -kf_p_mu, N_qm, wJe_q, True)

        entries.append((nne_m, ff_matrices, fm_matrices, mm_matrices))

    return entries


def _integrate_fracture_matrix_coupling_residual(
    e: int,
    enodes_f: list[int],
    nodes_f: npt.NDArray[np.float64],
    N_q: npt.NDArray[np.float64],
    wJe_q: npt.NDArray[np.float64],
    mesh: dict[str, Any],
    pe_f: npt.NDArray[np.float64],
    Te_f: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    T: npt.NDArray[np.float64],
    kf_p: float,
    mu_q: npt.NDArray[np.float64],
    h: float,
    *,
    pressure_coupling: bool,
    thermal_coupling: bool,
) -> tuple[dict[str, npt.NDArray[np.float64]], list[tuple[int, dict[str, npt.NDArray[np.float64]]]]]:
    """Directly integrate the fracture-matrix leak-off/convective-exchange flux at the current
    state into nodal residual vectors - no coefficient matrices are assembled (this is evaluated
    fresh every Newton iteration, unlike the tangent, so building and then immediately
    contracting a matrix would be wasted work).

    `kf_p`/`mu_q`/`h` are always the caller's real, already-resolved region coefficients (`mu_q`
    is the fracture element's own quadrature-point viscosity - leak-off resistance is evaluated
    at the fracture-side fluid state, not the matrix side); `pressure_coupling`/`thermal_coupling`
    select which is actually used this call, matching `_integrate_fracture_matrix_coupling_tangent`.
    `p`/`T` are the full current-iterate primary-variable vectors, indexed here at each matched
    matrix element's own nodes (mirrors how `pe`/`Te` are sliced from them in the caller for the
    fracture element).

    Mirrors the tangent function's contract: returns `(fracture_vectors, matches)`, where
    `fracture_vectors` accumulates directly onto this fracture element's own residual
    ("F_Pdiffusion"/"F_Tdiffusion", like any other element vector) and `matches` is a list of
    `(nne_m, matrix_vectors)` - one entry per matched matrix element, each `matrix_vectors` dict
    keyed "F_Pdiffusion_mm"/"F_Tdiffusion_mm" for the caller to store into the cached
    `row_vec_flux_m`-indexed buffer, the vector analogue of the tangent's `_mm` matrix bucket.
    """
    fracture_vectors: dict[str, npt.NDArray[np.float64]] = {}
    matches: list[tuple[int, dict[str, npt.NDArray[np.float64]]]] = []
    if not pressure_coupling and not thermal_coupling:
        return fracture_vectors, matches

    p_f_q = N_q @ pe_f
    T_f_q = N_q @ Te_f
    Rf_p = np.zeros((len(enodes_f), 1))
    Rf_T = np.zeros((len(enodes_f), 1))

    for _e_m, enodes_m, nne_m, N_qm in _iter_fracture_matrix_matches(e, enodes_f, nodes_f, N_q, mesh):
        matrix_vectors: dict[str, npt.NDArray[np.float64]] = {}
        if pressure_coupling:
            p_m_q = N_qm @ p[enodes_m].reshape(-1, 1)
            flux_p_q = kf_p / mu_q * (p_f_q - p_m_q)
            Rf_p = Rf_p + _vctrzd_qp_vctr_integrate(N_q, flux_p_q, wJe_q, True)
            matrix_vectors["F_Pdiffusion_mm"] = _vctrzd_qp_vctr_integrate(N_qm, -flux_p_q, wJe_q, True)
        if thermal_coupling:
            T_m_q = N_qm @ T[enodes_m].reshape(-1, 1)
            flux_T_q = h * (T_f_q - T_m_q)
            Rf_T = Rf_T + _vctrzd_qp_vctr_integrate(N_q, flux_T_q, wJe_q, True)
            matrix_vectors["F_Tdiffusion_mm"] = _vctrzd_qp_vctr_integrate(N_qm, -flux_T_q, wJe_q, True)
        matches.append((nne_m, matrix_vectors))

    if pressure_coupling:
        fracture_vectors["F_Pdiffusion"] = Rf_p
    if thermal_coupling:
        fracture_vectors["F_Tdiffusion"] = Rf_T
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


def _stack_fracture_vectors(
    body_vectors: dict[str, csc_matrix[Any]],
    fracture_vectors: dict[str, csc_matrix[Any]],
) -> dict[str, csc_matrix[Any]]:
    """Stack matrix-mesh and fracture-mesh residual vectors into one `(nn_m + nn_f, 1)` vector.

    Unlike `_stack_fracture_matrices`, there is no off-diagonal block to build here: any
    fracture-matrix coupling contribution must already have been folded into the two vectors
    (matrix-side into `body_vectors`, fracture-side into `fracture_vectors`) before this runs.
    """
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
        w = mat["fracture"]["aperture"][r]
        lambda_f = mat["fluid"]["therm_cond"]
        lambda_e = lambda_f * w
        nusselt = mat["fracture"]["nusselt"][r]
        h = nusselt * lambda_f / 2 / w  # interface heat transfer coefficient, coupling only
        m_e = mat["fluid"]["density"] * mat["fluid"]["specific_heat"] * w
        return lambda_e, True, h, m_e, np.full((1, 1), np.nan)

    # Bulk case
    lambda_e = mat["solid"]["therm_cond_eff"][r]
    isotropic_cond = bool(isinstance(lambda_e, (float, int)))
    m_e = mat["solid"]["volumetric_heat_capacity_eff"][r]
    alpha_e = np.asarray(mat["solid"]["therm_expansion_eff"][r])
    return lambda_e, isotropic_cond, float("nan"), m_e, alpha_e


def _get_region_solid_data(
    mat: dict[str, Any], r: int
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
    K_d = _get_region_value(mat["solid"]["drained_bulk_mod"], r)
    phi_0 = _get_region_value(mat["solid"]["porosity"], r)
    biot = _get_region_value(mat["solid"]["biot_coeff"], r)
    rho_s = _get_region_value(mat["solid"]["density"], r)
    cp_s = _get_region_value(mat["solid"]["specific_heat"], r)
    alpha_s = _get_region_value(mat["solid"]["therm_expansion"], r)
    lambda_s = _get_region_value(mat["solid"]["therm_cond"], r)

    return K_d, phi_0, biot, rho_s, cp_s, alpha_s, lambda_s


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
    g, ng = _get_gravity(mesh["nsd"], controls["gravity"])

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
        # Pressure (leak-off) coupling is not attempted here: it needs a per-element viscosity,
        # which this state-independent, once-per-simulation pass has no p/T to evaluate - the
        # active (compressible) solvers recompute it fresh every Newton iteration regardless, in
        # integrate_residuals/integrate_tangent_matrices.
        list_of_ff_matrices = ["KTT_ff"]
        list_of_fm_matrices = ["KTT_fm"]
        list_of_mm_matrices = ["KTT_mm"]
        # This function's own (deprecated, soon-to-be-removed) row/col index vectors - local
        # to its own KTT_ff/fm/mm assembly below. The cached copy the *active* solvers rely on
        # (integrate_residuals/integrate_tangent_matrices) is built by integrate_residuals
        # itself, on demand, since it can no longer assume this function ran first.
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
            _vec_count_flux_m,
            _row_vec_flux_m,
        ) = _create_fracture_coupling_vectors(conn, conn_fm, mesh["conn"])

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
                "KM": _vctrzd_quad_point_integration(N_q, comp_f, N_q, wJe_q, True),
                "C": _vctrzd_quad_point_integration(N_q, m_e, N_q, wJe_q, True),
                "KTT": _vctrzd_quad_point_integration(B_q, lambda_e, B_q, wJe_q, isotropic_cond),
            }
            if not is_fracture:
                element_matrices["KpT"] = _vctrzd_quad_point_integration(N_q, alpha_e, N_q, wJe_q, True)
            if mu_linear:
                element_matrices["Kpp"] = _vctrzd_quad_point_integration(
                    B_q, kf / mu, B_q, wJe_q, isotropic_perm
                )
                # KbT is a scalar product once projected onto the fracture's tangent, a matmul
                # product for the bulk mesh - `is_fracture` doubles as that scalar/matmul flag.
                element_matrices["KbT"] = _vctrzd_quad_point_integration(N_q, kf_b, B_q, wJe_q, is_fracture)

            # ---- Assemble element contributions into global matrices
            stop = vec_counter + nne * nne  # vectorized index count end
            _store_vctrzd_element_matrix(vec, element_matrices, vec_counter, stop)
            vec_counter = stop

            # ---- Fractures
            if is_fracture:
                # Coupling matrices
                coupling_entries = _integrate_fracture_matrix_coupling_tangent(
                    e,
                    enodes,
                    nodes_f,
                    N_q,
                    wJe_q,
                    mesh,
                    kf_p,
                    float("nan"),  # unread: pressure_coupling is always off here, see above
                    h,
                    pressure_coupling=False,
                    thermal_coupling=True,
                )

                for nne_m, ff_matrices, fm_matrices, mm_matrices in coupling_entries:
                    stop = vec_counter_ff + nne * nne  # vectorized index count end
                    _store_vctrzd_element_matrix(vec_ff, ff_matrices, vec_counter_ff, stop)
                    vec_counter_ff = stop
                    stop = vec_counter_fm + nne_m * nne  # vectorized index count end
                    _store_vctrzd_element_matrix(vec_fm, fm_matrices, vec_counter_fm, stop)
                    vec_counter_fm = stop
                    stop = vec_counter_mm + nne_m * nne_m  # vectorized index count end
                    _store_vctrzd_element_matrix(vec_mm, mm_matrices, vec_counter_mm, stop)
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


def integrate_residuals(  # noqa
    mesh: dict[str, Any],
    mat: dict[str, Any],
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

    p_0 = mat["p_0"]
    T_0 = mat["T_0"]

    # density_nonlinear = True
    # fully_coupled = bool(controls["fully_coupled"])
    axisymmetry = bool(controls["axisymmetric_on"])
    fracture_on = bool(controls["fracture_on"])
    advection_on = bool(controls["advection_on"])
    volumetric_strain_on = bool(controls["volumetric_strain_on"])
    steady_pres = controls["steady_pressure"]
    steady_temp = controls["steady_temperature"]

    fluid = mat["fluid"]
    lambda_f = fluid["therm_cond"]
    # mu = fluid["viscosity"]
    # mu_q = mu
    # cp_f = fluid["specific_heat"]
    # rho_f0 = fluid["density"]
    # rho_f_q = rho_f0
    # if mu_nonlinear:
    #     alpha_f = fluid["therm_expansion"]

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

        # Connectivity-only (mesh-static) fracture/matrix coupling index vectors, built once and
        # cached on fmesh for reuse by every Newton iteration (this function's own later calls,
        # and integrate_tangent_matrices, which always runs after this one within an iteration)
        # rather than re-derived every time. integrate_residuals is the entry point that owns
        # this cache - it no longer assumes integrate_linear_matrices ran first to build it.
        if "vec_mat_asmbl" not in fmesh:
            (
                _vec_count_ff,
                row_vec_ff,
                col_vec_ff,
                _vec_count_fm,
                row_vec_fm,
                col_vec_fm,
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
                "row_vec_mm": row_vec_mm,
                "col_vec_mm": col_vec_mm,
                "row_vec_flux_m": row_vec_flux_m,
            }
        row_vec_flux_m = fmesh["vec_mat_asmbl"]["row_vec_flux_m"]
        coupling_flux_names = (["F_Pdiffusion_mm"] if not temperature_only else []) + (
            ["F_Tdiffusion_mm"] if not pressure_only else []
        )
        vec_flux_m = {name: np.zeros(row_vec_flux_m.size) for name in coupling_flux_names}
        vec_counter_flux_m = 0

    # ----- gravity vector
    g, ng = _get_gravity(mesh["nsd"], controls["gravity"])

    vec = _initiate_vectors(list_of_matrices, nne_list, elem_indx, ismatrix=False)
    vec_counter = 0  # vectorized storage index counter

    # ----- Matrix integration element loop
    for r in range(num_regions):
        kf_0, isotropic_perm, nodes_r, w0, u, kf_p, _ = _get_region_flow_data(
            mesh, mat, fmesh, nodes_f, is_fracture, r
        )
        # Gravity projected onto the fracture's own local (tangential) coordinate, matching
        # the reduced dimensionality of the fracture's own gradP/B_q - mirrors the buoyancy
        # projection `g.T @ u` already used for the fracture in integrate_linear_matrices.
        g_r = (u.T @ g) if is_fracture else g
        lambda_e, isotropic_cond, h, _, _ = _get_region_thermal_data(mat, is_fracture, r)
        K_d, phi_0, biot, rho_s, cp_s, alpha_s, lambda_s = _get_region_solid_data(mat, r)

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

            Te = T[enodes_dof]  # element nodal temperatures
            Te_flow = T_flow[enodes_dof]
            pe = p[enodes_dof]  # element nodal pressures
            T_0e = T_0[enodes_dof].reshape(-1, 1)  # element reference temperatures
            p_0e = p_0[enodes_dof].reshape(-1, 1)  # element reference pressures
            L_vec = (xI.max(axis=0) - xI.min(axis=0)).reshape(-1, 1)  # element lengths # noqa

            T_q = N_q @ Te  # quadrature point temperatures, nq x 1 x (n, n-1, n-2, n-3)
            T_flow_q = N_q @ Te_flow  # quadrature point temperatures, nq x 1 x (n, n-1, n-2, n-3)
            P_q = N_q @ pe  # quadrature point pressures, nq x 1 x (n, n-1, n-2, n-3)
            gradT = B_q @ Te[:, 0].reshape(-1, 1)  # qudarature point temperature gradient, nq x nsd x 1
            gradP = B_q @ pe[:, 0].reshape(-1, 1)  # quadrature point pressure gradient, nq x nsd x 1

            # quadrature point permeabilities
            kf, _, _ = get_effective_permeability(kf_0, T_flow_q[:, :, 0], P_q[:, :, 0], fluid)
            # quadrature point viscosities, nq x 1 x 1
            mu_q, _, _ = get_viscosity(T_flow_q[:, :, 0], P_q[:, :, 0], fluid)
            mu_q = mu_q.reshape(-1, 1, 1)
            # quadrature point densities, nq x 1 x (n, n-1, n-2, n-3)
            rho_f_q, _, _ = get_density(T_flow_q, P_q, fluid)
            rho_f_qn = rho_f_q[:, :, 0].reshape(-1, 1, 1)

            darcy_flux = (
                -kf / mu_q * (gradP - rho_f_qn * g_r)
                if isotropic_perm
                else -kf / mu_q @ (gradP - rho_f_qn * g_r)
            )

            if is_fracture:
                # For the fracture case, the phi terms are effectively replaced by aperture
                phi_L = w0 * np.ones_like(T_q)
            else:
                # Matrix case
                phi_L, _, _ = get_porosity_lagrangian(
                    P_q,
                    N_q @ p_0e,
                    T_flow_q,
                    N_q @ T_0e,
                    phi_0,
                    K_d,
                    biot,
                    alpha_s,
                    volumetric_strain_on=volumetric_strain_on,
                )

            # Nodal evaluation for self-consistent lumping of the storage terms (see
            # integrate_tangent_matrices for the matching tangent-side computation).
            if is_fracture:
                # Fracture case
                phi_L_nodal = w0 * np.ones((nne, 4))  # nne x 1 x (n, n-1, n-2, n-3)
            else:
                # Matrix case
                phi_L_nodal, _, _ = get_porosity_lagrangian(
                    pe,
                    p_0e,
                    Te_flow,
                    T_0e,
                    phi_0,
                    K_d,
                    biot,
                    alpha_s,
                    volumetric_strain_on=volumetric_strain_on,
                )
            rho_f_nodal, _, _ = get_density(Te_flow, pe, fluid)
            m_lump = _vctrzd_quad_point_integration(N_q, 1.0, N_q, wJe_q, True).sum(axis=1).reshape(-1, 1)

            # ------ Conservation of mass
            if not temperature_only:
                # Storage vector (lumped)
                f_mass_nodal = phi_L_nodal * rho_f_nodal
                mass_dot_nodal = (1 - steady_pres) * (
                    cv[0] * f_mass_nodal[:, 0:1] + cv[1] * f_mass_nodal[:, 1:2] + cv[2] * f_mass_nodal[:, 2:3]
                )
                element_vectors["F_Pstorage"] = m_lump * mass_dot_nodal

                # Diffusion vector
                element_vectors["F_Pdiffusion"] = _vctrzd_qp_vctr_integrate(
                    B_q, rho_f_qn * darcy_flux, wJe_q, isotropic_perm
                )

            # ------ Conservation of energy
            if not pressure_only:
                # Thermo-dynamic internal variables
                u_fq, _, _ = get_internal_energy(T_q, P_q, fluid)  # fluid internal energy, nq x 1 x 1
                h_fq, _, _, cp_fq = get_enthalpy(
                    P_q[:, :, 0:1], T_q[:, :, 0:1], T_flow_q[:, :, 0:1], fluid
                )  # fluid enthalpy, nq x 1 x 1
                # total energy, nq x 1 x 1
                E_q = (1 - phi_L) * rho_s * cp_s * T_q * (1 - is_fracture) + phi_L * rho_f_q * u_fq
                energy_dot_q = (1 - steady_temp) * (
                    cv[0] * E_q[:, :, 0:1] + cv[1] * E_q[:, :, 1:2] + cv[2] * E_q[:, :, 2:3]
                )

                # Effective conductivity
                phi_L_n = phi_L[:, :, 0].reshape(-1, 1, 1)
                if is_fracture:
                    lambda_e = w0 * lambda_f
                else:
                    lambda_e = (
                        phi_L_n * lambda_f + (1 - phi_L_n) * lambda_s
                        if isotropic_cond
                        else (phi_L_n * lambda_f * np.eye(nsd) + (1 - phi_L_n) * lambda_s)
                    )

                u_f_nodal, _, _ = get_internal_energy(Te, pe, fluid)
                E_nodal = (1 - phi_L_nodal) * rho_s * cp_s * Te * (
                    1 - is_fracture
                ) + phi_L_nodal * rho_f_nodal * u_f_nodal
                energy_dot_nodal = (1 - steady_temp) * (
                    cv[0] * E_nodal[:, 0:1] + cv[1] * E_nodal[:, 1:2] + cv[2] * E_nodal[:, 2:3]
                )
                element_vectors["F_Tstorage"] = m_lump * energy_dot_nodal

                # Diffusion vector
                element_vectors["F_Tdiffusion"] = _vctrzd_qp_vctr_integrate(
                    B_q, lambda_e * gradT, wJe_q, isotropic_cond
                )

                # Advection and SUPG stabilization
                if advection_on:
                    flux = rho_f_qn * cp_fq * darcy_flux
                    if nsd == 3:
                        flux_mag = np.sqrt(flux[:, 0, 0] ** 2 + flux[:, 1, 0] ** 2 + flux[:, 2, 0] ** 2)
                    elif nsd == 2:
                        flux_mag = np.sqrt(flux[:, 0, 0] ** 2 + flux[:, 1, 0] ** 2)
                    else:
                        # Keep flux_mag 1D (nq,) like the nsd 2/3 branches above, not (nq, 1, 1) -
                        # `flux[indx_noflow, :, :]` below silently corrupts unrelated entries when
                        # indx_noflow comes from nonzero() on a >1D array (a 3-tuple of index
                        # arrays gets nested inside the outer index tuple instead of used directly).
                        flux_mag = np.abs(flux[:, 0, 0])
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
                        dl_eff = flux_mag / np.sqrt(
                            (flux[:, 0, 0] / L_vec[0, 0]) ** 2 + (flux[:, 1, 0] / L_vec[1, 0]) ** 2
                        )
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
                    hf_e, _, _, _ = get_enthalpy(
                        pe[:, 0].reshape(-1, 1), Te[:, 0].reshape(-1, 1), Te_flow[:, 0].reshape(-1, 1), fluid
                    )  # fluid enthalpy, nq x 1 x 1
                    grad_hf = B_q @ hf_e
                    rhof_e, _, _ = get_density(
                        Te_flow[:, 0].reshape(-1, 1), pe[:, 0].reshape(-1, 1), fluid
                    )  # fluid density, nq x 1 x 1
                    grad_rhof = B_q @ rhof_e
                    RTh = (
                        energy_dot_q + (grad_hf * rho_f_qn + h_fq * grad_rhof).transpose(0, 2, 1) @ darcy_flux
                    )  # discretized residual

                    # advection (subtracted at the residual level, like the mass flux term) and
                    # its SUPG stabilization (a Petrov-Galerkin test-function correction, added
                    # with its own independent sign - not part of the divergence term being
                    # integrated by parts, so it does not flip sign along with plain advection)
                    integrand_advection = rho_f_qn * darcy_flux * h_fq
                    integrand_SUPG = cp_fq * rho_f_qn * darcy_flux * tau * RTh
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
                    pe[:, 0:1],
                    Te[:, 0:1],
                    p[:, 0],
                    T[:, 0],
                    kf_p,
                    mu_q,
                    h,
                    pressure_coupling=not temperature_only,
                    thermal_coupling=not pressure_only,
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
        # via the cached row_vec_flux_m (duplicate row entries at shared matrix nodes are summed
        # by the sparse constructor) into a (nn_m, 1) vector, for the caller to fold into its own
        # F_P/Tdiffusion - the vector analogue of how the tangent's "_mm" matrices are assembled.
        for name in vec_flux_m:
            integrated_vectors[name] = _make_sparse(
                vec_flux_m[name], row_vec_flux_m, np.zeros_like(row_vec_flux_m), nn_m, 1
            )
        return integrated_vectors

    if fracture_on:
        # Recursion call
        integrated_fracture_vectors = integrate_residuals(
            mesh,
            mat,
            primary_variables,
            controls,
            cv,
            T_flow,
            is_fracture=True,
            pressure_only=pressure_only,
            temperature_only=temperature_only,
        )
        # The fracture pass also returns the matrix-side half of the fracture-matrix
        # leak-off/convective-exchange flux under a "_mm" suffix, sized to the matrix mesh
        # rather than the fracture mesh - fold each into the matching matrix-side residual
        # before stacking the two meshes together below.
        for name in ("F_Pdiffusion", "F_Tdiffusion"):
            if name + "_mm" in integrated_fracture_vectors:
                flux_mm = integrated_fracture_vectors[name + "_mm"]
                integrated_vectors[name] = integrated_vectors[name] + flux_mm
        # Combine fracture and matrix vectors into one
        integrated_vectors = _stack_fracture_vectors(integrated_vectors, integrated_fracture_vectors)

    return integrated_vectors


def integrate_enthlpy_bndry_residual(
    mesh: dict[str, Any],
    mat: dict[str, Any],
    primary_variables: dict[str, Any],
    controls: dict[str, Any],
    T_flow: npt.NDArray[np.float64],  # temperature field used for flow-property evaluations
    BC_config: list[dict[str, Any]],
) -> dict[str, Any]:
    """Integrate the advective enthalpy flux boundary vector.

    Only applies to fluid BC without an associated temperature BC. "q" already
    carries both flux-type (via F_extp) and pressure-type (via the Dirichlet
    reaction force) nodal mass flow - see solvers.py's `q[nF_p] += F_extp[nF_p]`
    - so both bc_types are handled by the same node-value lookup.
    """
    advection_off = not bool(controls["advection_on"])  # noqa
    nn_total = mesh["nn_total"]

    if advection_off:
        return {"F_enth_bnd": csc_matrix((nn_total, 1), dtype=float)}

    F_enth_bnd = np.zeros((nn_total, 1))
    fluid = mat["fluid"]
    T = primary_variables["T"][:, 0:1]
    p = primary_variables["p"][:, 0:1]
    q = primary_variables["q"][:, 0:1]
    # Collected keyed by node (not scattered in place) so a node referenced more
    # than once - two line segments sharing an endpoint, or two groups sharing a
    # node - is idempotent instead of double-counted.
    node_value: dict[int, float] = {}

    for group in BC_config:
        if "hydraulics" in group and "thermal" not in group and "_conn" in group:
            nodes_i = np.asarray(group["nodes"], dtype=np.int32)
            q_i = q[nodes_i, 0]
            T_i = T[nodes_i, 0]
            T_flow_i = T_flow[nodes_i, 0]
            p_i = p[nodes_i, 0]
            h_f_i, _, _, _ = get_enthalpy(p_i, T_i, T_flow_i, fluid)
            node_value.update(zip(nodes_i.tolist(), (q_i * h_f_i).tolist()))

    if node_value:
        node_ids = np.fromiter(node_value.keys(), dtype=np.int32)
        F_enth_bnd[node_ids, 0] = np.fromiter(node_value.values(), dtype=np.float64)

    return {"F_enth_bnd": csc_matrix(F_enth_bnd)}


def integrate_enthlpy_bndry_tangent(
    mesh: dict[str, Any],
    mat: dict[str, Any],
    primary_variables: dict[str, Any],
    controls: dict[str, Any],
    T_flow: npt.NDArray[np.float64],  # temperature field used for flow-property evaluations
    BC_config: list[dict[str, Any]],
    *,
    fully_coupled: bool = False,
    Kpp: csc_matrix[Any] = csc_matrix((0, 0)),
    KpT: csc_matrix[Any] = csc_matrix((0, 0)),
) -> dict[str, Any]:
    """Integrate the advective enthalpy flux boundary tangent.

    Only applies to fluid BC without an associated temperature BC; see the
    residual function for why "q" unifies both bc_types. d(F_enth_bnd)/dT is
    q*dhf_dT (diagonal) plus h_f*dq/dT, which is zero unless q itself depends on
    T/p - true for a Dirichlet-pressure node under `fully_coupled`, where that
    dependence is exactly the rows of "Kpp"/"KpT" (the pressure equation's own
    Jacobian blocks, already assembled every iteration by
    integrate_tangent_matrices(fully_coupled=True)) at that node. Callers not
    passing `fully_coupled` never touch Kpp/KpT.
    """
    advection_off = not bool(controls["advection_on"])  # noqa
    nn_total = mesh["nn_total"]

    if advection_off:
        return {"K_enth_bnd": csc_matrix((nn_total, nn_total), dtype=float)}

    K_enth_bnd = csc_matrix((nn_total, nn_total), dtype=float)
    K_enth_bnd_Tp = csc_matrix((nn_total, nn_total), dtype=float)
    fluid = mat["fluid"]
    T = primary_variables["T"][:, 0:1]
    p = primary_variables["p"][:, 0:1]
    q = primary_variables["q"][:, 0:1]
    # See the residual function for why contributions are collected (keyed by
    # node) rather than scattered in place.
    node_diag_value: dict[int, float] = {}

    for group in BC_config:
        if "hydraulics" in group and "thermal" not in group and "_conn" in group:
            nodes_i = np.asarray(group["nodes"], dtype=np.int32)
            q_i = q[nodes_i, 0]
            T_i = T[nodes_i, 0]
            T_flow_i = T_flow[nodes_i, 0]
            p_i = p[nodes_i, 0]
            h_f_i, dhf_dT_i, _, _ = get_enthalpy(p_i, T_i, T_flow_i, fluid)
            node_diag_value.update(zip(nodes_i.tolist(), (q_i * dhf_dT_i).tolist()))

            # A flux-type node's q is a prescribed BC value (never a function of
            # p/T); only a pressure-type node's q (a mass-residual row) needs the
            # Kpp/KpT cross-terms.
            if fully_coupled and group["hydraulics"]["type"] == "pressure":
                rows = KpT[nodes_i, :].tocoo()
                K_enth_bnd = K_enth_bnd + csc_matrix(
                    (rows.data * h_f_i[rows.row], (nodes_i[rows.row], rows.col)),
                    shape=(nn_total, nn_total),
                )
                rows = Kpp[nodes_i, :].tocoo()
                K_enth_bnd_Tp = K_enth_bnd_Tp + csc_matrix(
                    (rows.data * h_f_i[rows.row], (nodes_i[rows.row], rows.col)),
                    shape=(nn_total, nn_total),
                )

    if node_diag_value:
        node_ids = np.fromiter(node_diag_value.keys(), dtype=np.int32)
        diag_vals = np.fromiter(node_diag_value.values(), dtype=np.float64)
        K_enth_bnd = K_enth_bnd + csc_matrix((diag_vals, (node_ids, node_ids)), shape=(nn_total, nn_total))

    return {"K_enth_bnd": K_enth_bnd, "K_enth_bnd_Tp": K_enth_bnd_Tp}


def integrate_tangent_matrices(  # noqa
    mesh: dict[str, Any],
    mat: dict[str, Any],
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
    coupled to the unknown T (False - T_flow tracks T, e.g. the fully-coupled solver) or
    independent of it (True - T_flow is a frozen/extrapolated field, e.g. the linearized
    solver's temperature-only solve), which controls whether dqD/dT picks up the resulting
    dmu/dT, drho/dT, dkf/dT contributions.
    """
    p = primary_variables["p"]
    T = primary_variables["T"]

    p_0 = mat["p_0"]
    T_0 = mat["T_0"]

    # fully_coupled = bool(controls["fully_coupled"])
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

    fluid = mat["fluid"]
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

        # Connectivity-only (mesh-static) index vectors, built by integrate_residuals (which
        # always runs before this function within a Newton iteration) rather than re-derived
        # here every timestep/iteration. The coupling coefficients themselves (leak-off
        # resistance, convective exchange) are always re-evaluated below, since they depend on
        # the current fracture-side viscosity.
        cached_cv = fmesh["vec_mat_asmbl"]
        row_vec_ff, col_vec_ff = cached_cv["row_vec_ff"], cached_cv["col_vec_ff"]
        row_vec_fm, col_vec_fm = cached_cv["row_vec_fm"], cached_cv["col_vec_fm"]
        row_vec_mm, col_vec_mm = cached_cv["row_vec_mm"], cached_cv["col_vec_mm"]

        coupling_ff_names = (["Kpp_ff"] if not temperature_only else []) + (
            ["KTT_ff"] if not pressure_only else []
        )
        coupling_fm_names = (["Kpp_fm"] if not temperature_only else []) + (
            ["KTT_fm"] if not pressure_only else []
        )
        coupling_mm_names = (["Kpp_mm"] if not temperature_only else []) + (
            ["KTT_mm"] if not pressure_only else []
        )
        vec_ff = {name: np.zeros(row_vec_ff.size) for name in coupling_ff_names}
        vec_fm = {name: np.zeros(row_vec_fm.size) for name in coupling_fm_names}
        vec_mm = {name: np.zeros(row_vec_mm.size) for name in coupling_mm_names}

        vec_counter_ff = 0
        vec_counter_fm = 0
        vec_counter_mm = 0

    # ----- gravity vector
    g, ng = _get_gravity(mesh["nsd"], controls["gravity"])

    vec = _initiate_vectors(list_of_matrices, nne_list, elem_indx)
    vec_counter = 0  # vectorized storage index counter

    # ----- Matrix integration element loop
    for r in range(num_regions):
        kf_0, isotropic_perm, nodes_r, w0, u, kf_p, _ = _get_region_flow_data(
            mesh, mat, fmesh, nodes_f, is_fracture, r
        )
        # Gravity projected onto the fracture's own local (tangential) coordinate, matching
        # the reduced dimensionality of the fracture's own gradP/B_q - mirrors the buoyancy
        # projection `g.T @ u` already used for the fracture in integrate_linear_matrices.
        g_r = (u.T @ g) if is_fracture else g
        lambda_e, isotropic_cond, h, _, _ = _get_region_thermal_data(mat, is_fracture, r)
        K_d, phi_0, biot, rho_s, cp_s, alpha_s, lambda_s = _get_region_solid_data(mat, r)

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

            Te = T[enodes_dof]  # element nodal temperatures
            Te_flow = T_flow[enodes_dof]  # element nodal temperatures, for flow properties
            pe = p[enodes_dof]  # element nodal pressures
            T_0e = T_0[enodes_dof].reshape(-1, 1)  # element reference temperatures
            p_0e = p_0[enodes_dof].reshape(-1, 1)  # element reference pressures
            L_vec = (xI.max(axis=0) - xI.min(axis=0)).reshape(-1, 1)  # element lengths # noqa

            T_q = N_q @ Te  # quadrature point temperatures, nq x 1 x (n, n-1, n-2, n-3)
            T_flow_q = N_q @ Te_flow  # quadrature point temperatures, for flow properties
            P_q = N_q @ pe  # quadrature point pressures, nq x 1 x (n, n-1, n-2, n-3)
            gradT = B_q @ Te[:, 0].reshape(-1, 1)  # qudarature point temperature gradient, nq x nsd x 1
            gradP = B_q @ pe[:, 0].reshape(-1, 1)  # quadrature point pressure gradient, nq x nsd x 1

            # quadrature point permeabilities
            kf, dkf_dT, dkf_dp = get_effective_permeability(
                kf_0, T_flow_q[:, :, 0], P_q[:, :, 0], fluid, T_flow_lagged
            )
            dkf_dT = dkf_dT.reshape(-1, 1, 1)
            dkf_dp = dkf_dp.reshape(-1, 1, 1)

            # quadrature point viscosities, nq x 1 x 1
            mu_q, dmu_dT, dmu_dp = get_viscosity(T_flow_q[:, :, 0], P_q[:, :, 0], fluid, T_flow_lagged)
            mu_q = mu_q.reshape(-1, 1, 1)
            dmu_dT = dmu_dT.reshape(-1, 1, 1)
            dmu_dp = dmu_dp.reshape(-1, 1, 1)
            # quadrature point densities, nq x 1 x (n, n-1, n-2, n-3)
            rho_f, drho_dT, drho_dp = get_density(T_flow_q, P_q, fluid, T_flow_lagged)
            rho_f_n = rho_f[:, :, 0].reshape(-1, 1, 1)
            drho_dT = drho_dT[:, :, 0].reshape(-1, 1, 1)
            drho_dp = drho_dp[:, :, 0].reshape(-1, 1, 1)

            gradP_mod = gradP - rho_f_n * g_r

            # Darcy flux
            darcy_flux = -kf / mu_q * gradP_mod if isotropic_perm else -kf / mu_q @ gradP_mod
            # Darcy mass flux
            q_D = rho_f_n * darcy_flux
            # porosity - for the fracture case, the phi terms are effectively replaced by
            # aperture, and (being a fixed material property) do not depend on p or T.
            if is_fracture:
                phi_L = w0 * np.ones_like(T_q)
                dphi_L_dT = np.zeros((nq, 1, 1))
                dphi_L_dp = np.zeros((nq, 1, 1))
            else:
                phi_L, dphi_L_dT, dphi_L_dp = get_porosity_lagrangian(
                    P_q,
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
                dphi_L_dT = dphi_L_dT[:, :, 0].reshape(-1, 1, 1)
                dphi_L_dp = dphi_L_dp[:, :, 0].reshape(-1, 1, 1)
            phi_L_n = phi_L[:, :, 0].reshape(-1, 1, 1)

            # Nodal evaluation for self-consistent lumping of the storage terms: density,
            # porosity, and internal energy are evaluated at each node's own (T, p) rather
            # than at quadrature points, so the lumped storage residual/tangent are exact
            # derivatives of one another by construction.
            if is_fracture:
                phi_L_nodal = w0 * np.ones((nne, 1))
                dphi_L_dT_nodal = np.zeros((nne, 1))
                dphi_L_dp_nodal = np.zeros((nne, 1))
            else:
                phi_L_nodal, dphi_L_dT_nodal, dphi_L_dp_nodal = get_porosity_lagrangian(
                    pe[:, 0:1],
                    p_0e,
                    Te_flow[:, 0:1],
                    T_0e,
                    phi_0,
                    K_d,
                    biot,
                    alpha_s,
                    T_flow_lagged,
                    volumetric_strain_on,
                )
            rho_f_nodal, drho_dT_nodal, drho_dp_nodal = get_density(
                Te_flow[:, 0:1], pe[:, 0:1], fluid, T_flow_lagged
            )
            m_lump = _vctrzd_quad_point_integration(N_q, 1.0, N_q, wJe_q, True).sum(axis=1)

            # dqD/dp
            if isotropic_perm:
                dqD_dp = (
                    rho_f_n * kf / mu_q * (B_q + drho_dp * N_q)
                    + (kf / mu_q * drho_dp + rho_f_n / mu_q * dkf_dp + rho_f_n * kf / mu_q**2 * dmu_dp)
                    * gradP_mod
                    * N_q
                )
            else:
                dqD_dp = (
                    rho_f_n * kf / mu_q @ (B_q + drho_dp * N_q)
                    + (kf / mu_q * drho_dp + rho_f_n / mu_q * dkf_dp + rho_f_n * kf / mu_q**2 * dmu_dp)
                    @ gradP_mod
                    * N_q
                )

            # ------------------------------------ Conservation of mass wrt pressure
            if not temperature_only:
                # Storage vector - Fs (lumped: nodal dFs/dp times the lumped mass weight)
                integrand_dFsdp_nodal = (
                    (1 - steady_pres) * cv[0] * (dphi_L_dp_nodal * rho_f_nodal + phi_L_nodal * drho_dp_nodal)
                )
                element_matrices["Kpp"] = np.diag(m_lump * integrand_dFsdp_nodal.ravel())

                # Diffusion vector - Fk
                # dFk/dp
                element_matrices["Kpp"] += _vctrzd_qp_vctr_integrate(B_q, dqD_dp, wJe_q, False)

            if not pressure_only:
                # -------------------------------- Conservation of energy wrt temperature
                # Thermo-dynamic internal variables
                u_f, duf_dT, duf_dp = get_internal_energy(
                    T_q, P_q, fluid
                )  # fluid internal energy, nq x 1 x 1
                u_f_n = u_f[:, :, 0:1]
                duf_dT_n = duf_dT[:, :, 0:1]
                duf_dp_n = duf_dp[:, :, 0:1]
                h_f, dhf_dT, dhf_dp, cp_f = get_enthalpy(
                    P_q[:, :, 0:1], T_q[:, :, 0:1], T_flow_q[:, :, 0:1], fluid, T_flow_lagged
                )  # fluid enthalpy, nq x 1 x 1
                E_q = (1 - phi_L) * rho_s * cp_s * T_q * (
                    1 - is_fracture
                ) + phi_L * rho_f * u_f  # total energy, nq x 1 x 1
                energy_dot = (1 - steady_temp) * (
                    cv[0] * E_q[:, :, 0:1] + cv[1] * E_q[:, :, 1:2] + cv[2] * E_q[:, :, 2:3]
                )

                # Effective conductivity
                if is_fracture:
                    lambda_e = w0 * lambda_f
                else:
                    lambda_e = (
                        phi_L_n * lambda_f + (1 - phi_L_n) * lambda_s
                        if isotropic_cond
                        else (phi_L_n * lambda_f * np.eye(nsd) + (1 - phi_L_n) * lambda_s)
                    )

                # Energy storage vector - FC (lumped: nodal dFC/dT times the lumped mass weight)
                u_f_nodal, duf_dT_nodal, _ = get_internal_energy(Te[:, 0:1], pe[:, 0:1], fluid)
                integrand_dFCdT_nodal = (
                    (1 - steady_temp)
                    * cv[0]
                    * (
                        (1 - phi_L_nodal) * rho_s * cp_s * (1 - is_fracture)
                        + phi_L_nodal * (drho_dT_nodal * u_f_nodal + rho_f_nodal * duf_dT_nodal)
                        + dphi_L_dT_nodal * (rho_f_nodal * u_f_nodal - rho_s * cp_s * Te[:, 0:1])
                    )
                )
                element_matrices["KTT"] = np.diag(m_lump * integrand_dFCdT_nodal.ravel())

                # Themal diffusion vector
                integrand_dFlambdadT = lambda_e * B_q + dphi_L_dT * (lambda_f - lambda_s) * gradT * N_q
                element_matrices["KTT"] += _vctrzd_qp_vctr_integrate(B_q, integrand_dFlambdadT, wJe_q, False)

                # Advection and SUPG stabilization
                if advection_on or fully_coupled:
                    # dqD/dT
                    if isotropic_perm:
                        dqD_dT = (
                            rho_f_n * kf / mu_q * (drho_dT * g_r * N_q)
                            + (
                                kf / mu_q * drho_dT
                                + rho_f_n / mu_q * dkf_dT
                                + rho_f_n * kf / mu_q**2 * dmu_dT
                            )
                            * gradP_mod
                            * N_q
                        )
                    else:
                        dqD_dT = (
                            rho_f_n * kf / mu_q @ (drho_dT * g_r * N_q)
                            + (
                                kf / mu_q * drho_dT
                                + rho_f_n / mu_q * dkf_dT
                                + rho_f_n * kf / mu_q**2 * dmu_dT
                            )
                            @ gradP_mod
                            * N_q
                        )

                if advection_on:
                    # SUPG
                    flux = rho_f_n * cp_f * darcy_flux
                    if nsd == 3:
                        flux_mag = np.sqrt(flux[:, 0, 0] ** 2 + flux[:, 1, 0] ** 2 + flux[:, 2, 0] ** 2)
                    elif nsd == 2:
                        flux_mag = np.sqrt(flux[:, 0, 0] ** 2 + flux[:, 1, 0] ** 2)
                    else:
                        # Keep flux_mag 1D (nq,) like the nsd 2/3 branches above, not (nq, 1, 1) -
                        # `flux[indx_noflow, :, :]` below silently corrupts unrelated entries when
                        # indx_noflow comes from nonzero() on a >1D array (a 3-tuple of index
                        # arrays gets nested inside the outer index tuple instead of used directly).
                        flux_mag = np.abs(flux[:, 0, 0])
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
                        dl_eff = flux_mag / np.sqrt(
                            (flux[:, 0, 0] / L_vec[0, 0]) ** 2 + (flux[:, 1, 0] / L_vec[1, 0]) ** 2
                        )
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
                    hf_e, _, _, _ = get_enthalpy(
                        pe[:, 0].reshape(-1, 1), Te[:, 0].reshape(-1, 1), Te_flow[:, 0].reshape(-1, 1), fluid
                    )  # fluid enthalpy, nq x 1 x 1
                    grad_hf = B_q @ hf_e
                    rhof_e, _, _ = get_density(
                        Te_flow[:, 0].reshape(-1, 1), pe[:, 0].reshape(-1, 1), fluid
                    )  # fluid density, nq x 1 x 1
                    grad_rhof = B_q @ rhof_e
                    RTh = (
                        energy_dot + (grad_hf * rho_f_n + h_f * grad_rhof).transpose(0, 2, 1) @ darcy_flux
                    )  # discretized residual

                    D2 = grad_rhof.transpose(0, 2, 1) @ q_D  # d/dTe of (grad_rhof . q_D), nq x 1 x 1
                    dR_dT = (
                        (1 - steady_temp)
                        * cv[0]
                        * (
                            (1 - phi_L_n) * rho_s * cp_s * (1 - is_fracture)
                            + dphi_L_dT * (rho_f_n * u_f_n - rho_s * cp_s * T_q[:, :, 0:1])
                            + phi_L_n * (drho_dT * u_f_n + rho_f_n * duf_dT_n)
                        )
                        * N_q
                        + (dhf_dT + h_f / rho_f_n * drho_dT) * (q_D.transpose(0, 2, 1) @ B_q)
                        + (grad_hf + h_f * grad_rhof / rho_f_n).transpose(0, 2, 1) @ dqD_dT
                        + (dhf_dT / rho_f_n - h_f * drho_dT / rho_f_n**2) * D2 * N_q
                    )

                    # Combined integrands
                    integrand_advection = dhf_dT * q_D * N_q + h_f * dqD_dT
                    integrand_SUPG = tau * cp_f * dR_dT * q_D + tau * cp_f * RTh * dqD_dT
                    element_matrices["KTT"] -= _vctrzd_qp_vctr_integrate(
                        B_q, integrand_advection, wJe_q, False
                    )
                    element_matrices["KTT"] += _vctrzd_qp_vctr_integrate(B_q, integrand_SUPG, wJe_q, False)

            # Fully coupled - off diagonal matrices
            if fully_coupled:
                # ----------- Conservation of mass wrt temperature
                # Storage vector - Fs
                integrand_dFsdT = (1 - steady_pres) * cv[0] * (dphi_L_dT * rho_f_n + phi_L_n * drho_dT)
                element_matrices["KpT"] = _vctrzd_quad_point_integration(
                    N_q, integrand_dFsdT, N_q, wJe_q, scalar=True
                )

                # Diffusion vector - Fk
                integrand_dFkdT = dqD_dT
                element_matrices["KpT"] += _vctrzd_qp_vctr_integrate(B_q, integrand_dFkdT, wJe_q, False)

                # ----------- Conservation of energy wrt pressure
                # Energy storage vector - FC
                integrand_dFCdT = (
                    (1 - steady_temp)
                    * cv[0]
                    * (
                        phi_L_n * (drho_dp * u_f_n + rho_f_n * duf_dp_n)
                        + dphi_L_dp * (rho_f_n * u_f_n - rho_s * cp_s * T_q[:, :, 0:1])
                    )
                )
                element_matrices["KTp"] = _vctrzd_quad_point_integration(
                    N_q, integrand_dFCdT, N_q, wJe_q, scalar=True
                )

                # Themal diffusion vector
                integrand_dFlambdadT = dphi_L_dp * (lambda_f - lambda_s) * gradT * N_q
                element_matrices["KTp"] += _vctrzd_qp_vctr_integrate(B_q, integrand_dFlambdadT, wJe_q, False)

                if advection_on:
                    dR_dp = (
                        (1 - steady_temp)
                        * cv[0]
                        * (
                            dphi_L_dp * (rho_f_n * u_f_n - rho_s * cp_s * T_q[:, :, 0:1])
                            + phi_L_n * (drho_dp * u_f_n + rho_f_n * duf_dp_n)
                        )
                        * N_q
                        + (dhf_dp + h_f / rho_f_n * drho_dp) * q_D * B_q
                        + (grad_hf + h_f * grad_rhof / rho_f_n) * dqD_dp
                        + (dhf_dp * grad_rhof / rho_f_n - h_f * grad_rhof / rho_f_n**2 * drho_dp)
                    )

                    # Advection tangent matrices
                    integrand_advection = dhf_dp * q_D * N_q + h_f * dqD_dp
                    integrand_SUPG = tau * cp_f * dR_dp * q_D + tau * cp_f * RTh * dqD_dp
                    element_matrices["KTp"] -= _vctrzd_qp_vctr_integrate(
                        B_q, integrand_advection, wJe_q, False
                    )
                    element_matrices["KTp"] += _vctrzd_qp_vctr_integrate(B_q, integrand_SUPG, wJe_q, False)

            # ----- Fracture-matrix leak-off/convective-exchange coupling
            if is_fracture:
                coupling_entries = _integrate_fracture_matrix_coupling_tangent(
                    e,
                    enodes,
                    nodes_f,
                    N_q,
                    wJe_q,
                    mesh,
                    kf_p,
                    mu_q,
                    h,
                    pressure_coupling=not temperature_only,
                    thermal_coupling=not pressure_only,
                )

                for nne_m, ff_matrices, fm_matrices, mm_matrices in coupling_entries:
                    stop = vec_counter_ff + nne * nne  # vectorized index count end
                    _store_vctrzd_element_matrix(vec_ff, ff_matrices, vec_counter_ff, stop)
                    vec_counter_ff = stop
                    stop = vec_counter_fm + nne_m * nne  # vectorized index count end
                    _store_vctrzd_element_matrix(vec_fm, fm_matrices, vec_counter_fm, stop)
                    vec_counter_fm = stop
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
        # "Kpp_ff/fm/mm"/"KTT_ff/fm/mm" keys line up with the plain "Kpp"/"KTT" keys already in
        # `list_of_matrices` above, so `_stack_fracture_matrices` merges them in directly.
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
        integrated_fracture_matrices = integrate_tangent_matrices(
            mesh,
            mat,
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


def integrate_nonlinear_matrices(  # noqa
    mesh: dict[str, Any],
    mat: dict[str, Any],
    primary_variables: dict[str, Any],
    controls: dict[str, Any],
    *,
    is_fracture: bool = False,
    pressure_only: bool = False,
) -> dict[str, Any]:
    """Integrate non-linear matrices inside time integration loop.

    Returns "K_{block}_R"/"K_{block}_tan" per equation block ("pp", "pT", "TT", "Tp"): the
    residual-evaluation matrix and the running sum of tangent-only corrections, signs baked in
    so `K_{block}_R + K_{block}_tan` is the block's tangent contribution directly.

    When called normally (`is_fracture=False`) and `fracture_on`, this recurses once onto
    the fracture mesh (`is_fracture=True`).

    `pressure_only` skips the TT advection/SUPG assembly, the viscosity-temperature tangent
    correction, and the fully-coupled off-diagonal Tp terms - for use inside a pressure-only
    Newton-Raphson loop where only "K_pp_R"/"K_pp_tan"/"K_pT_R" are needed.
    """
    p = primary_variables["p"][:, 0]
    T = primary_variables["T"][:, 0]
    mu_nonlinear = bool(controls["viscosity_nonconstant"])
    mu_linear = not mu_nonlinear
    density_nonlinear = bool(controls["density_nonconstant"])
    # density_nonlinear = True
    density_linear = not density_nonlinear
    fully_coupled = bool(controls["fully_coupled"])
    axisymmetry = bool(controls["axisymmetric_on"])
    fracture_on = bool(controls["fracture_on"])
    advection_on = bool(controls["advection_on"])

    fluid = mat["fluid"]
    mu = fluid["viscosity"]
    mu_q = mu
    cp_f = fluid["specific_heat"]
    rho_f0 = fluid["density"]
    rho_f_q = rho_f0
    if mu_nonlinear:
        alpha_f = fluid["therm_expansion"]

    # fixed accumulator buckets; new nonlinear terms just sum onto one, no new name needed
    list_of_matrices = ["K_pp_R", "K_pp_tan", "K_pT_R", "K_pT_tan"]
    if not pressure_only:
        list_of_matrices = ["K_TT_R", "K_TT_tan", "K_Tp_tan"] + list_of_matrices
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
            # Connectivity-only (mesh-static) data, cached by integrate_residuals rather than
            # re-derived here every timestep/Newton iteration.
            cached_cv = fmesh["vec_mat_asmbl"]
            row_vec_ff, col_vec_ff = cached_cv["row_vec_ff"], cached_cv["col_vec_ff"]
            row_vec_fm, col_vec_fm = cached_cv["row_vec_fm"], cached_cv["col_vec_fm"]
            row_vec_mm, col_vec_mm = cached_cv["row_vec_mm"], cached_cv["col_vec_mm"]

            vec_ff = {"Kpp_ff": np.zeros(row_vec_ff.size)}
            vec_fm = {"Kpp_fm": np.zeros(row_vec_fm.size)}
            vec_mm = {"Kpp_mm": np.zeros(row_vec_mm.size)}

            vec_counter_ff = 0
            vec_counter_fm = 0
            vec_counter_mm = 0

    # ----- gravity vector
    g, ng = _get_gravity(mesh["nsd"], controls["gravity"])

    vec = _initiate_vectors(list_of_matrices, nne_list, elem_indx)
    vec_counter = 0  # vectorized storage index counter

    # ----- Matrix integration element loop
    for r in range(num_regions):
        kf, isotropic_perm, nodes_r, w, u, kf_p, _ = _get_region_flow_data(
            mesh, mat, fmesh, nodes_f, is_fracture, r
        )
        # h (thermal coupling) is unused here - this legacy function never assembled the
        # thermal fracture-matrix coupling, only pressure - so thermal_coupling=False below.
        lambda_e, isotropic_cond, h, _, _ = _get_region_thermal_data(mat, is_fracture, r)

        # advection velocity = advection_conductivity * pressure gradient
        if mu_linear and density_linear:
            advec_conduct = cp_f * rho_f_q * kf / mu  # advection conductivity

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
                P_q = N_q @ pe  # quadrature point pressures
                gradT = B_q @ Te  # qudarature point temperature gradient
                gradP = B_q @ pe  # quadrature point pressure gradient
                mu_q, dmu_dT_q, dmu_dp_q = get_viscosity(T_q, P_q, fluid)  # quadrature point viscosities

                advec_conduct = cp_f * rho_f_q * kf / mu_q  # advection conductivity
                kf_b = (
                    ng.T @ np.dot(kf, ng) / mu_q * alpha_f * rho_f_q @ (g.T @ u if is_fracture else g.T)
                )  # kf component in gravity direction

                # Integrate the tangent matrices: Kpp = dFk/dp (Darcy's Law), KbT = dFb/dT (Buoyancy)
                Kpp = _vctrzd_quad_point_integration(B_q, kf / mu_q, B_q, wJe_q, isotropic_perm)
                KbT = _vctrzd_quad_point_integration(N_q, kf_b, B_q, wJe_q, is_fracture)

                integrand_Kpp_dmudp = -dmu_dp_q / mu_q**2 * (kf * gradP if isotropic_perm else kf @ gradP)
                Kpp_dmudp = _vctrzd_quad_point_integration(
                    B_q, integrand_Kpp_dmudp, N_q, wJe_q, isotropic_perm
                )
                integrand_KbT_dmudp = -dmu_dp_q / mu_q * kf_b @ gradT
                KbT_dmudp = _vctrzd_quad_point_integration(N_q, integrand_KbT_dmudp, N_q, wJe_q, is_fracture)

                _accumulate_element_matrices(
                    element_matrices,
                    {
                        "K_pp_R": -Kpp,
                        "K_pp_tan": -(Kpp_dmudp + KbT_dmudp),
                        "K_pT_R": KbT,
                    },
                )

            # ----------------------------------- Advection and stabilization
            # Calculate fluxes, stabilization coefficicent, pressure gradients
            if density_nonlinear:
                T_q = N_q @ Te  # quadrature point temperatures
                P_q = N_q @ pe  # quadrature point pressures
                rho_f_q, drho_f_dT_q, drho_f_dp_q = get_density(T_q, P_q, fluid)  # quadrature point densities
                advec_conduct = cp_f * rho_f_q * kf / mu_q  # advection conductivity

            if not pressure_only:
                # Darcy flux and SUPG (Shakib, Hughes, & Zohan (1991)) stabilization terms
                gradp = B_q @ pe  # nq x nsd x 1

                flux = -advec_conduct * gradp if isotropic_perm else -advec_conduct @ gradp
                flux_T = flux.transpose(0, 2, 1)  # quadrature-point-wise transpose, calculate once
                if nsd == 3:
                    flux_mag = np.sqrt(flux[:, 0, 0] ** 2 + flux[:, 1, 0] ** 2 + flux[:, 2, 0] ** 2)
                elif nsd == 2:
                    flux_mag = np.sqrt(flux[:, 0, 0] ** 2 + flux[:, 1, 0] ** 2)
                else:
                    # Keep flux_mag 1D (nq,) like the nsd 2/3 branches above, not (nq, 1, 1) -
                    # `flux[indx_noflow, :, :]` below silently corrupts unrelated entries when
                    # indx_noflow comes from nonzero() on a >1D array (a 3-tuple of index arrays
                    # gets nested inside the outer index tuple instead of used directly).
                    flux_mag = np.abs(flux[:, 0, 0])
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
                    h_eff = flux_mag / np.sqrt(
                        (flux[:, 0, 0] / L_vec[0, 0]) ** 2 + (flux[:, 1, 0] / L_vec[1, 0]) ** 2
                    )
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

                # advection + SUPG stabilization -> "K_TT_R"; skip the integration outright when off
                if advection_on:
                    KATT = _vctrzd_quad_point_integration(N_q, flux_T, B_q, wJe_q, False)
                    K_stab_TT = _vctrzd_quad_point_integration(B_q, integrand_Kstab, B_q, wJe_q, False)
                    _accumulate_element_matrices(element_matrices, {"K_TT_R": -(KATT + K_stab_TT)})

                # ---------------------------------- Viscosity correction
                if mu_nonlinear:
                    # Integrate tangent matrices: KATT_mu = dFA/dmu * dmu/dT,
                    # K_stab_TT_mu = dF_stab/dmu * dmu/dT
                    FA = flux_T @ gradT
                    integrand_KATT_mu = -1 / mu_q @ FA * dmu_dT_q

                    Fstab = (tau * flux) @ (flux_T @ gradT)

                    # tau derivative wrt mu(T) correction factor
                    correction_factor = 0.5 * (Pe**2 + 18) / (Pe**2 + 9)
                    integrand_Kstab_mu = -2 / mu_q * Fstab * dmu_dT_q * correction_factor.reshape(-1, 1, 1)

                    KATT_mu = _vctrzd_quad_point_integration(N_q, integrand_KATT_mu, N_q, wJe_q, False)
                    K_stab_TT_mu = _vctrzd_quad_point_integration(B_q, integrand_Kstab_mu, N_q, wJe_q, False)

                    _accumulate_element_matrices(element_matrices, {"K_TT_tan": KATT_mu + K_stab_TT_mu})

                # ----------------------------------- Fully coupled off diagonal terms
                if fully_coupled:
                    if advection_on:
                        # Integrate off-diagonal tangents wrt pressure: KATp = dFA/dp, K_stab_Tp = dF_stab/dp
                        gradT = B_q @ Te  # qudarature point temperature gradient
                        Rth_dp = -advec_conduct * gradT if isotropic_perm else -advec_conduct @ gradT
                        integrand_KATp = Rth_dp.transpose(0, 2, 1)  # nq x 1 x nsd

                        # Stabilization wrt p
                        Rth = flux_T @ gradT
                        inner_term = Rth * np.eye(nsd) + flux @ gradT.transpose(0, 2, 1)

                        flux_mag_sq = flux_mag.reshape(-1, 1, 1) ** 2
                        pe_factor = Pe**2 / (Pe**2 + 9)
                        tau_correction_Tp = (
                            pe_factor.reshape(-1, 1, 1) * (Rth / flux_mag_sq) * (flux @ flux_T)
                        )
                        inner_term = inner_term - tau_correction_Tp

                        integrand_Kstab_Tp = (
                            -tau * advec_conduct * inner_term
                            if isotropic_perm
                            else -tau * (inner_term @ advec_conduct)
                        )

                        KATp = _vctrzd_quad_point_integration(N_q, integrand_KATp, B_q, wJe_q, False)
                        K_stab_Tp = _vctrzd_quad_point_integration(B_q, integrand_Kstab_Tp, B_q, wJe_q, False)

                        _accumulate_element_matrices(element_matrices, {"K_Tp_tan": -(KATp + K_stab_Tp)})

                    if mu_nonlinear:
                        # If non-linear viscosity, integrate the viscosity tangents (unrelated to
                        # advection - this stays unconditional on advection_on)

                        # viscosity wrt T: KpT_mu_p = dFk/dmu * dmu/dT (Darcy's law),
                        # KpT_mu_b = dFb/dmu * dmu/dT (buoyancy)
                        del_T = Te - fluid["reference_temperature"]
                        Fk = kf / mu_q * gradp if isotropic_perm else kf / mu_q @ gradp  # (nq, nsd, 1)
                        Fb = kf_b @ B_q @ del_T  # (nq, 1, 1)

                        #   Darcy's law component
                        integrand_KpT_mu_p = -(dmu_dT_q / mu_q) * Fk  # (nq, nsd, 1), elementwise
                        #   Buoyancy component
                        integrand_KpT_mu_b = -1.0 / mu_q * Fb @ dmu_dT_q

                        KpT_mu_p = _vctrzd_quad_point_integration(B_q, integrand_KpT_mu_p, N_q, wJe_q, False)
                        KpT_mu_b = _vctrzd_quad_point_integration(N_q, integrand_KpT_mu_b, N_q, wJe_q, False)

                        _accumulate_element_matrices(element_matrices, {"K_pT_tan": KpT_mu_p + KpT_mu_b})

            # ----- Fracture-matrix pressure coupling
            if is_fracture and mu_nonlinear:
                # Coupling matrices
                coupling_entries = _integrate_fracture_matrix_coupling_tangent(
                    e,
                    enodes,
                    nodes_f,
                    N_q,
                    wJe_q,
                    mesh,
                    kf_p,
                    mu_q,
                    h,
                    pressure_coupling=True,
                    thermal_coupling=False,
                )

                for nne_m, ff_matrices, fm_matrices, mm_matrices in coupling_entries:
                    stop = vec_counter_ff + nne * nne  # vectorized index count end
                    _store_vctrzd_element_matrix(vec_ff, ff_matrices, vec_counter_ff, stop)
                    vec_counter_ff = stop
                    stop = vec_counter_fm + nne_m * nne  # vectorized index count end
                    _store_vctrzd_element_matrix(vec_fm, fm_matrices, vec_counter_fm, stop)
                    vec_counter_fm = stop
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
        if mu_nonlinear:
            # rename to match the "K_pp_R" bucket name for `_stack_fracture_matrices`
            nn_m = mesh["nn"]
            pp_coupling_names = {"Kpp_ff": "K_pp_R_ff", "Kpp_fm": "K_pp_R_fm", "Kpp_mm": "K_pp_R_mm"}
            for name in vec_ff:
                integrated_matrices[pp_coupling_names[name]] = _make_sparse(
                    vec_ff[name], row_vec_ff, col_vec_ff, nn, nn
                )
            for name in vec_fm:
                integrated_matrices[pp_coupling_names[name]] = _make_sparse(
                    vec_fm[name], row_vec_fm, col_vec_fm, nn_m, nn
                )
            for name in vec_mm:
                integrated_matrices[pp_coupling_names[name]] = _make_sparse(
                    vec_mm[name], row_vec_mm, col_vec_mm, nn_m, nn_m
                )
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
    # Map each global node id in conn_face to a local index.
    # Used for assembly.
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
        # Boundary faces may be misordered relative to their global position,
        # relative in negative jacobians.
        # This shouldn't be an issue because all elements are defined with the same
        # relative numbering. Flux directions can then be controlled via
        # the sign of the applied traction.
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
