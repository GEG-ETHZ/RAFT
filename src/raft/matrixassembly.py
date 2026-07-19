"""Assembles global linear FEM matrices."""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.sparse import csc_matrix

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
    integrand: npt.NDArray[np.float64],
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


def _get_parent_coordinates(  # noqa
    etype: str, nodes: npt.NDArray[np.float64], goal_pts: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Determine the parent coordinates of a physical point in an isoparametric element.

    Assumes that the point is on the element boundary.
    """
    parent_coords = np.empty((np.size(nodes, 1), 1, np.size(goal_pts, -1)))

    if etype == "Q4":
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        x_nodes = nodes[:, 0]
        y_nodes = nodes[:, 1]
        tol = 1.0e-7

        for gp in range(np.size(goal_pts, 2)):
            target = goal_pts[gp]
            x_target = target[0][0]
            y_target = target[0][1]
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

                # 2. The point must be collinear AND fall within the segment endpoints [-1, 1]
                if is_collinear and (-1.0 - tol <= t <= 1.0 + tol):
                    t = max(-1.0, min(1.0, t))  # Clean up minor numerical fuzz

                    # Map the 1D parameter 't' back to 2D parent space based on the edge index
                    if idx == 0:  # Edge 1-2: η = -1
                        xi = t
                        eta = -1.0
                        break
                    elif idx == 1:  # Edge 2-3: ξ = 1
                        xi = 1.0
                        eta = t
                        break
                    elif idx == 2:  # Edge 3-4: η = 1 (t runs opposite to ξ direction)
                        xi = -t
                        eta = 1.0
                        break
                    elif idx == 3:  # Edge 4-1: ξ = -1 (t runs opposite to η direction)
                        xi = -1.0
                        eta = -t
                        break

            parent_coords[gp] = np.array([xi, eta])

    elif etype == "T3":
        # Direct mapping
        x1 = nodes[0, 0]
        x2 = nodes[1, 0]
        x3 = nodes[2, 0]
        y1 = nodes[0, 1]
        y2 = nodes[1, 1]
        y3 = nodes[2, 1]

        # Determinant (2x the physical area of the triangle)
        D = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)

        for gp in range(np.size(goal_pts, 2)):
            target = goal_pts[gp]
            x_target = target[0][0]
            y_target = target[0][1]
            xi = ((y3 - y1) * (x_target - x1) - (x3 - x1) * (y_target - y1)) / D
            eta = (-(y2 - y1) * (x_target - x1) + (x2 - x1) * (y_target - y1)) / D
            parent_coords[gp] = np.array([xi, eta])

    return parent_coords


def _create_fracture_coupling_vectors(
    conn_f: list[list[int]], conn_fm: list[list[int]], conn_m: list[list[int]]
) -> tuple[int, npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Create the row and column vectors for vectorized fracture coupling matrix assembly."""
    count_vec = 0
    row_list = []
    col_list = []

    ne_f = len(conn_f)
    for e_f in range(ne_f):
        e_nodes_f = conn_f[e_f]
        nne_f = len(e_nodes_f)
        for f in range(2):
            e_m = conn_fm[e_f][f]
            e_nodes_m = conn_m[e_m]
            nne_m = len(e_nodes_m)

            row_list.append(
                np.repeat(np.array(e_nodes_m, dtype=np.int32).reshape(-1, 1), nne_f, axis=1).ravel()
            )
            col_list.append(np.tile(np.array(e_nodes_f, dtype=np.int32), (1, nne_m)).ravel())
            count_vec += nne_f * nne_m

    row_vec = np.concatenate(row_list).astype(np.int32)
    col_vec = np.concatenate(col_list).astype(np.int32)
    return count_vec, row_vec, col_vec


def integrate_linear_fracture_matrices(  # noqa
    mesh: dict[str, Any],
    mat: dict[str, Any],
    controls: dict[str, Any],
) -> tuple[dict[str, csc_matrix[Any]], dict[str, Any]]:
    """Integrate the linear matrices associated with the lower dimensional fractures."""
    # ----- integration controls
    mu_linear = not bool(controls["viscosity_nonconstant"])
    axisymmetry = bool(controls["axisymmetric_on"])

    # ----- fluid properties, universal
    if mu_linear:
        mu = mat["fluid"]["viscosity"]
        alpha_f = mat["fluid"]["therm_expansion"]
    rho_f = mat["fluid"]["density"]
    lambda_f = mat["fluid"]["therm_cond"]
    comp_f0 = 1.0 / mat["fluid"]["bulk_mod"]
    cp_f = mat["fluid"]["specific_heat"]

    # ----- mesh data
    # Matrix data
    nn_m, nsd = mesh["nn"], mesh["nsd"]
    conn_m = mesh["conn"]  # (ne, nne)  - connectivity matrix
    nodes_m = mesh["nodes"]  # (nn,  nsd) - nodal degrees of freedom
    etype_list_m = mesh["element_type"]
    elem_indx_m = mesh["elem_indx"]

    # Fracture data
    fmesh = mesh["fracture_mesh"]
    nn_f, nne_list_f = fmesh["nn"], fmesh["nne"]
    conn_f = fmesh["conn"]  # (ne, nne)  - connectivity matrix
    nodes_f = fmesh["nodes"]  # (nn,  nsd) - nodal degrees of freedom
    etype_list_f = fmesh["element_type"]
    elem_indx_f = fmesh["elem_indx"]
    nq_list = fmesh["n_quadrature"]
    fracture_lists = fmesh["region_list"]
    num_fractures = len(fracture_lists)
    conn_fm = fmesh["matrix_conn"]

    # ----- gravity vector
    g, ng = _get_gravity(nsd)

    # Reduce nsd for fracture
    nsd -= 1

    # ----- Vectorized matrix assembly
    # Assemble lists of rows, columns, and entries and then assemble the sparse
    # matrix all at once.
    # For the scalar pressure and temperature, there are nne_1 * nne_2 entries
    # per element. For the coupled matrices, because the assembly is ragged,
    # we will  assemble them once and then save them for future use.
    list_of_matrices = ["KM", "KpT", "C", "KTT"]
    if mu_linear:
        list_of_matrices.extend(["KbT", "Kpp"])

    vec = _initiate_vectors(list_of_matrices, nne_list_f, elem_indx_f)
    vec_counter = 0  # vectorized storage index counter

    list_of_coupled_matrices = ["Kpp_fm", "KTT_fm"]
    vec_count, row_vec_coupling, col_vec_coupling = _create_fracture_coupling_vectors(conn_f, conn_fm, conn_m)
    mesh["fracture_mesh"]["vectorized_matrix_assembly_rows"] = row_vec_coupling
    mesh["fracture_mesh"]["vectorized_matrix_assembly_cols"] = col_vec_coupling
    # Initialize the matrices
    vec_coupled = {name: np.zeros(vec_count) for name in list_of_coupled_matrices}
    vec_counter_coupled = 0

    # Matrices to assemble:
    # Kpp = Pressure conductance matrix
    # KbT = Pressure buoyancy matrix - body
    # KM =  Pressure storage matrix
    # KpT =  Pressurization matrix
    # C = Thermal capacity matrix
    # KTT =  Thermal conductance matrix
    # Kpp_fm = Pressure fracture-matrix coupling
    # KTT_fm = Temperature fracture-matrix coupling

    # ----- Matrix integration element loop
    for r in range(num_fractures):
        # fracture properties - by region
        w = mat["fracture"]["aperture"][r]
        nusselt = mat["fracture"]["nusselt"][r]

        # Derived fracture properties
        kf_l = w**3 / 12  # longitudinal permeability
        kf_p = w / 12  # transverse permeability
        h = nusselt * lambda_f / 2 / w  # heat transfer coefficient
        comp_f = comp_f0 * w  # integrated compressibility
        m_e = rho_f * cp_f * w

        u = fmesh["fracture_tangent"][r]
        # Project the mesh nodes into local coordinates
        nodes_local = (nodes_f @ u).reshape(-1, 1)

        if mu_linear:
            kf_b = ng.T @ np.dot(kf_l, ng) / mu * alpha_f * rho_f @ g.T @ u  # local gravity component

        # find element type of first element in region
        elem_type_counter = -1  # force an update on the first element

        for e in fracture_lists[r]:
            # get fracture shape functions
            new_type_idx = bisect.bisect_left(elem_indx_f, e)
            if new_type_idx != elem_type_counter:
                # Update for different element type
                # Evaluate quadrature points and get weights and shape functions.
                W, nq, N_q, GN_q = _get_quad_points_and_shape_functions(
                    nq_list[new_type_idx], nsd, etype_list_f[new_type_idx]
                )
                nne_f = nne_list_f[new_type_idx]
                elem_type_counter = new_type_idx
                Q, _ = getquadrature(nq_list[new_type_idx], nsd, etype_list_f[new_type_idx])

            enodes = conn_f[e]  # local element nodes
            xI = nodes_local[enodes, :]
            B_q, detJe_q = getB(GN_q, xI, nsd)
            wJe_q = W * detJe_q  # Weight * det(Jacobian), used in every integrand
            if axisymmetry:
                # Add radial component to integrand for axisymmetry
                r_coord = N_q @ xI[:, 0]
                wJe_q = wJe_q * r_coord.T

            # Element matrices
            element_matrices = _initialize_element_matrices(list_of_matrices, nne_f, nne_f)

            element_matrices["KM"] = _vectorized_quad_point_integration(N_q, comp_f, N_q, wJe_q, True)
            element_matrices["C"] = _vectorized_quad_point_integration(N_q, m_e, N_q, wJe_q, True)
            element_matrices["KTT"] = _vectorized_quad_point_integration(B_q, lambda_f, B_q, wJe_q, True)
            if mu_linear:
                element_matrices["Kpp"] = _vectorized_quad_point_integration(B_q, kf_l / mu, B_q, wJe_q, True)
                element_matrices["KbT"] = _vectorized_quad_point_integration(N_q, kf_b, B_q, wJe_q, True)

            # ----- Assemble element contributions into global matrices
            stop = vec_counter + nne_f * nne_f  # vectorized index count end
            _store_vectorized_element_matrix(vec, element_matrices, vec_counter, stop)
            vec_counter = stop

            # ---- Loop over faces for matrix coupling
            matrix_elements = conn_fm[e]
            for e_m in matrix_elements:
                enodes_m = conn_m[e_m]
                nne_m = len(enodes_m)
                element_matrices = _initialize_element_matrices(list_of_coupled_matrices, nne_m, nne_f)

                etype = etype_list_m[bisect.bisect_left(elem_indx_m, e_m)]
                getN_m = NMATRIX_FUNCS[etype]
                Q = _get_parent_coordinates(etype, nodes_m[enodes_m, :], N_q @ nodes_f[enodes, :])
                N_qm_list = []
                for q in range(np.size(Q, -1)):
                    xi = Q[q, :].flatten()
                    N_qm_list.append(getN_m(xi))
                N_qm = np.array(N_qm_list)
                element_matrices["Kpp_fm"] = _vectorized_quad_point_integration(N_qm, kf_p, N_q, wJe_q, True)
                element_matrices["KTT_fm"] = _vectorized_quad_point_integration(N_qm, h, N_q, wJe_q, True)
                # ----- Assemble element contributions into global matrices
                stop = vec_counter_coupled + nne_m * nne_f  # vectorized index count end
                _store_vectorized_element_matrix(vec_coupled, element_matrices, vec_counter_coupled, stop)
                vec_counter_coupled = stop

    # Assemble element matrices
    integrated_matrices = _assemble_global_matrices(
        vec, conn_f, nn_f, nne_list_f, elem_indx_f, np.concatenate(fracture_lists)
    )
    # Assemble coupled matrices
    for name in vec_coupled:
        integrated_matrices[name] = _make_sparse(
            vec_coupled[name], row_vec_coupling, col_vec_coupling, nn_m, nn_f
        )

    return (integrated_matrices, mesh)


def integrate_linear_matrices(
    mesh: dict[str, Any], mat: dict[str, Any], controls: dict[str, Any]
) -> tuple[dict[str, csc_matrix[Any]], dict[str, Any], dict[str, Any]]:
    """Integrate linear matrices outside of time integration loop."""
    # ----- integration controls
    mu_linear = not bool(controls["viscosity_nonconstant"])
    axisymmetry = bool(controls["axisymmetric_on"])
    fracture_on = bool(controls["fracture_on"])

    # ----- fluid properties, universal
    if mu_linear:
        mu = mat["fluid"]["viscosity"]
        alpha_f = mat["fluid"]["therm_expansion"]
    rho_f = mat["fluid"]["density"]

    # ----- mesh data
    nn, nne_list, nsd = mesh["nn"], mesh["nne"], mesh["nsd"]
    conn = mesh["conn"]  # (ne, nne)  - connectivity matrix
    nodes = mesh["nodes"]  # (nn,  nsd) - nodal degrees of freedom
    region_lists = mesh["region_lists"]
    num_regions = mesh["num_regions"]
    etype_list = mesh["element_type"]
    nq_list = mesh["n_quadrature"]
    elem_indx = mesh["elem_indx"]

    # ----- gravity vector
    g, ng = _get_gravity(nsd)

    # ----- Vectorized matrix assembly
    # Assemble lists of rows, columns, and entries and then assemble the sparse
    # matrix all at once.
    # For the scalar pressure and temperature, there are nne * nne entries
    # per element.
    list_of_matrices = ["Kpp", "KM", "KpT", "KbT", "C", "KTT"] if mu_linear else ["KM", "KpT", "C", "KTT"]
    vec = _initiate_vectors(list_of_matrices, nne_list, elem_indx)
    vec_counter = 0  # vectorized storage index counter

    # Matrices to assemble:
    # Kpp = Pressure conductance matrix
    # KbT = Pressure buoyancy matrix - body
    # KM =  Pressure storage matrix
    # KpT =  Pressurization matrix
    # C = Thermal capacity matrix
    # KTT =  Thermal conductance matrix

    # ----- Matrix integration element loop
    for r in range(num_regions):
        # solid properties - by region
        alpha_e = np.asarray(mat["solid"]["therm_expansion_eff"][r])
        comp_f = np.asarray(1 / mat["solid"]["storage_mod"][r])
        kf = mat["solid"]["permeability_eff"][r]
        m_e = mat["solid"]["volumetric_heat_capacity_eff"][r]
        lambda_e = mat["solid"]["therm_cond_eff"][r]

        if mu_linear:
            kf_b = ng.T @ np.dot(kf, ng) / mu * alpha_f * rho_f @ g.T  # kf component in gravity direction

        # Check for region isotropy
        isotropic_cond = bool(isinstance(lambda_e, (float, int)))
        isotropic_perm = bool(isinstance(kf, (float, int)))

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
            xI = nodes[enodes, :]
            B_q, detJe_q = getB(GN_q, xI, nsd)
            wJe_q = W * detJe_q  # Weight * det(Jacobian), used in every integrand
            if axisymmetry:
                # Add radial component to integrand for axisymmetry
                r_coord = N_q @ xI[:, 0]
                wJe_q = wJe_q * r_coord.T

            # Element matrices
            element_matrices = _initialize_element_matrices(list_of_matrices, nne, nne)

            # Vectorized integrand computation
            # N_q: [nq, 1, nne]
            # B_q: [nq, nsd, nne]
            # wJe: [nq,]

            # Vectorized integration performs the following loop:
            # for q in range(nq):
            #     B = B_q[q]
            #     N = N_q[q]
            #     wJe = wJe_q[q]

            #     element_matrices["KM"] += N.T @ (comp_f * N) * wJe
            #     element_matrices["Kpp"] += B.T @ np.dot(kf / mu, B) * wJe
            #     element_matrices["KpT"] += N.T @ (alpha_e * N) * wJe
            #     element_matrices["KbT"] += N.T @ (kf_b @ g.T) @ B * wJe
            #     element_matrices["C"] += N.T @ (m_e * N) * wJe
            #     element_matrices["KTT"] += B.T @ np.dot(lambda_e, B) * wJe

            element_matrices["KM"] = _vectorized_quad_point_integration(N_q, comp_f, N_q, wJe_q, True)
            element_matrices["KpT"] = _vectorized_quad_point_integration(N_q, alpha_e, N_q, wJe_q, True)
            element_matrices["C"] = _vectorized_quad_point_integration(N_q, m_e, N_q, wJe_q, True)
            element_matrices["KTT"] = _vectorized_quad_point_integration(
                B_q, lambda_e, B_q, wJe_q, isotropic_cond
            )
            if mu_linear:
                element_matrices["Kpp"] = _vectorized_quad_point_integration(
                    B_q, kf / mu, B_q, wJe_q, isotropic_perm
                )
                element_matrices["KbT"] = _vectorized_quad_point_integration(N_q, kf_b, B_q, wJe_q, False)

            # ----- Assemble element contributions into global matrices
            stop = vec_counter + nne * nne  # vectorized index count end
            _store_vectorized_element_matrix(vec, element_matrices, vec_counter, stop)
            vec_counter = stop

    integrated_matrices = _assemble_global_matrices(
        vec, conn, nn, nne_list, elem_indx, np.concatenate(mesh["region_lists"])
    )
    if fracture_on:
        integrated_fracture_matrices = integrate_linear_fracture_matrices(mesh, mat, controls)  # noqa
        # Assemble the connected fracture-matrix matrices
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
    else:
        flux_mag = np.sqrt(flux[:, 0, 0] ** 2 + flux[:, 1, 0] ** 2)  # flux magnitude
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
    else:
        h_eff = flux_mag / np.sqrt((flux[:, 0, 0] / L_vec[0, 0]) ** 2 + (flux[:, 1, 0] / L_vec[1, 0]) ** 2)
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
) -> dict[str, npt.NDArray[np.float64]]:
    """Integrate the viscosity-dependent pressure conductance (Kpp) and buoyancy (KbT) matrices."""
    return {
        "Kpp": _vectorized_quad_point_integration(B_q, kf / mu_q, B_q, wJe_q, isotropic_perm),
        "KbT": _vectorized_quad_point_integration(N_q, kf_b, B_q, wJe_q, False),
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
) -> dict[str, Any]:
    """Integrate non-linear matrices inside time integration loop.

    KATT = Temperature advection tangent matrix wrt temperature
    K_stab_TT = Advection stabilization wrt temperature
    KATp = Temperature advection tangent matrix wrt pressure
    K_stab_Tp = Advection stabilization wrt pressure
    Kpp, KbT, KpT_mu_p, KpT_mu_b, KATT_mu, K_stab_TT_mu = viscosity-nonlinear corrections, see helpers below.
    """
    p = primary_variables["p"][:, 0]
    T = primary_variables["T"][:, 0]
    # ----- integration controls
    mu_nonlinear = controls["viscosity_nonconstant"]
    mu_linear = not mu_nonlinear
    fully_coupled = controls["fully_coupled"]
    axisymmetry = bool(controls["axisymmetric_on"])

    # ----- fluid properties
    mu = mat["fluid"]["viscosity"]
    cp_f = mat["fluid"]["specific_heat"]
    rho_f = mat["fluid"]["density"]

    # ----- mesh data
    nn, nne_list, nsd = mesh["nn"], mesh["nne"], mesh["nsd"]
    conn = mesh["conn"]  # connectivity
    nodes = mesh["nodes"]  # (nn,  nsd) - nodal degrees of freedom
    region_lists = mesh["region_lists"]
    num_regions = mesh["num_regions"]
    etype_list = mesh["element_type"]
    nq_list = mesh["n_quadrature"]
    elem_indx = mesh["elem_indx"]

    # ----- Pre-allocate sparse matrices (use lil for efficient assembly)
    list_of_matrices = ["KATT", "K_stab_TT"]
    if fully_coupled:
        list_of_matrices.extend(["KATp", "K_stab_Tp"])

    if mu_nonlinear:
        list_of_matrices.extend(["Kpp", "KbT", "KpT_mu_b", "KpT_mu_p", "KATT_mu", "K_stab_TT_mu"])
        alpha_f = mat["fluid"]["therm_expansion"]

    # ----- gravity vector
    g, ng = _get_gravity(nsd)

    vec = _initiate_vectors(list_of_matrices, nne_list, elem_indx)
    vec_counter = 0  # vectorized storage index counter

    # ----- Matrix integration element loop
    for r in range(num_regions):
        # solid properties - by region
        kf = mat["solid"]["permeability_eff"][r]
        lambda_e = mat["solid"]["therm_cond_eff"][r]

        # Check for region isotropy
        isotropic_cond = bool(isinstance(lambda_e, (float, int)))
        isotropic_perm = bool(isinstance(kf, (float, int)))

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
            xI = nodes[enodes, :]
            B_q, detJe_q = getB(GN_q, xI, nsd)
            wJe_q = W * detJe_q  # Weight * det(Jacobian), used in every integrand
            if axisymmetry:
                # Add radial component to integrand for axisymmetry
                r_coord = N_q @ xI[:, 0]
                wJe_q = wJe_q * r_coord.T

            # Element matrices initialize
            element_matrices = _initialize_element_matrices(list_of_matrices, nne, nne)

            Te = T[enodes].reshape(-1, 1)  # element nodal temperatures
            pe = p[enodes].reshape(-1, 1)  # element nodal pressures
            enodes_coords = nodes[enodes, :]
            L_vec = (enodes_coords.max(axis=0) - enodes_coords.min(axis=0)).reshape(-1, 1)  # element lengths

            # ------------------------------------ Pressure and fluid flow
            if mu_nonlinear:
                # If mu is nonlinear, the pressure diffusion is also non-linear and must
                # be re-integrated at each timestep.
                T_q = N_q @ Te  # quadrature point temperatures
                mu_q, dmu_dT_q = get_viscosity(T_q, mu)  # quadrature point viscosities

                advec_conduct = cp_f * rho_f * kf / mu_q  # advection conductivity
                kf_b = (
                    ng.T @ np.dot(kf, ng) / mu_q * alpha_f * rho_f @ g.T
                )  # kf component in gravity direction

                # Integrate the tangent matrices
                # Kpp = dFk / dp - Darcy's Law
                # KbT = dFb / dT - Buoyancy
                element_matrices.update(
                    _integrate_pressure_flow_terms(N_q, B_q, wJe_q, kf, mu_q, kf_b, isotropic_perm)
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
                    T_0e = mat["T_0"][enodes]
                    del_T = Te - T_0e.reshape(-1, 1)
                    element_matrices.update(
                        _integrate_viscosity_pressure_correction(
                            N_q, B_q, wJe_q, supg, kf, mu_q, dmu_dT_q, kf_b, del_T, isotropic_perm
                        )
                    )

            # ----- Assemble element contributions into global matrices
            stop = vec_counter + nne * nne  # vectorized index count end
            _store_vectorized_element_matrix(vec, element_matrices, vec_counter, stop)
            vec_counter = stop

    # ----- Return and convert to sparse matrix
    return _assemble_global_matrices(vec, conn, nn, nne_list, elem_indx, np.concatenate(mesh["region_lists"]))


def integrate_NLpressure_matrices(
    mesh: dict[str, Any],
    mat: dict[str, Any],
    primary_variables: dict[str, Any],
    controls: dict[str, Any],  # noqa
) -> dict[str, Any]:
    """Integrate non-linear fluid flow matrices inside time integration loop."""
    T = primary_variables["T"][:, 0]  # noqa: F841

    # ----- fluid properties
    mu_coeff = mat["fluid"]["viscosity"]
    alpha_f = mat["fluid"]["therm_expansion"]
    rho_f = mat["fluid"]["density"]

    # ----- mesh data
    nn, nne_list, nsd = mesh["nn"], mesh["nne"], mesh["nsd"]
    conn = mesh["conn"]  # connectivity
    nodes = mesh["nodes"]  # (nn,  nsd) - nodal degrees of freedom
    region_lists = mesh["region_lists"]
    num_regions = mesh["num_regions"]
    etype_list = mesh["element_type"]
    nq_list = mesh["n_quadrature"]
    elem_indx = mesh["elem_indx"]
    axisymmetry = bool(controls["axisymmetric_on"])

    # ----- gravity vector
    g, ng = _get_gravity(nsd)

    # ----- Pre-allocate sparse matrices (use lil for efficient assembly)
    list_of_matrices = ["Kpp", "KbT"]

    vec = _initiate_vectors(list_of_matrices, nne_list, elem_indx)
    vec_counter = 0  # vectorized storage index counter

    # Kpp = Pressure conductance matrix
    # KbT = Pressure buoyancy matrix - body

    # ----- Matrix integration element loop
    for r in range(num_regions):
        # solid properties - by region
        kf = mat["solid"]["permeability_eff"][r]

        isotropic_perm = bool(isinstance(kf, (float, int)))

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
            xI = nodes[enodes, :]
            B_q, detJe_q = getB(GN_q, xI, nsd)
            wJe_q = W * detJe_q  # Weight * det(Jacobian), used in every integrand
            if axisymmetry:
                # Add radial component to integrand for axisymmetry
                r_coord = N_q @ xI[:, 0]
                wJe_q = wJe_q * r_coord.T

            # Element matrices
            element_matrices = _initialize_element_matrices(list_of_matrices, nne, nne)

            # vectorized integration
            Te = T[enodes].reshape(-1, 1)  # element nodal temperatures
            T_q = N_q @ Te  # quadrature point temperatures
            mu_q, _ = get_viscosity(T_q, mu_coeff)  # quadrature point viscosities

            kf_b = ng.T @ np.dot(kf, ng) / mu_q * alpha_f * rho_f @ g.T  # kf component in gravity direction

            element_matrices["Kpp"] = _vectorized_quad_point_integration(
                B_q, kf / mu_q, B_q, wJe_q, isotropic_perm
            )
            element_matrices["KbT"] = _vectorized_quad_point_integration(N_q, kf_b, B_q, wJe_q, False)

            # ----- Assemble element contributions into global matrices
            stop = vec_counter + nne * nne  # vectorized index count end
            _store_vectorized_element_matrix(vec, element_matrices, vec_counter, stop)
            vec_counter = stop

    # ----- Return and convert to sparse matrix
    return _assemble_global_matrices(vec, conn, nn, nne_list, elem_indx, np.concatenate(mesh["region_lists"]))


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
