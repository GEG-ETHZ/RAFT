# Copyright 2026 The RAFT Community
#
# This file is part of RAFT.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Assembles global linear FEM matrices."""

from __future__ import annotations

import bisect
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from numba import njit  # type: ignore[import-untyped]
from scipy.sparse import csc_matrix, hstack, vstack

from .matprops import (
    EnthalpyResult,
    PropertyResult,
    get_effective_permeability,
    get_EOS,
    get_friction_factor,
    get_nusselt_number,
    get_porosity_lagrangian,
    get_viscosity,
)
from .physics import Physics
from .shapefunctions import (
    GNMATRIX_FUNCS,
    NMATRIX_FUNCS,
    getB,
    getquadrature,
)


# ---- Data classes
@dataclass(frozen=True, slots=True)
class ElementKinematics:
    """This element's nodal primary variables, sliced out and interpolated to quadrature points.

    `Te`/`Te_flow`/`pe`/`T_q`/`T_flow_q`/`p_q` carry the full time history (one column per
    stored iterate/timestep, current iterate first); the rest are already current-iterate-only
    (`T_0e`/`p_0e` are reference-state constants, `L_vec` is geometric, and `gradT`/`gradP` are
    built from column 0 alone). Use `.n` to slice down to a current-iterate-only view.
    """

    Te: npt.NDArray[np.float64]  # Element nodal temperature (nne x 4)
    Te_flow: npt.NDArray[np.float64]  # Element temperatures for flow evaluation (nne x 4)
    pe: npt.NDArray[np.float64]  # Element nodal pressure (nne x 4)
    T_0e: npt.NDArray[np.float64]  # Element nodal reference state temperature (nne x 1)
    p_0e: npt.NDArray[np.float64]  # Element nodal reference state pressure (nne x 1)
    L_vec: npt.NDArray[np.float64]  # Geometric element lengths in each direction (nsd)
    T_q: npt.NDArray[np.float64]  # Quad point temperatures (nq x 4)
    T_flow_q: npt.NDArray[np.float64]  # Quad point flow evaluation temperature (nq x 4)
    p_q: npt.NDArray[np.float64]  # Quad point pressures (nq x 4)
    gradT: npt.NDArray[np.float64]  # Quad point temperature gradient (nq x nsd x 1) # noqa: N815
    gradP: npt.NDArray[np.float64]  # Quad point pressure gradient (nq x nsd x 1) # noqa: N815

    @property
    def n(self) -> ElementKinematics:
        """Current-iterate (column 0) slice of every history-carrying field."""
        return ElementKinematics(
            self.Te[:, 0:1],
            self.Te_flow[:, 0:1],
            self.pe[:, 0:1],
            self.T_0e,
            self.p_0e,
            self.L_vec,
            self.T_q[:, :, 0:1],
            self.T_flow_q[:, :, 0:1],
            self.p_q[:, :, 0:1],
            self.gradT,
            self.gradP,
        )


@dataclass(frozen=True, slots=True)
class PropertyState:
    """A material property at quadrature points, full time history, with its current-iterate slice."""

    q: PropertyResult

    @property
    def q_n(self) -> PropertyResult:
        """Current-iterate (column 0) slice of the quadrature-point value."""
        return PropertyResult(self.q.value[:, :, 0:1], self.q.dp[:, :, 0:1], self.q.dT[:, :, 0:1])


@dataclass(frozen=True, slots=True)
class NodalPropertyState(PropertyState):
    """An extension of PropertyState with element nodal values as well as quad point values.

    A.e : Values of A at element nodes.
    A.q : Vallues of A at quad points.
    A.e_n: Values of A at element nodes for current timestep n.
    A.q_n: Values of A at quad points for current timestep n.
    """

    e: PropertyResult

    @property
    def e_n(self) -> PropertyResult:
        """Current-iterate (column 0) slice of the nodal value."""
        return PropertyResult(self.e.value[:, 0:1], self.e.dp[:, 0:1], self.e.dT[:, 0:1])


@dataclass(frozen=True, slots=True)
class NodalEnthalpyState:
    """`NodalPropertyState`'s counterpart for `EnthalpyResult`, which also carries `cp`.

    Same quad point (`.q`)/element node (`.e`) shape as `NodalPropertyState`, evaluated once at
    nodes in `_evaluate_nodal_fluid_state` and interpolated to quad points from there,
    rather than re-derived per consumer (`_evaluate_advection_terms`'s `.e_n`/`.q_n`).
    """

    q: EnthalpyResult
    e: EnthalpyResult

    @property
    def q_n(self) -> EnthalpyResult:
        """Current-iterate (column 0) slice of the quadrature-point value."""
        q = self.q
        return EnthalpyResult(q.value[:, :, 0:1], q.dp[:, :, 0:1], q.dT[:, :, 0:1], q.cp[:, :, 0:1])

    @property
    def e_n(self) -> EnthalpyResult:
        """Current-iterate (column 0) slice of the nodal value."""
        e = self.e
        return EnthalpyResult(e.value[:, 0:1], e.dp[:, 0:1], e.dT[:, 0:1], e.cp[:, 0:1])


@dataclass(frozen=True, slots=True)
class RegionMeshData:
    """Mesh/region data resolved from the bulk mesh, or the fracture/well mesh on their recursive passes."""

    nn: int
    nsd: int
    conn: list[list[int]]
    nne_list: list[int]
    etype_list: list[str]
    elem_indx: list[int]
    nq_list: list[int]
    region_lists: list[list[int]]
    num_regions: int
    submesh: dict[str, Any]  # fracture or well mesh; {} sentinel for the bulk mesh
    submesh_nodes: npt.NDArray[np.float64]  # sub-mesh nodes; empty sentinel for the bulk mesh


@dataclass(frozen=True, slots=True)
class RegionSolidData:
    """Per-region solid properties."""

    K_d: float
    phi_0: float
    biot: float
    rho_s: float
    cp_s: float
    alpha_s: float
    lambda_s: npt.NDArray[np.float64] | float  # anisotropic case is a conductivity tensor


@dataclass(frozen=True, slots=True)
class RegionFlowData:
    """Per-region flow properties; NaN sentinels where a field does not apply."""

    kf: npt.NDArray[np.float64] | float  # permeability; dummy 1 for wells
    isotropic_perm: bool
    nodes_r: npt.NDArray[np.float64]  # fracture: projected nodes; well: 3D nodes, projected per element
    w0: float  # fracture aperture or well area; scales storage and conduction
    u: npt.NDArray[np.float64]  # fracture: region tangent (nsd, 1); well: element tangents (ne, nsd)
    kf_perp: PropertyResult  # fracture transverse permeability, coupling only
    diameter: float  # well only
    radius: float  # well only
    perforation_conductance: float  # well only; well-to-rock conductance per unit wall area (m)
    roughness: float  # well only


# ---- General helpers


def _get_gravity(nsd: int, magnitude: float) -> npt.NDArray[np.float64]:
    """Create gravity vector. Acts in z in 3D and y in 2D."""
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
    """Make sparse matrix from vectorized input."""
    return csc_matrix((vals, (row_vec, col_vec)), shape=(nn_1, nn_2))


def _pad_sparse(matrix: csc_matrix, shape: tuple[int, int]) -> csc_matrix:
    """Zero-pad a sparse matrix to `shape`, keeping entries in the top-left."""
    coo = matrix.tocoo()
    return csc_matrix((coo.data, (coo.row, coo.col)), shape=shape)


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
    """Sum each term onto `element_matrices` in place instead of overwriting."""
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

    # choose between multiply types for scalar vs. non-scalar material properties
    middle = integrand * right_matrix if scalar else integrand @ right_matrix

    # Stack quad points into the contraction: sum_q L_q^T M_q w_q as a single matmul
    nq, a, n_left = left_matrix.shape
    weighted = (middle * weights.reshape(-1, 1, 1)).reshape(nq * a, -1)
    result: npt.NDArray[np.float64] = left_matrix.reshape(nq * a, n_left).T @ weighted
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

    # Stack quad points into the contraction: sum_q L_q^T f_q w_q as a single matmul
    nq, a, n_left = left_matrix.shape
    weighted = (integrand * weights.reshape(-1, 1, 1)).reshape(nq * a, -1)
    result: npt.NDArray[np.float64] = left_matrix.reshape(nq * a, n_left).T @ weighted
    return result


def _lumped_mass(N_q: npt.NDArray[np.float64], wJe_q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Row-sum lumped mass, sum_j int(N_i N_j) = int(N_i) since the shape functions sum to 1."""
    result: npt.NDArray[np.float64] = wJe_q.reshape(-1) @ N_q[:, 0, :]
    return result


def _diag_matrix(d: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Square matrix with diagonal `d`; `np.diag`'s generality costs more than the fill here."""
    n = d.size
    M = np.zeros((n, n))
    M.flat[:: n + 1] = d
    return M


@njit(cache=True)  # type: ignore[untyped-decorator]
def _get_parent_coordinates_q4(
    nodes: npt.NDArray[np.float64], goal_pts: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Numba kernel: map physical boundary points to Q4 parent coordinates by edge projection."""
    n_loop = goal_pts.shape[0]
    parent_coords = np.empty((n_loop, 1, 2))
    x_nodes = nodes[:, 0]
    y_nodes = nodes[:, 1]
    tol = 1.0e-7

    # Q4 boundary edges as (node_i, node_j, fixed_axis, fixed_value, t_sign): the local
    # parameter `t` from the edge projection maps to whichever of xi/eta isn't pinned to +-1.
    edge_i = (0, 1, 2, 3)
    edge_j = (1, 2, 3, 0)
    fixed_axis = (1, 0, 1, 0)
    fixed_val = (-1.0, 1.0, 1.0, -1.0)
    t_sign = (1.0, 1.0, -1.0, -1.0)

    for gp in range(n_loop):
        x_target = goal_pts[gp, 0, 0]
        y_target = goal_pts[gp, 0, 1]
        xi = 0.0
        eta = 0.0
        for idx in range(4):
            i = edge_i[idx]
            j = edge_j[idx]
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
                free_val = t_sign[idx] * t
                if fixed_axis[idx] == 0:
                    xi, eta = fixed_val[idx], free_val
                else:
                    xi, eta = free_val, fixed_val[idx]
                break

        parent_coords[gp, 0, 0] = xi
        parent_coords[gp, 0, 1] = eta

    return parent_coords


@njit(cache=True)  # type: ignore[untyped-decorator]
def _get_parent_coordinates_t3(
    nodes: npt.NDArray[np.float64], goal_pts: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Numba kernel: map physical points to T3 parent coordinates via the direct linear solve."""
    n_loop = goal_pts.shape[0]
    parent_coords = np.empty((n_loop, 1, 2))

    x1, x2, x3 = nodes[0, 0], nodes[1, 0], nodes[2, 0]
    y1, y2, y3 = nodes[0, 1], nodes[1, 1], nodes[2, 1]

    # Determinant (2x the physical area of the triangle)
    D = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)

    for gp in range(n_loop):
        x_target = goal_pts[gp, 0, 0]
        y_target = goal_pts[gp, 0, 1]
        parent_coords[gp, 0, 0] = ((y3 - y1) * (x_target - x1) - (x3 - x1) * (y_target - y1)) / D
        parent_coords[gp, 0, 1] = (-(y2 - y1) * (x_target - x1) + (x2 - x1) * (y_target - y1)) / D

    return parent_coords


@njit(cache=True)  # type: ignore[untyped-decorator]
def _get_parent_coordinates_t4(
    nodes: npt.NDArray[np.float64], goal_pts: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Numba kernel: map physical points to T4 parent coordinates via the direct linear solve."""
    n_loop = goal_pts.shape[0]
    parent_coords = np.empty((n_loop, 1, 3))

    # Affine map x = x1 + J xi, with J columns the edges from node 0
    J = np.empty((3, 3))
    for k in range(3):
        J[:, k] = nodes[k + 1, :] - nodes[0, :]
    J_inv = np.linalg.inv(J)

    for gp in range(n_loop):
        parent_coords[gp, 0, :] = J_inv @ (goal_pts[gp, 0, :] - nodes[0, :])

    return parent_coords


@njit(cache=True)  # type: ignore[untyped-decorator]
def _get_parent_coordinates_b8(
    nodes: npt.NDArray[np.float64], goal_pts: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Numba kernel: map physical boundary points to B8 parent coordinates by bilinear face inversion."""
    n_loop = goal_pts.shape[0]
    parent_coords = np.zeros((n_loop, 1, 3))
    tol = 1.0e-7 * (np.max(nodes) - np.min(nodes))  # relative to element size

    # B8 faces as (fixed_axis, fixed_value, s_axis, t_axis, nodes at (s, t) = (-,-), (+,-), (+,+), (-,+))
    fixed_axis = (2, 2, 1, 1, 0, 0)
    fixed_val = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    s_axis = (0, 0, 0, 0, 1, 1)
    t_axis = (1, 1, 2, 2, 2, 2)
    face_nodes = np.array(
        [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [3, 2, 6, 7], [0, 3, 7, 4], [1, 2, 6, 5]]
    )

    for gp in range(n_loop):
        x_target = goal_pts[gp, 0, :]
        for f in range(6):
            P0 = nodes[face_nodes[f, 0], :]
            P1 = nodes[face_nodes[f, 1], :]
            P2 = nodes[face_nodes[f, 2], :]
            P3 = nodes[face_nodes[f, 3], :]
            # Bilinear face map x = a + b s + c t + d s t
            a = 0.25 * (P0 + P1 + P2 + P3)
            b = 0.25 * (-P0 + P1 + P2 - P3)
            c = 0.25 * (-P0 - P1 + P2 + P3)
            d = 0.25 * (P0 - P1 + P2 - P3)
            r = x_target - a

            # Project to 2D by dropping the axis closest to the face normal
            normal = np.cross(b, c)
            drop = np.argmax(np.abs(normal))
            i, j = (1, 2) if drop == 0 else ((0, 2) if drop == 1 else (0, 1))

            # Eliminating t gives A s^2 + B s + C = 0 (2D cross products)
            A = b[i] * d[j] - b[j] * d[i]
            B = (b[i] * c[j] - b[j] * c[i]) - (r[i] * d[j] - r[j] * d[i])
            C = -(r[i] * c[j] - r[j] * c[i])
            if abs(A) < 1.0e-12 * abs(B):
                s_pt = -C / B  # parallelogram face: linear in s
            else:
                disc = max(B * B - 4.0 * A * C, 0.0)
                s_pt = (-B + np.sqrt(disc)) / (2.0 * A)
                if abs(s_pt) > 1.0 + 1.0e-7:  # take the root inside the face
                    s_pt = (-B - np.sqrt(disc)) / (2.0 * A)
            if abs(s_pt) > 1.0 + 1.0e-7:
                continue

            # Least-squares t along the s-isoline
            ct = c + d * s_pt
            t_pt = np.dot(r - b * s_pt, ct) / np.dot(ct, ct)
            if abs(t_pt) > 1.0 + 1.0e-7:
                continue

            # Accept the face only if the point lies on it in 3D
            if np.linalg.norm(a + b * s_pt + c * t_pt + d * s_pt * t_pt - x_target) < tol:
                parent_coords[gp, 0, fixed_axis[f]] = fixed_val[f]
                parent_coords[gp, 0, s_axis[f]] = max(-1.0, min(1.0, s_pt))
                parent_coords[gp, 0, t_axis[f]] = max(-1.0, min(1.0, t_pt))
                break

    return parent_coords


def _get_parent_coordinates(
    etype: str, nodes: npt.NDArray[np.float64], goal_pts: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Determine the parent coordinates of physical points in an isoparametric element.

    Q4 assumes the points lie on the element boundary. Used for mapping fracture/well
    quadrature points to matrix element points. `goal_pts` and the result are (nq, 1, nsd).
    """
    parent_coords: npt.NDArray[np.float64]
    match etype:
        case "Q4":
            parent_coords = _get_parent_coordinates_q4(nodes, goal_pts)
        case "T3":
            parent_coords = _get_parent_coordinates_t3(nodes, goal_pts)
        case "T4":
            parent_coords = _get_parent_coordinates_t4(nodes, goal_pts)
        case "B8":
            parent_coords = _get_parent_coordinates_b8(nodes, goal_pts)
        case _:
            raise NotImplementedError(f"Parent coordinate mapping for {etype} elements.")
    return parent_coords


def _create_fracture_coupling_vectors(
    conn_f: list[list[int]], conn_fm: list[list[int]], conn_m: list[list[int]]
) -> dict[str, npt.NDArray[np.int32]]:
    """Create the row and column vectors for vectorized fracture coupling matrix assembly."""
    row_list_ff = []  # Fracture-fracture on-diagonal
    col_list_ff = []

    row_list_fm = []  # Fracture-matrix off-diagonal (row=matrix, col=fracture)
    col_list_fm = []

    row_list_mf = []  # Matrix-fracture off-diagonal (row=fracture, col=matrix)
    col_list_mf = []

    row_list_mm = []  # Matrix-matrix on-diagonal
    col_list_mm = []

    # Matrix-side scatter target for the coupling *residual* (a vector, so
    # one row index per matrix node per match - no column/nne_m^2 term)
    row_list_flux_m = []

    ne_f = len(conn_f)
    for e_f in range(ne_f):
        e_nodes_f = conn_f[e_f]
        nne_f = len(e_nodes_f)
        for e_m in conn_fm[e_f]:
            e_nodes_m = conn_m[e_m]
            nne_m = len(e_nodes_m)

            # ff: nne_f * nne_f entries
            row_list_ff.append(
                np.repeat(np.array(e_nodes_f, dtype=np.int32).reshape(-1, 1), nne_f, axis=1).ravel()
            )
            col_list_ff.append(np.tile(np.array(e_nodes_f, dtype=np.int32), (1, nne_f)).ravel())
            # fm: nne_m * nne_f entries
            row_list_fm.append(
                np.repeat(np.array(e_nodes_m, dtype=np.int32).reshape(-1, 1), nne_f, axis=1).ravel()
            )
            col_list_fm.append(np.tile(np.array(e_nodes_f, dtype=np.int32), (1, nne_m)).ravel())
            # mf: nne_f * nne_m entries
            row_list_mf.append(
                np.repeat(np.array(e_nodes_f, dtype=np.int32).reshape(-1, 1), nne_m, axis=1).ravel()
            )
            col_list_mf.append(np.tile(np.array(e_nodes_m, dtype=np.int32), (1, nne_f)).ravel())
            # mm: nne_m * nne_m entries
            row_list_mm.append(
                np.repeat(np.array(e_nodes_m, dtype=np.int32).reshape(-1, 1), nne_m, axis=1).ravel()
            )
            col_list_mm.append(np.tile(np.array(e_nodes_m, dtype=np.int32), (1, nne_m)).ravel())
            # flux (residual, matrix-side): nne_m entries
            row_list_flux_m.append(np.array(e_nodes_m, dtype=np.int32))

    return {
        "row_vec_ff": np.concatenate(row_list_ff).astype(np.int32),
        "col_vec_ff": np.concatenate(col_list_ff).astype(np.int32),
        "row_vec_fm": np.concatenate(row_list_fm).astype(np.int32),
        "col_vec_fm": np.concatenate(col_list_fm).astype(np.int32),
        "row_vec_mf": np.concatenate(row_list_mf).astype(np.int32),
        "col_vec_mf": np.concatenate(col_list_mf).astype(np.int32),
        "row_vec_mm": np.concatenate(row_list_mm).astype(np.int32),
        "col_vec_mm": np.concatenate(col_list_mm).astype(np.int32),
        "row_vec_flux_m": np.concatenate(row_list_flux_m).astype(np.int32),
    }


def _get_fracture_coupling_vectors(
    mesh: dict[str, Any], mesh_data: RegionMeshData
) -> dict[str, npt.NDArray[np.int32]]:
    """Fracture-matrix coupling index vectors, built once and cached on the fracture mesh."""
    fmesh = mesh_data.submesh
    if "vec_mat_asmbl" not in fmesh:
        fmesh["vec_mat_asmbl"] = _create_fracture_coupling_vectors(
            mesh_data.conn, fmesh["matrix_conn"], mesh["conn"]
        )
    vectors: dict[str, npt.NDArray[np.int32]] = fmesh["vec_mat_asmbl"]
    return vectors


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
        N_qm = np.array([getN_m(Q[q, 0, :]) for q in range(Q.shape[0])])

        yield e_m, enodes_m, nne_m, N_qm


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


def _create_well_coupling_vectors(
    conn_w: list[list[int]],
    matrix_conn: list[list[int]],
    matrix_node_conn: list[list[int]],
    elem_order: npt.NDArray[np.int64],
) -> dict[str, npt.NDArray[np.int32]]:
    """Create the row and column vectors for vectorized well coupling matrix assembly.

    Wells couple directly to the coincident matrix nodes, so every block is nne x nne.
    """
    # Coupled elements in integration order; elements outside the domain have no matrix nodes
    coupled = [e for e in elem_order if matrix_conn[e] != [-1]]
    e_nodes_w = np.asarray([conn_w[e] for e in coupled], dtype=np.int32).reshape(-1, 2)
    e_nodes_m = np.asarray([matrix_node_conn[e] for e in coupled], dtype=np.int32).reshape(-1, 2)
    nne = e_nodes_w.shape[1]

    return {
        # ww: Well-well on-diagonal; nne * nne entries
        "row_vec_ww": np.repeat(e_nodes_w, nne, axis=1).ravel(),
        "col_vec_ww": np.tile(e_nodes_w, (1, nne)).ravel(),
        # wm: Well-matrix off-diagonal (row=matrix, col=well); nne * nne entries
        "row_vec_wm": np.repeat(e_nodes_m, nne, axis=1).ravel(),
        "col_vec_wm": np.tile(e_nodes_w, (1, nne)).ravel(),
        # mw: Matrix-well off-diagonal (row=well, col=matrix); nne * nne entries
        "row_vec_mw": np.repeat(e_nodes_w, nne, axis=1).ravel(),
        "col_vec_mw": np.tile(e_nodes_m, (1, nne)).ravel(),
        # mm: Matrix-matrix on-diagonal; nne * nne entries
        "row_vec_mm": np.repeat(e_nodes_m, nne, axis=1).ravel(),
        "col_vec_mm": np.tile(e_nodes_m, (1, nne)).ravel(),
        # flux (residual, matrix-side): matrix-side scatter target, nne entries
        "row_vec_flux_m": e_nodes_m.ravel(),
    }


def _get_well_coupling_vectors(mesh_data: RegionMeshData) -> dict[str, npt.NDArray[np.int32]]:
    """Well-matrix coupling index vectors, built once and cached on the well mesh."""
    wmesh = mesh_data.submesh
    if "vec_mat_asmbl" not in wmesh:
        wmesh["vec_mat_asmbl"] = _create_well_coupling_vectors(
            mesh_data.conn,
            wmesh["matrix_conn"],
            wmesh["matrix_node_conn"],
            np.concatenate(mesh_data.region_lists),
        )
    vectors: dict[str, npt.NDArray[np.int32]] = wmesh["vec_mat_asmbl"]
    return vectors


def _stack_well_matrices(
    body_matrices: dict[str, csc_matrix[Any]],
    well_matrices: dict[str, csc_matrix[Any]],
    nn_m: int,
    nn_f: int,
    nn_w: int,
) -> dict[str, csc_matrix[Any]]:
    """Stack the well and (matrix + fracture) matrices along with their off-diagonal coupling matrices.

    Wells couple to matrix nodes only, so coupling matrices are zero-padded over the fracture nodes.
    """
    nn_b = nn_m + nn_f  # matrix + fracture
    for name in body_matrices:
        # On-diagonal well matrix
        if name in well_matrices:
            well_matrix = well_matrices[name].tocoo()
        else:
            well_matrix = csc_matrix((nn_w, nn_w)).tocoo()

        # on-diagonal correction to the well's own block
        if name + "_ww" in well_matrices:
            well_matrix = well_matrix + well_matrices[name + "_ww"].tocoo()

        # Off-diagonal coupling matrix (row=matrix, col=well)
        if name + "_wm" in well_matrices:
            wm_matrix = _pad_sparse(well_matrices[name + "_wm"], (nn_b, nn_w)).tocoo()
        else:
            wm_matrix = csc_matrix((nn_b, nn_w)).tocoo()

        # Off-diagonal coupling matrix (row=well, col=matrix)
        if name + "_mw" in well_matrices:
            mw_matrix = _pad_sparse(well_matrices[name + "_mw"], (nn_w, nn_b)).tocoo()
        else:
            mw_matrix = csc_matrix((nn_w, nn_b)).tocoo()

        # on-diagonal correction to the matrix's own block
        matrix_matrix = body_matrices[name].tocoo()
        if name + "_mm" in well_matrices:
            matrix_matrix = matrix_matrix + _pad_sparse(well_matrices[name + "_mm"], (nn_b, nn_b)).tocoo()

        row1 = hstack([matrix_matrix.tocoo(), wm_matrix])
        row2 = hstack([mw_matrix, well_matrix])
        body_matrices[name] = vstack([row1, row2]).tocsc()
    return body_matrices


def _stack_well_vectors(
    body_vectors: dict[str, csc_matrix[Any]],
    well_vectors: dict[str, csc_matrix[Any]],
) -> dict[str, csc_matrix[Any]]:
    """Stack body-mesh and well-mesh residual vectors into one `(nn_m + nn_f + nn_w, 1)` vector."""
    for name in body_vectors:
        body_vectors[name] = vstack([body_vectors[name], well_vectors[name]]).tocsc()
    return body_vectors


def _get_region_mesh_data(mesh: dict[str, Any], *, is_fracture: bool, is_well: bool) -> RegionMeshData:
    """Resolve mesh/region data from the bulk mesh, or the fracture/well mesh."""
    # Well case: 1D elements along the well
    if is_well:
        wmesh = mesh["well_mesh"]
        return RegionMeshData(
            wmesh["nn"],
            1,
            wmesh["conn"],
            wmesh["nne"],
            wmesh["element_type"],
            wmesh["elem_indx"],
            wmesh["n_quadrature"],
            wmesh["region_lists"],
            len(wmesh["region_lists"]),
            wmesh,
            wmesh["nodes"],
        )
    # Fracture case
    if is_fracture:
        fmesh = mesh["fracture_mesh"]
        return RegionMeshData(
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
    return RegionMeshData(
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
    mesh_data: RegionMeshData,
    r: int,
    *,
    is_fracture: bool,
    is_well: bool,
) -> RegionFlowData:
    """Resolve per-region permeability and node-projection data."""
    nan = float("nan")

    # Well case
    if is_well:
        wells = material["wells"]
        nan_arr = np.asarray(np.nan)
        A = wells["area"][r]
        return RegionFlowData(
            kf=A**2 / 8 / np.pi,
            isotropic_perm=True,
            nodes_r=mesh_data.submesh_nodes,  # projected per element by its tangent
            w0=A,
            u=mesh_data.submesh["well_tangent"],
            kf_perp=PropertyResult(nan_arr, nan_arr, nan_arr),  # leak-off uses perforation_conductance
            diameter=wells["diameter"][r],
            radius=wells["radius"][r],
            perforation_conductance=wells["perforation_conductance"][r],
            roughness=wells["roughness"][r],
        )

    # Fracture case
    if is_fracture:
        w = material["fracture"]["aperture"][r]
        kf_perp = np.asarray(w / 12)  # transverse permeability, coupling only; constant, so zero derivatives
        u = mesh_data.submesh["fracture_tangent"][r].reshape(-1, 1)
        zeros = np.zeros_like(kf_perp)
        return RegionFlowData(
            kf=w**3 / 12,  # longitudinal permeability, cubic law
            isotropic_perm=True,
            nodes_r=mesh_data.submesh_nodes @ u,  # project mesh nodes into local coordinates
            w0=w,
            u=u,
            kf_perp=PropertyResult(kf_perp, zeros, zeros),
            diameter=nan,
            radius=nan,
            perforation_conductance=nan,
            roughness=nan,
        )

    # Bulk case
    kf = material["solid"]["permeability_eff"][r]
    nan_arr = np.asarray(np.nan)
    return RegionFlowData(
        kf=kf,
        isotropic_perm=bool(isinstance(kf, (float, int))),
        nodes_r=mesh["nodes"],
        w0=nan,
        u=np.full((1, 1), np.nan),
        kf_perp=PropertyResult(nan_arr, nan_arr, nan_arr),
        diameter=nan,
        radius=nan,
        perforation_conductance=nan,
        roughness=nan,
    )


def _get_region_thermal_data(
    material: dict[str, Any], r: int, *, is_fracture: bool, is_well: bool
) -> tuple[bool, float]:
    """Resolve the region's isotropic-conductivity flag and heat transfer factor.

    The heat transfer factor multiplies the fracture/well heat transfer coefficient; it is a
    NaN sentinel for the bulk mesh (coupling-only quantity).
    """
    # Well case
    if is_well:
        return True, material["wells"]["heat_transfer_factor"][r]

    # Fracture case
    if is_fracture:
        return True, material["fracture"]["heat_transfer_factor"][r]

    # Bulk case
    lambda_e = material["solid"]["therm_cond_eff"][r]
    isotropic_cond = bool(isinstance(lambda_e, (float, int)))
    return isotropic_cond, float("nan")


def _get_region_solid_data(material: dict[str, Any], r: int, *, no_solid: bool) -> RegionSolidData:
    """Resolve per-region solid properties.

    Fractures and wells are void space with no solid properties.
    """
    # Fracture/well case
    if no_solid:
        zero = 0.0
        return RegionSolidData(
            K_d=zero, phi_0=zero, biot=zero, rho_s=zero, cp_s=zero, alpha_s=zero, lambda_s=zero
        )

    # Bulk case
    return RegionSolidData(
        K_d=_get_region_value(material["solid"]["drained_bulk_mod"], r),
        phi_0=_get_region_value(material["solid"]["porosity"], r),
        biot=_get_region_value(material["solid"]["biot_coeff"], r),
        rho_s=_get_region_value(material["solid"]["density"], r),
        cp_s=_get_region_value(material["solid"]["specific_heat"], r),
        alpha_s=_get_region_value(material["solid"]["therm_expansion"], r),
        lambda_s=_get_region_value(material["solid"]["therm_cond"], r),
    )


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


# ---- Physics helpers: called by both residual and tangent integrations

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
    want_tangent_dT: bool = False,
) -> tuple[
    npt.NDArray[np.float64],  # flux_p_q
    npt.NDArray[np.float64],  # dfluxp_dpf, `_EMPTY` unless want_tangent_dp
    npt.NDArray[np.float64],  # dfluxp_dpm, `_EMPTY` unless want_tangent_dp
    npt.NDArray[np.float64],  # dfluxp_dTf, `_EMPTY` unless want_tangent_dT
    npt.NDArray[np.float64],  # dfluxp_dTm, `_EMPTY` unless want_tangent_dT
]:
    """Evaluate the fracture-matrix leak-off mass flux and optionally its pressure tangent."""
    flux_coeff = rhof_q.value * kf_perp.value / mu_q.value  # leak-off conductance
    flux_p_q = flux_coeff * (p_f_q - p_m_q)  # leak-ff mass flux

    dfluxp_dpf = dfluxp_dpm = dfluxp_dTf = dfluxp_dTm = _EMPTY
    if want_tangent_dp:
        dcoeff_dp = rhof_q.dp * kf_perp.value + rhof_q.value * kf_perp.dp - flux_coeff * mu_q.dp
        dfluxp_dpf = flux_coeff + dcoeff_dp / mu_q.value * (p_f_q - p_m_q)
        dfluxp_dpm = -flux_coeff
    if want_tangent_dT:
        dcoeff_dT = rhof_q.dT * kf_perp.value + rhof_q.value * kf_perp.dT - flux_coeff * mu_q.dT
        dfluxp_dTf = dcoeff_dT / mu_q.value * (p_f_q - p_m_q)
        dfluxp_dTm = np.zeros_like(dfluxp_dTf)  # Flux properties are evaluated with fracture state

    return flux_p_q, dfluxp_dpf, dfluxp_dpm, dfluxp_dTf, dfluxp_dTm


def _evaluate_fracture_heat_transfer_coefficient(
    heat_transfer_factor: float, lambda_f: float, aperture: float
) -> float:
    """Fracture-matrix heat transfer coefficient h = factor * Nu * lambda_f / (2w)."""
    # Cubic-law flow is laminar: the dummy Re = 1 (and Pr = 1) always evaluates the laminar Nu
    Nu, _, _ = get_nusselt_number(np.ones(1), np.ones(1), 0.0, parallel_plates=True)
    return float(heat_transfer_factor * Nu.item() * lambda_f / (2.0 * aperture))  # hydraulic diameter 2w


def _evaluate_well_heat_transfer_coefficient(
    heat_transfer_factor: float,
    lambda_f: float,
    radius: float,
    roughness: float,
    Re: npt.NDArray[np.float64],
    Pr: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Well-matrix heat transfer coefficient h = factor * Nu(Re, Pr) * lambda_f / (2R), and dh/dRe, dh/dPr."""
    Nu, dNu_dRe, dNu_dPr = get_nusselt_number(Re, Pr, roughness / (2.0 * radius), parallel_plates=False)
    h_scale = heat_transfer_factor * lambda_f / (2.0 * radius)  # hydraulic diameter 2R
    return h_scale * Nu, h_scale * dNu_dRe, h_scale * dNu_dPr


def _evaluate_fracture_conduction_flux(
    heat_transfer_factor: float,
    lambda_f: float,
    aperture: float,
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
    hc = _evaluate_fracture_heat_transfer_coefficient(heat_transfer_factor, lambda_f, aperture)
    flux_hc_T_q = hc * (T_f_q - T_m_q)  # convection heat flux
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
    flux_p_q: npt.NDArray[np.float64],  # MAss flux
    fluid: dict[str, Any],
    kf_perp: PropertyResult,
    mu_q: PropertyResult,
    rhof_q: PropertyResult,
    *,
    dfluxp_dpf: npt.NDArray[np.float64] = _EMPTY,
    dfluxp_dpm: npt.NDArray[np.float64] = _EMPTY,
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
        # Use fracture properties
        T_eval, T_flow_eval, p_eval = T_f_q, T_flow_fq, p_f_q
    else:
        # Use matrix properties
        T_eval, T_flow_eval, p_eval = T_m_q, T_flow_mq, p_m_q

    _, _, hf_q = get_EOS(
        T_eval,
        T_flow_eval,
        p_eval,
        fluid,
        T_flow_lagged=T_flow_lagged,
        want_enthalpy=True,
        want_dp=want_tangent,
        want_dT=want_tangent,
    )
    flux_adv_T_q = hf_q.value * flux_p_q  # Advective heat flux

    dqadv_dTf = dqadv_dTm = dqadv_dpf = dqadv_dpm = _EMPTY
    if want_tangent:
        flux_coeff = rhof_q.value * kf_perp.value / mu_q.value
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


def _evaluate_well_leakoff_flux(
    rhof_q: PropertyResult,
    perforation_conductance: float,  # well-to-rock conductance per unit wall area (m); 0 is unperforated
    radius: float,
    mu_q: PropertyResult,
    p_w_q: npt.NDArray[np.float64],
    p_m_q: npt.NDArray[np.float64],
    *,
    want_tangent_dp: bool = False,
    want_tangent_dT: bool = False,
) -> tuple[
    npt.NDArray[np.float64],  # flux_p_q
    npt.NDArray[np.float64],  # dfluxp_dpw, `_EMPTY` unless want_tangent_dp
    npt.NDArray[np.float64],  # dfluxp_dpm, `_EMPTY` unless want_tangent_dp
    npt.NDArray[np.float64],  # dfluxp_dTw, `_EMPTY` unless want_tangent_dT
    npt.NDArray[np.float64],  # dfluxp_dTm, `_EMPTY` unless want_tangent_dT
]:
    """Evaluate the well-matrix leak-off mass flux per unit well length and optionally its tangents.

    Flux = P C rho / mu (p_w - p_m), with wetted perimeter P = 2 pi R and perforation conductance C.
    """
    leakoff_conductance = 2.0 * np.pi * radius * perforation_conductance  # per unit well length (m^2)
    flux_coeff = rhof_q.value * leakoff_conductance / mu_q.value
    flux_p_q = flux_coeff * (p_w_q - p_m_q)  # leak-off mass flux

    dfluxp_dpw = dfluxp_dpm = dfluxp_dTw = dfluxp_dTm = _EMPTY
    if want_tangent_dp:
        dcoeff_dp = rhof_q.dp * leakoff_conductance - flux_coeff * mu_q.dp
        dfluxp_dpw = flux_coeff + dcoeff_dp / mu_q.value * (p_w_q - p_m_q)
        dfluxp_dpm = -flux_coeff
    if want_tangent_dT:
        dcoeff_dT = rhof_q.dT * leakoff_conductance - flux_coeff * mu_q.dT
        dfluxp_dTw = dcoeff_dT / mu_q.value * (p_w_q - p_m_q)
        dfluxp_dTm = np.zeros_like(dfluxp_dTw)  # Flux properties are evaluated with well state

    return flux_p_q, dfluxp_dpw, dfluxp_dpm, dfluxp_dTw, dfluxp_dTm


def _evaluate_well_conduction_flux(
    heat_transfer_factor: float,
    lambda_f: float,
    radius: float,
    roughness: float,
    Re_c: npt.NDArray[np.float64],  # element (midpoint) Reynolds number
    mu_c: PropertyResult,  # midpoint viscosity
    cp_c: npt.NDArray[np.float64],  # midpoint specific heat, lagged in the tangent
    N_c: npt.NDArray[np.float64],  # midpoint shape functions
    N_q: npt.NDArray[np.float64],
    T_w_q: npt.NDArray[np.float64],
    T_m_q: npt.NDArray[np.float64],
    *,
    dRe_dp: npt.NDArray[np.float64] = _EMPTY,
    dRe_dT: npt.NDArray[np.float64] = _EMPTY,
    want_tangent_dT: bool = False,
    want_tangent_dp: bool = False,
) -> tuple[
    npt.NDArray[np.float64],  # flux_hc_T_q
    npt.NDArray[np.float64],  # dfluxc_dTw, (nq, 1, nne); `_EMPTY` unless want_tangent_dT
    npt.NDArray[np.float64],  # dfluxc_dTm, (nq, 1, nne); `_EMPTY` unless want_tangent_dT
    npt.NDArray[np.float64],  # dfluxc_dpw, (nq, 1, nne); `_EMPTY` unless want_tangent_dp
]:
    """Evaluate the well-matrix convective conduction flux per unit well length and its well node tangents.

    Flux = h P (T_w - T_m), with wetted perimeter P = 2 pi R and
    h = heat_transfer_factor * Nu(Re, Pr) * lambda_f / D, Pr = mu cp / lambda_f, evaluated once
    per element at its midpoint.
    """
    # Element heat transfer coefficient from the pipe Nusselt number
    Pr = mu_c.value * cp_c / lambda_f
    hc, dh_dRe, dh_dPr = _evaluate_well_heat_transfer_coefficient(
        heat_transfer_factor, lambda_f, radius, roughness, Re_c, Pr
    )

    perimeter = 2.0 * np.pi * radius  # wetted perimeter: h is per unit wall area
    dT_q = T_w_q - T_m_q
    flux_hc_T_q = hc * perimeter * dT_q  # convection heat flux per unit length

    # h depends on the well state through Re (flow) and Pr (viscosity; cp lagged)
    dfluxc_dTw = dfluxc_dTm = dfluxc_dpw = _EMPTY
    if want_tangent_dT:
        dh_dTw = dh_dRe * dRe_dT + dh_dPr * cp_c / lambda_f * mu_c.dT * N_c
        dfluxc_dTw = perimeter * (hc * N_q + dT_q * dh_dTw)
        dfluxc_dTm = -perimeter * hc * N_q
    if want_tangent_dp:
        dh_dpw = dh_dRe * dRe_dp + dh_dPr * cp_c / lambda_f * mu_c.dp * N_c
        dfluxc_dpw = perimeter * dT_q * dh_dpw
    return flux_hc_T_q, dfluxc_dTw, dfluxc_dTm, dfluxc_dpw


def _evaluate_well_advection_flux(
    p_w_q: npt.NDArray[np.float64],
    p_m_q: npt.NDArray[np.float64],
    p_w_q_lag: npt.NDArray[np.float64],
    p_m_q_lag: npt.NDArray[np.float64],
    T_w_q: npt.NDArray[np.float64],
    T_m_q: npt.NDArray[np.float64],
    T_flow_wq: npt.NDArray[np.float64],
    T_flow_mq: npt.NDArray[np.float64],
    flux_p_q: npt.NDArray[np.float64],  # Mass flux
    fluid: dict[str, Any],
    perforation_conductance: float,  # matching the leak-off flux
    radius: float,
    mu_q: PropertyResult,
    rhof_q: PropertyResult,
    *,
    dfluxp_dpw: npt.NDArray[np.float64] = _EMPTY,
    dfluxp_dpm: npt.NDArray[np.float64] = _EMPTY,
    want_tangent: bool = False,
    T_flow_lagged: bool = False,
) -> tuple[
    npt.NDArray[np.float64],  # flux_adv_T_q
    npt.NDArray[np.float64],  # dqadv_dTw, `_EMPTY` unless want_tangent
    npt.NDArray[np.float64],  # dqadv_dTm, `_EMPTY` unless want_tangent
    npt.NDArray[np.float64],  # dqadv_dpw, `_EMPTY` unless want_tangent
    npt.NDArray[np.float64],  # dqadv_dpm, `_EMPTY` unless want_tangent
]:
    """Evaluate the well-matrix advective enthalpy flux and tangents, with the well leak-off conductance.

    Upwind direction is picked from the lagged pressures `p_w_q_lag`/`p_m_q_lag`.
    """
    well_upwind = np.mean(p_w_q_lag) > np.mean(p_m_q_lag)
    if well_upwind:
        # Use well properties
        T_eval, T_flow_eval, p_eval = T_w_q, T_flow_wq, p_w_q
    else:
        # Use matrix properties
        T_eval, T_flow_eval, p_eval = T_m_q, T_flow_mq, p_m_q

    _, _, hf_q = get_EOS(
        T_eval,
        T_flow_eval,
        p_eval,
        fluid,
        T_flow_lagged=T_flow_lagged,
        want_enthalpy=True,
        want_dp=want_tangent,
        want_dT=want_tangent,
    )
    flux_adv_T_q = hf_q.value * flux_p_q  # Advective heat flux

    dqadv_dTw = dqadv_dTm = dqadv_dpw = dqadv_dpm = _EMPTY
    if want_tangent:
        leakoff_conductance = 2.0 * np.pi * radius * perforation_conductance
        flux_coeff = rhof_q.value * leakoff_conductance / mu_q.value
        dcoeff_dT = rhof_q.dT * leakoff_conductance - flux_coeff * mu_q.dT
        if well_upwind:
            dqadv_dTw = hf_q.dT * flux_p_q + hf_q.value * dcoeff_dT / mu_q.value * (p_w_q - p_m_q)
            dqadv_dTm = np.zeros_like(dqadv_dTw)
            dqadv_dpw = hf_q.dp * flux_p_q + hf_q.value * dfluxp_dpw
            dqadv_dpm = hf_q.value * dfluxp_dpm
        else:
            dqadv_dTw = hf_q.value * dcoeff_dT / mu_q.value * (p_w_q - p_m_q)
            dqadv_dTm = hf_q.dT * flux_p_q
            dqadv_dpw = hf_q.value * dfluxp_dpw
            dqadv_dpm = hf_q.dp * flux_p_q + hf_q.value * dfluxp_dpm

    return flux_adv_T_q, dqadv_dTw, dqadv_dTm, dqadv_dpw, dqadv_dpm


# Sentinel default for an optional `ElementKinematics` parameter (mirrors `_EMPTY` for arrays
# and `PropertyResult(_EMPTY, _EMPTY, _EMPTY)` for an optional `PropertyResult` parameter).
_EMPTY_KINEMATICS = ElementKinematics(
    _EMPTY, _EMPTY, _EMPTY, _EMPTY, _EMPTY, _EMPTY, _EMPTY, _EMPTY, _EMPTY, _EMPTY, _EMPTY
)


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
    *,
    want_lengths: bool,
) -> ElementKinematics:
    """Slice this element's nodal primary variables and interpolate them to quadrature points.

    `want_lengths` computes `L_vec` (SUPG only); otherwise it is `_EMPTY`.
    """
    Te = T[enodes_dof]  # element nodal temperatures
    Te_flow = T_flow[enodes_dof]  # element nodal temperatures, for flow properties
    pe = p[enodes_dof]  # element nodal pressures
    T_0e = T_0[enodes_dof].reshape(-1, 1)  # element reference temperatures
    p_0e = p_0[enodes_dof].reshape(-1, 1)  # element reference pressures
    L_vec = (xI.max(axis=0) - xI.min(axis=0)).reshape(-1, 1) if want_lengths else _EMPTY  # element lengths

    T_q = N_q @ Te  # quadrature point temperatures, nq x 1 x (n, n-1, n-2, n-3)
    T_flow_q = N_q @ Te_flow  # quadrature point temperatures, for flow properties
    p_q = N_q @ pe  # quadrature point pressures, nq x 1 x (n, n-1, n-2, n-3)
    gradT = B_q @ Te[:, 0:1]  # qudarature point temperature gradient, nq x nsd x 1
    gradP = B_q @ pe[:, 0:1]  # quadrature point pressure gradient, nq x nsd x 1

    return ElementKinematics(Te, Te_flow, pe, T_0e, p_0e, L_vec, T_q, T_flow_q, p_q, gradT, gradP)


def _property_derivative_flags(calc_derivatives: bool, solve_p: bool, solve_T: bool) -> tuple[bool, bool]:
    """Resolve (want_dp, want_dT) for property evaluations: only wrt fields solved in this domain."""
    return calc_derivatives and solve_p, calc_derivatives and solve_T


def _get_pass_dofs(mesh: dict[str, Any], nn: int, *, is_fracture: bool, is_well: bool) -> slice:
    """Global DoFs of the bulk, fracture, or well mesh nodes, in local node order."""
    offset = 0
    if is_well:
        offset = mesh["nn_total"] - nn  # wells are numbered last
    elif is_fracture:
        offset = mesh["nn"]
    return slice(offset, offset + nn)


@dataclass(frozen=True, slots=True)
class NodalFluidState:
    """EOS results at every node of the current mesh pass, gathered per element in the loop."""

    rho_f: PropertyResult
    u_f: PropertyResult
    hf: EnthalpyResult


def _evaluate_nodal_fluid_state(
    T: npt.NDArray[np.float64],
    T_flow: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    fluid: dict[str, Any],
    *,
    calc_derivatives: bool = False,
    T_flow_lagged: bool = False,
    solve_p: bool = True,
    solve_T: bool = True,
    want_energy: bool = True,
) -> NodalFluidState:
    """Evaluate density, internal energy, and enthalpy once per node.

    Density is evaluated with T_flow for consistency with the mass flow rates.
    Internal energy is evaluated with T, the current temperature field.
    Enthalpy uses a mixed formulation. It is reconstructed from the combination of u and rho.
    """
    want_dp, want_dT = _property_derivative_flags(calc_derivatives, solve_p, solve_T)
    rho_f, u_f, hf = get_EOS(
        T,
        T_flow,
        p,
        fluid,
        T_flow_lagged=T_flow_lagged,
        want_density=True,
        want_internal_energy=want_energy,
        want_enthalpy=want_energy,
        want_dp=want_dp,
        want_dT=want_dT or want_energy,
    )
    return NodalFluidState(rho_f, u_f, hf)


def _evaluate_nodal_porosity(
    p: npt.NDArray[np.float64],
    p_0: npt.NDArray[np.float64],
    T_flow: npt.NDArray[np.float64],
    T_0: npt.NDArray[np.float64],
    solid: RegionSolidData,
    w0: float,
    *,
    no_solid: bool,
    T_flow_lagged: bool = False,
    volumetric_strain_on: bool = True,
) -> PropertyResult:
    """Evaluate one region's Lagrangian porosity once per node.

    For fractures/wells, "porosity" is the fracture aperture or well area.
    """
    if no_solid:
        # Fracture/well case: phi is replaced by aperture/area, a fixed property pending geomechanics.
        return PropertyResult(w0 * np.ones_like(p), np.zeros_like(p), np.zeros_like(p))
    return get_porosity_lagrangian(
        p,
        p_0,
        T_flow,
        T_0,
        solid.phi_0,
        solid.K_d,
        solid.biot,
        solid.alpha_s,
        T_flow_lagged,
        volumetric_strain_on,
    )


@dataclass(frozen=True, slots=True)
class StackedNodalState:
    """Nodal porosity and EOS fields stacked column-wise, so one `take` gathers an element's rows.

    Per-call numpy overhead dwarfs the copy for 4-row gathers, so one `take` instead of 13 halves
    the gather cost. `columns` holds each field's column slice, or None for an `_EMPTY` field.
    """

    stack: npt.NDArray[np.float64]
    columns: tuple[slice | None, ...]  # phi_L (value, dp, dT), rho_f (3), u_f (3), hf (value, dp, dT, cp)


def _stack_nodal_state(phi_L: PropertyResult, nodal_fluid: NodalFluidState) -> StackedNodalState:
    """Stack one region's nodal porosity with the pass's nodal EOS fields."""
    rho_f, u_f, hf = nodal_fluid.rho_f, nodal_fluid.u_f, nodal_fluid.hf
    fields = (
        phi_L.value, phi_L.dp, phi_L.dT,
        rho_f.value, rho_f.dp, rho_f.dT,
        u_f.value, u_f.dp, u_f.dT,
        hf.value, hf.dp, hf.dT, hf.cp,
    )  # fmt: skip
    columns: list[slice | None] = []
    start = 0
    for arr in fields:
        if arr.size:
            columns.append(slice(start, start + arr.shape[1]))
            start += arr.shape[1]
        else:
            columns.append(None)
    return StackedNodalState(np.hstack([f for f in fields if f.size]), tuple(columns))


def _gather_nodal_state(
    state: StackedNodalState, idx: npt.NDArray[np.intp]
) -> tuple[PropertyResult, PropertyResult, PropertyResult, EnthalpyResult]:
    """Gather an element's nodal (phi_L, rho_f, u_f, hf) rows."""
    rows = state.stack.take(idx, axis=0)
    f = [_EMPTY if s is None else rows[:, s] for s in state.columns]
    return (
        PropertyResult(f[0], f[1], f[2]),
        PropertyResult(f[3], f[4], f[5]),
        PropertyResult(f[6], f[7], f[8]),
        EnthalpyResult(f[9], f[10], f[11], f[12]),
    )


def _evaluate_flow_and_storage_state(
    kf_0: npt.NDArray[np.float64] | float,
    kin_n: ElementKinematics,
    N_q: npt.NDArray[np.float64],
    fluid: dict[str, Any],
    nodal_state: StackedNodalState,
    enodes: npt.NDArray[np.intp],
    *,
    calc_derivatives: bool = False,
    T_flow_lagged: bool = False,
    solve_p: bool = True,
    solve_T: bool = True,
    want_energy: bool = True,
) -> tuple[
    PropertyResult,  # kf
    PropertyResult,  # mu_q
    NodalPropertyState,  # rho_f
    NodalPropertyState,  # phi_L
    NodalPropertyState,  # u_f
    NodalEnthalpyState,  # hf
]:
    """Evaluate permeability and viscosity, and gather nodal porosity, density, and internal energy.

    `enodes` indexes this element's rows in `nodal_state`.
    """
    want_dp, want_dT = _property_derivative_flags(calc_derivatives, solve_p, solve_T)

    # ---- Quadrature-point flow properties
    kf = get_effective_permeability(kf_0, kin_n.T_flow_q, kin_n.p_q, fluid, T_flow_lagged)
    mu_q = get_viscosity(kin_n.T_flow_q, kin_n.p_q, fluid, T_flow_lagged, want_dp, want_dT)

    phi_L_e, rho_f_e, u_f_e, hf_e = _gather_nodal_state(nodal_state, enodes)

    # ---- Porosity
    phi_L_q = PropertyResult(N_q @ phi_L_e.value, N_q @ phi_L_e.dp[:, 0:1], N_q @ phi_L_e.dT[:, 0:1])
    phi_L = NodalPropertyState(q=phi_L_q, e=phi_L_e)

    # ---- Equation of state variables: density, internal energy, and enthalpy
    rho_f_q = PropertyResult(N_q @ rho_f_e.value, N_q @ rho_f_e.dp[:, 0:1], N_q @ rho_f_e.dT[:, 0:1])
    rho_f = NodalPropertyState(q=rho_f_q, e=rho_f_e)

    if want_energy:
        u_f_q = PropertyResult(N_q @ u_f_e.value, N_q @ u_f_e.dp[:, 0:1], N_q @ u_f_e.dT[:, 0:1])
        hf_q = EnthalpyResult(
            N_q @ hf_e.value, N_q @ hf_e.dp[:, 0:1], N_q @ hf_e.dT[:, 0:1], N_q @ hf_e.cp[:, 0:1]
        )
    else:
        u_f_q = u_f_e
        hf_q = hf_e
    u_f = NodalPropertyState(q=u_f_q, e=u_f_e)
    hf = NodalEnthalpyState(q=hf_q, e=hf_e)

    return kf, mu_q, rho_f, phi_L, u_f, hf


def _evaluate_darcy_flux(
    N_q: npt.NDArray[np.float64],
    B_q: npt.NDArray[np.float64],
    gradP: npt.NDArray[np.float64],
    g_r: npt.NDArray[np.float64],
    kf: PropertyResult,
    mu_q: PropertyResult,
    rho_f: PropertyState,
    *,
    isotropic_perm: bool,
    want_tangent_dp: bool = False,
    want_tangent_dT: bool = False,
) -> tuple[
    npt.NDArray[np.float64],  # darcy_flux
    npt.NDArray[np.float64],  # q_D
    npt.NDArray[np.float64],  # dqD_dp, `_EMPTY` unless want_tangent_dp
    npt.NDArray[np.float64],  # dqD_dT, `_EMPTY` unless want_tangent_dT
]:
    """Evaluate the Darcy flux and derivatives at quadrature points."""
    rho_f_n = rho_f.q_n
    gradP_mod = gradP - rho_f_n.value * g_r  # modified pressure gradient
    kfmu = kf.value / mu_q.value
    darcy_flux = -kfmu * gradP_mod if isotropic_perm else -kfmu @ gradP_mod  # volume flux
    q_D = rho_f_n.value * darcy_flux  # mass flux

    dqD_dp = _EMPTY
    if want_tangent_dp:
        term_2 = (
            kfmu * rho_f_n.dp
            + rho_f_n.value / mu_q.value * kf.dp
            - rho_f_n.value * kf.value / mu_q.value**2 * mu_q.dp
        )
        if isotropic_perm:
            dqD_dp = rho_f_n.value * kfmu * (B_q + rho_f_n.dp * g_r * N_q) + term_2 * gradP_mod * N_q
        else:
            dqD_dp = rho_f_n.value * kfmu @ (B_q + rho_f_n.dp * g_r * N_q) + term_2 @ gradP_mod * N_q

    dqD_dT = _EMPTY
    if want_tangent_dT:
        term_2 = (
            kfmu * rho_f_n.dT
            + rho_f_n.value / mu_q.value * kf.dT
            - rho_f_n.value * kf.value / mu_q.value**2 * mu_q.dT
        )
        if isotropic_perm:
            dqD_dT = -rho_f_n.value * kfmu * (rho_f_n.dT * g_r * N_q) + term_2 * gradP_mod * N_q
        else:
            dqD_dT = -rho_f_n.value * kfmu @ (rho_f_n.dT * g_r * N_q) + term_2 @ gradP_mod * N_q

    return darcy_flux, q_D, dqD_dp, dqD_dT


_WELL_FLOW_TOL = 1.0e-10  # relative tolerance of the well momentum residual
_WELL_FLOW_MAX_ITER = 50
_WELL_FLOW_MAX_HALVINGS = 20
_N_MID_L2 = np.full((1, 1, 2), 0.5)  # L2 shape functions at the element midpoint


def _well_flow_residual(
    q: npt.NDArray[np.float64],
    G: npt.NDArray[np.float64],
    rho: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    grad_rhof: npt.NDArray[np.float64],
    radius: float,
    rel_roughness: float,
) -> tuple[npt.NDArray[np.float64], ...]:
    """Well momentum residual F = q|q| (f - sign(q) 4R drho/ds / rho) + 4 rho pi^2 R^5 G and dF/dq.

    Written with phi = f Re and |q| = a Re so it stays finite at zero flow.
    """
    a = 0.5 * np.pi * radius * mu  # |q| = a Re
    Re = np.abs(q) / a
    phi, dphi = get_friction_factor(Re, rel_roughness)
    c = 4.0 * radius * grad_rhof / rho  # density gradient (acceleration) coefficient
    K = 4.0 * rho * np.pi**2 * radius**5

    F = a * phi * q - c * q**2 + K * G
    dF_dq = a * (phi + Re * dphi) - 2.0 * c * q
    F_scale = np.abs(K * G) + np.abs(a * phi * q) + np.abs(c * q**2)  # convergence reference
    return F, dF_dq, F_scale, Re, phi, dphi


def _evaluate_well_flow(
    N_c: npt.NDArray[np.float64],
    B_c: npt.NDArray[np.float64],
    gradP: npt.NDArray[np.float64],
    g_r: npt.NDArray[np.float64],
    mu_c: PropertyResult,
    rho_e: PropertyResult,
    radius: float,
    roughness: float,
    *,
    want_tangent_dp: bool = False,
    want_tangent_dT: bool = False,
) -> tuple[
    npt.NDArray[np.float64],  # volume flow rate
    npt.NDArray[np.float64],  # q_w, mass flow rate
    npt.NDArray[np.float64],  # dqD_dp, (1, 1, nne); `_EMPTY` unless want_tangent_dp
    npt.NDArray[np.float64],  # dqD_dT, (1, 1, nne); `_EMPTY` unless want_tangent_dT
    npt.NDArray[np.float64],  # Re, Reynolds number
    npt.NDArray[np.float64],  # dRe_dp, (1, 1, nne); `_EMPTY` unless want_tangent_dp
    npt.NDArray[np.float64],  # dRe_dT, (1, 1, nne); `_EMPTY` unless want_tangent_dT
]:
    """Evaluate the well element's mass flow rate, Reynolds number, and nodal derivatives.

    Solves q|q| (f - sign(q) 4R/rho drho/ds) = -4 rho pi^2 R^5 (dp/ds + rho g dz/ds) by
    safeguarded Newton, with the Churchill Darcy friction factor f(Re). On an L2 element the
    pressure and density gradients are constant, so one evaluation at the midpoint (shape
    functions N_c, gradients B_c) gives the element flow; rho is the average nodal density, the
    exact element mean of the gravity term. Derivatives follow `_evaluate_darcy_flux`'s sign
    convention (dqD_dp = -dq/dp).
    """
    mu = mu_c.value
    rel_roughness = roughness / (2.0 * radius)

    # Element density, modified pressure gradient, and density gradient
    rho = N_c @ rho_e.value  # average nodal density
    G = gradP - rho * g_r  # modified pressure gradient
    grad_rhof = B_c @ rho_e.value  # density gradient along the well

    # Initial guess without acceleration: lesser of the laminar and explicit Colebrook flow rates
    a = 0.5 * np.pi * radius * mu  # |q| = a Re
    Lam = 16.0 * rho * radius**3 * np.abs(G) / mu**2  # known f Re^2
    sqrt_Lam = np.maximum(np.sqrt(Lam), np.finfo(float).tiny)
    Re_lam = Lam / 64.0
    Re_turb = sqrt_Lam * -2.0 * np.log10(rel_roughness / 3.7 + 2.51 / sqrt_Lam)
    Re_0 = np.where(Re_turb > 0.0, np.minimum(Re_lam, Re_turb), Re_lam)
    q = -np.sign(G) * a * Re_0

    # Safeguarded Newton on the signed mass flow rate
    for _ in range(_WELL_FLOW_MAX_ITER):
        F, dF_dq, F_scale, _, _, _ = _well_flow_residual(q, G, rho, mu, grad_rhof, radius, rel_roughness)
        if np.all(np.abs(F) <= _WELL_FLOW_TOL * F_scale):
            break
        dq = -F / np.where(dF_dq != 0.0, dF_dq, np.finfo(float).tiny)

        # Backtrack: halve the step wherever it does not reduce |F|
        step = np.ones_like(q)
        for _ in range(_WELL_FLOW_MAX_HALVINGS):
            F_trial = _well_flow_residual(q + step * dq, G, rho, mu, grad_rhof, radius, rel_roughness)[0]
            worse = np.abs(F_trial) > np.abs(F)
            if not np.any(worse):
                break
            step = np.where(worse, 0.5 * step, step)
        q = q + step * dq
    else:
        warnings.warn("Well flow rate did not converge in some well elements.", stacklevel=2)

    vol_flow = q / rho  # volume flow rate

    Re = 2.0 * np.abs(q) / (np.pi * radius * mu)  # Reynolds number

    dqD_dp = dqD_dT = dRe_dp = dRe_dT = _EMPTY
    if want_tangent_dp or want_tangent_dT:
        # Implicit function theorem: dq/dX = -dF_dX / dF_dq at the converged root
        _, dF_dq, _, Re, phi, dphi = _well_flow_residual(q, G, rho, mu, grad_rhof, radius, rel_roughness)
        dF_dG = 4.0 * rho * np.pi**2 * radius**5
        dF_drho = 4.0 * np.pi**2 * radius**5 * G + 4.0 * radius * grad_rhof * q**2 / rho**2
        dF_dgrad_rhof = -4.0 * radius * q**2 / rho
        dF_dmu = 0.5 * np.pi * radius * q * (phi - Re * dphi)

    if want_tangent_dp:
        drho_dp = N_c * rho_e.dp.T  # exact nodal derivative of the average density
        dG_dp = B_c - g_r * drho_dp
        dF_dp = (
            dF_dG * dG_dp + dF_drho * drho_dp + dF_dgrad_rhof * (B_c * rho_e.dp.T) + dF_dmu * mu_c.dp * N_c
        )
        dqD_dp = dF_dp / dF_dq  # = -dq/dp
        # Re = 2|q| / (pi R mu)
        dRe_dp = -2.0 * np.sign(q) / (np.pi * radius * mu) * dqD_dp - Re / mu * mu_c.dp * N_c

    if want_tangent_dT:
        drho_dT = N_c * rho_e.dT.T
        dG_dT = -g_r * drho_dT
        dF_dT = (
            dF_dG * dG_dT + dF_drho * drho_dT + dF_dgrad_rhof * (B_c * rho_e.dT.T) + dF_dmu * mu_c.dT * N_c
        )
        dqD_dT = dF_dT / dF_dq  # = -dq/dT
        dRe_dT = -2.0 * np.sign(q) / (np.pi * radius * mu) * dqD_dT - Re / mu * mu_c.dT * N_c

    return vol_flow, q, dqD_dp, dqD_dT, Re, dRe_dp, dRe_dT


@njit(cache=True)  # type: ignore[untyped-decorator]
def _total_energy_values(
    T: npt.NDArray[np.float64],
    phi_L_value: npt.NDArray[np.float64],
    rho_f_value: npt.NDArray[np.float64],
    u_f_value: npt.NDArray[np.float64],
    rho_s: float,
    cp_s: float,
    no_solid: bool,
) -> npt.NDArray[np.float64]:
    """Numba kernal for total stored energy."""
    solid_factor = 0.0 if no_solid else 1.0
    return (1 - phi_L_value) * rho_s * cp_s * T * solid_factor + phi_L_value * rho_f_value * u_f_value


@njit(cache=True)  # type: ignore[untyped-decorator]
def _mass_storage_integrands_values(
    phi_L_e_value: npt.NDArray[np.float64],
    phi_L_e_dp: npt.NDArray[np.float64],
    phi_L_e_dT: npt.NDArray[np.float64],
    rho_f_e_value: npt.NDArray[np.float64],
    rho_f_e_dp: npt.NDArray[np.float64],
    rho_f_e_dT: npt.NDArray[np.float64],
    cv: npt.NDArray[np.float64],
    steady_pres: float,
    want_tangent_dp: bool,
    want_tangent_dT: bool,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Numba kernel for mass storage integrands."""
    f_mass_e = phi_L_e_value * rho_f_e_value
    mass_dot_e = (1 - steady_pres) * (
        cv[0] * f_mass_e[:, 0:1] + cv[1] * f_mass_e[:, 1:2] + cv[2] * f_mass_e[:, 2:3]
    )

    integrand_dFsdp_e = _EMPTY
    if want_tangent_dp:
        integrand_dFsdp_e = (
            (1 - steady_pres) * cv[0] * (phi_L_e_dp * rho_f_e_value + phi_L_e_value * rho_f_e_dp)
        )

    integrand_dFsdT = _EMPTY
    if want_tangent_dT:
        integrand_dFsdT = (
            (1 - steady_pres) * cv[0] * (phi_L_e_dT * rho_f_e_value + phi_L_e_value * rho_f_e_dT)
        )

    return mass_dot_e, integrand_dFsdp_e, integrand_dFsdT


def _evaluate_mass_storage_integrands(
    phi_L: NodalPropertyState,
    rho_f: NodalPropertyState,
    cv: npt.NDArray[np.float64],
    steady_pres: float,
    *,
    want_tangent_dp: bool = False,
    want_tangent_dT: bool = False,
) -> tuple[
    npt.NDArray[np.float64],  # mass_dot_e
    npt.NDArray[np.float64],  # integrand_dFsdp_e (Kpp), `_EMPTY` unless want_tangent_dp
    npt.NDArray[np.float64],  # integrand_dFsdT (KpT), `_EMPTY` unless want_tangent_dT
]:
    """Evaluate the mass storage rate (nodal, lumped) and optionally its pressure and temperature tangents.

    Mass storage uses a lumped-mass discretization to prevent oscillations, thus uses
    nodal quantities instead of quad point quantities.
    """
    phi_L_e = phi_L.e
    rho_f_e = rho_f.e

    # Evaluated in numba kernel
    mass_dot_e, integrand_dFsdp_e, integrand_dFsdT = _mass_storage_integrands_values(
        phi_L_e.value,
        phi_L_e.dp,
        phi_L_e.dT,
        rho_f_e.value,
        rho_f_e.dp,
        rho_f_e.dT,
        cv,
        steady_pres,
        want_tangent_dp,
        want_tangent_dT,
    )
    return mass_dot_e, integrand_dFsdp_e, integrand_dFsdT


@njit(cache=True)  # type: ignore[untyped-decorator]
def _energy_storage_integrands_values(
    Te: npt.NDArray[np.float64],
    phi_L_e_value: npt.NDArray[np.float64],
    phi_L_e_dp: npt.NDArray[np.float64],
    phi_L_e_dT: npt.NDArray[np.float64],
    rho_f_e_value: npt.NDArray[np.float64],
    rho_f_e_dp: npt.NDArray[np.float64],
    rho_f_e_dT: npt.NDArray[np.float64],
    u_f_e_value: npt.NDArray[np.float64],
    u_f_e_dp: npt.NDArray[np.float64],
    u_f_e_dT: npt.NDArray[np.float64],
    rho_s: float,
    cp_s: float,
    cv: npt.NDArray[np.float64],
    steady_temp: float,
    no_solid: bool,
    kin_n_Te: npt.NDArray[np.float64],
    want_tangent_dT: bool,
    want_tangent_dp: bool,
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]:
    """Numba kernel for energy storage integrands."""
    E_e = _total_energy_values(Te, phi_L_e_value, rho_f_e_value, u_f_e_value, rho_s, cp_s, no_solid)
    energy_dot_e = (1 - steady_temp) * (cv[0] * E_e[:, 0:1] + cv[1] * E_e[:, 1:2] + cv[2] * E_e[:, 2:3])

    solid_factor = 0.0 if no_solid else 1.0
    integrand_dFCdT_e = _EMPTY
    if want_tangent_dT:
        integrand_dFCdT_e = (
            (1 - steady_temp)
            * cv[0]
            * (
                (1 - phi_L_e_value) * rho_s * cp_s * solid_factor
                + phi_L_e_value * (rho_f_e_dT * u_f_e_value + rho_f_e_value * u_f_e_dT)
                + phi_L_e_dT * (rho_f_e_value * u_f_e_value - rho_s * cp_s * kin_n_Te)
            )
        )

    integrand_dFCdp = _EMPTY
    if want_tangent_dp:
        integrand_dFCdp = (
            (1 - steady_temp)
            * cv[0]
            * (
                phi_L_e_value * (rho_f_e_dp * u_f_e_value + rho_f_e_value * u_f_e_dp)
                + phi_L_e_dp * (rho_f_e_value * u_f_e_value - rho_s * cp_s * kin_n_Te)
            )
        )

    return E_e, energy_dot_e, integrand_dFCdT_e, integrand_dFCdp


def _evaluate_energy_storage_integrands(
    kin: ElementKinematics,
    phi_L: NodalPropertyState,
    rho_f: NodalPropertyState,
    u_f: NodalPropertyState,
    solid: RegionSolidData,
    cv: npt.NDArray[np.float64],
    steady_temp: float,
    *,
    no_solid: bool,
    kin_n: ElementKinematics = _EMPTY_KINEMATICS,
    want_tangent_dT: bool = False,
    want_tangent_dp: bool = False,
) -> tuple[
    npt.NDArray[np.float64],  # E_e
    npt.NDArray[np.float64],  # energy_dot_e
    npt.NDArray[np.float64],  # integrand_dFCdT_e (KTT), `_EMPTY` unless want_tangent_dT
    npt.NDArray[np.float64],  # integrand_dFCdp (KTp), `_EMPTY` unless want_tangent_dp
]:
    """Evaluate the nodal (lumped) energy storage rate and optionally temperature and pressure tangents.

    Energy storage uses a lumped-mass discretization to prevent oscillations, thus uses
    nodal quantities instead of quad point quantities.
    """
    phi_L_e = phi_L.e
    rho_f_e = rho_f.e
    u_f_e = u_f.e

    # Evaluated in numba kernel.
    E_e, energy_dot_e, integrand_dFCdT_e, integrand_dFCdp = _energy_storage_integrands_values(
        kin.Te,
        phi_L_e.value,
        phi_L_e.dp,
        phi_L_e.dT,
        rho_f_e.value,
        rho_f_e.dp,
        rho_f_e.dT,
        u_f_e.value,
        u_f_e.dp,
        u_f_e.dT,
        solid.rho_s,
        solid.cp_s,
        cv,
        steady_temp,
        no_solid,
        kin_n.Te,
        want_tangent_dT,
        want_tangent_dp,
    )
    return E_e, energy_dot_e, integrand_dFCdT_e, integrand_dFCdp


def _evaluate_effective_conductivity(
    phi_L: PropertyState,
    solid: RegionSolidData,
    w0: float,
    lambda_f: float,
    nsd: int,
    *,
    no_solid: bool,
    isotropic_cond: bool,
) -> npt.NDArray[np.float64] | float:
    """Evaluate the region's effective thermal conductivity."""
    phi_L_n = phi_L.q_n
    lambda_s = solid.lambda_s
    lambda_e: npt.NDArray[np.float64] | float
    if no_solid:
        lambda_e = w0 * lambda_f
    else:
        lambda_e = (
            phi_L_n.value * lambda_f + (1 - phi_L_n.value) * lambda_s
            if isotropic_cond
            else (phi_L_n.value * lambda_f * np.eye(nsd) + (1 - phi_L_n.value) * lambda_s)
        )

    return lambda_e


@njit(cache=True)  # type: ignore[untyped-decorator]
def _flux_magnitude(flux_vec: npt.NDArray[np.float64], nsd: int) -> float:
    """`|flux_vec|`, resolved per `nsd` since numba can't loop over a runtime-sized slice."""
    if nsd == 3:
        return float(np.sqrt(flux_vec[0, 0] ** 2 + flux_vec[1, 0] ** 2 + flux_vec[2, 0] ** 2))
    if nsd == 2:
        return float(np.sqrt(flux_vec[0, 0] ** 2 + flux_vec[1, 0] ** 2))
    return float(abs(flux_vec[0, 0]))


@njit(cache=True)  # type: ignore[untyped-decorator]
def _effective_length(
    flux_vec: npt.NDArray[np.float64], flux_mag: float, L_vec: npt.NDArray[np.float64], nsd: int
) -> float:
    """SUPG effective length, weighted by the flux direction relative to the element extents."""
    if nsd == 3:
        return float(
            flux_mag
            / np.sqrt(
                (flux_vec[0, 0] / L_vec[0, 0]) ** 2
                + (flux_vec[1, 0] / L_vec[1, 0]) ** 2
                + (flux_vec[2, 0] / L_vec[2, 0]) ** 2
            )
        )
    if nsd == 2:
        return float(
            flux_mag / np.sqrt((flux_vec[0, 0] / L_vec[0, 0]) ** 2 + (flux_vec[1, 0] / L_vec[1, 0]) ** 2)
        )
    return float(L_vec[0, 0])


@njit(cache=True)  # type: ignore[untyped-decorator]
def _supg_tau_values(
    hf_cp: npt.NDArray[np.float64],
    q_D: npt.NDArray[np.float64],
    L_vec: npt.NDArray[np.float64],
    lambda_e: npt.NDArray[np.float64],
    heat_capacity: npt.NDArray[np.float64],  # volumetric heat capacity at quad points
    rate: float,  # transient rate, cv[0] (= 1/dt for first order); 0 when steady
    nsd: int,
    isotropic_cond: bool,
) -> npt.NDArray[np.float64]:
    """Numba kernel for the SUPG stabilization coefficient tau, one quad point at a time.

    `lambda_e` must already be broadcast to a (nq, a, a) shape (a=1 isotropic, a=nsd
    anisotropic), mirroring how a region-constant material property is prepped before
    crossing into a jit kernel elsewhere in this module.
    """
    nq = q_D.shape[0]
    tau = np.empty((nq, 1, 1))
    flux_vec = np.empty((nsd, 1))
    for q in range(nq):
        for d in range(nsd):
            flux_vec[d, 0] = hf_cp[q, 0, 0] * q_D[q, d, 0]

        flux_mag = _flux_magnitude(flux_vec, nsd)

        no_flow = flux_mag == 0.0
        if no_flow:
            # prevent divide by zero
            flux_mag = 1.0e-15
            for d in range(nsd):
                flux_vec[d, 0] = 1.0e-15

        dl_eff = _effective_length(flux_vec, flux_mag, L_vec, nsd)

        if isotropic_cond:
            lambda_e_projected = lambda_e[q, 0, 0]
        else:
            # anisotropic case - project conductivity along flowlines
            n_stream = flux_vec / flux_mag
            lambda_e_projected = (n_stream.T @ lambda_e[q] @ n_stream)[0, 0]

        # Peclet number
        Pe = dl_eff * flux_mag / lambda_e_projected
        # Advective, diffusive and transient (Tezduyar, 2002) limits
        inv_t_sq = (2 * flux_mag / dl_eff) ** 2 * (1 + 9 / Pe**2) + (2 * heat_capacity[q, 0, 0] * rate) ** 2
        t = inv_t_sq ** (-0.5)
        if no_flow:
            t = 0.0
        tau[q, 0, 0] = t

    return tau


def _evaluate_total_energy(
    T: npt.NDArray[np.float64],
    phi_L: PropertyResult,
    rho_f: PropertyResult,
    u_f: PropertyResult,
    solid: RegionSolidData,
    *,
    no_solid: bool,
) -> npt.NDArray[np.float64]:
    """Evaluate the total (solid + fluid) stored energy."""
    # Evaluated in numba kernel.
    E: npt.NDArray[np.float64] = _total_energy_values(
        T, phi_L.value, rho_f.value, u_f.value, solid.rho_s, solid.cp_s, no_solid
    )
    return E


def _evaluate_advection_terms(
    N_q: npt.NDArray[np.float64],  # Shape functions
    B_q: npt.NDArray[np.float64],  # Shape function gradients
    nsd: int,  # number of space dimensions
    nq: int,  # number of quad points
    kin: ElementKinematics,  # kinematics with time history
    kin_n: ElementKinematics,  # kinematics at current iterate
    solid: RegionSolidData,  # solid mat props object
    q_D: npt.NDArray[np.float64],  # mass flux
    rho_f: NodalPropertyState,  # fluid density
    phi_L: PropertyState,  # Lagrangian porosity, or aperture/area (fracture/well case)
    u_f: NodalPropertyState,  # Internal energy
    hf: NodalEnthalpyState,  # Enthalpy
    lambda_e: npt.NDArray[np.float64] | float,  # thermal conductivity
    cv: npt.NDArray[np.float64],  # rate coefficient
    steady_temp: float,  # steady-state temperature control
    *,
    no_solid: bool,
    isotropic_cond: bool,
    dqD_dT: npt.NDArray[np.float64] = _EMPTY,  # mass flux temperature partial  derivative
    dqD_dp: npt.NDArray[np.float64] = _EMPTY,  # mass flux pressure partial derivative
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
    rho_s, cp_s = solid.rho_s, solid.cp_s
    rho_f_n = rho_f.q_n
    hf_q = hf.q_n
    darcy_flux = q_D / rho_f_n.value  # Volume flux

    # lambda_e is a region constant: scalar-like (isotropic fracture case), (nq, 1, 1)
    # (isotropic bulk case), or (nq, nsd, nsd) (anisotropic). Broadcast into a fresh array so
    # the numba kernel always sees the same contiguous, writable type; a read-only view would
    # cost an extra ~1s compiled specialization.
    lambda_e_bcast = np.empty((nq, 1, 1) if isotropic_cond else (nq, nsd, nsd))
    lambda_e_bcast[...] = lambda_e

    # Volumetric heat capacity and transient rate for the tau transient limit
    phi_q = phi_L.q_n.value
    heat_capacity = np.ascontiguousarray(
        (1 - phi_q) * rho_s * cp_s * (1 - no_solid) + phi_q * rho_f_n.value * hf_q.cp
    )
    rate = (1 - steady_temp) * cv[0]
    # Evaluated in numba kernel.
    tau = _supg_tau_values(hf_q.cp, q_D, kin.L_vec, lambda_e_bcast, heat_capacity, rate, nsd, isotropic_cond)

    # Discretized strong form residual at quadrature points
    # Energy rate
    E_q = _evaluate_total_energy(kin.T_q, phi_L.q, rho_f.q, u_f.q, solid, no_solid=no_solid)
    energy_dot = (1 - steady_temp) * (
        cv[0] * E_q[:, :, 0:1] + cv[1] * E_q[:, :, 1:2] + cv[2] * E_q[:, :, 2:3]
    )
    # Enthalpy at nodes
    grad_hf = B_q @ hf.e_n.value  # Enthalpy gradient
    grad_rhof = B_q @ rho_f.e_n.value  # density gradient
    RTh = (
        energy_dot + (grad_hf * rho_f_n.value + hf_q.value * grad_rhof).transpose(0, 2, 1) @ darcy_flux
    )  # discretized residual

    integrand_advection = q_D * hf_q.value
    integrand_SUPG = hf_q.cp * q_D * tau * RTh

    integrand_advection_dT = _EMPTY
    integrand_SUPG_dT = _EMPTY
    if want_tangent_dT:
        # Common terms in dT and dp. want_tangent_dp never appears without want_tangent_dT.
        phi_L_n = phi_L.q_n
        u_f_n = u_f.q_n
        internal_energy_term = rho_f_n.value * u_f_n.value - rho_s * cp_s * kin_n.T_q
        hf_over_rho = hf_q.value / rho_f_n.value
        grad_hf_eff = grad_hf + hf_over_rho * grad_rhof

        # Specific to dT
        D2 = grad_rhof.transpose(0, 2, 1) @ q_D  # d/dTe of (grad_rhof . q_D), nq x 1 x 1
        dqD_dT_direct = -dqD_dT  # +d(q_D)/dT, opposite sign convention from Kpp/KpT's dqD_dT

        # Partial derivative of residual
        dR_dT = (
            (1 - steady_temp)
            * cv[0]
            * (
                (1 - phi_L_n.value) * rho_s * cp_s * (1 - no_solid)
                + phi_L_n.dT * internal_energy_term
                + phi_L_n.value * (rho_f_n.dT * u_f_n.value + rho_f_n.value * u_f_n.dT)
            )
            * N_q
            + (hf_q.dT + hf_over_rho * rho_f_n.dT) * (q_D.transpose(0, 2, 1) @ B_q)
            + grad_hf_eff.transpose(0, 2, 1) @ dqD_dT_direct
            + (hf_q.dT / rho_f_n.value - hf_over_rho * rho_f_n.dT / rho_f_n.value) * D2 * N_q
        )
        integrand_advection_dT = hf_q.dT * q_D * N_q + hf_q.value * dqD_dT_direct
        integrand_SUPG_dT = tau * hf_q.cp * dR_dT * q_D + tau * hf_q.cp * RTh * dqD_dT_direct

    integrand_advection_dp = _EMPTY
    integrand_SUPG_dp = _EMPTY
    if want_tangent_dp:
        dqD_dp_direct = -dqD_dp  # +d(q_D)/dp, opposite sign convention from Kpp/KpT's dqD_dp
        # Partial derivative of residual
        dR_dp = (
            (1 - steady_temp)
            * cv[0]
            * (
                phi_L_n.dp * internal_energy_term
                + phi_L_n.value * (rho_f_n.dp * u_f_n.value + rho_f_n.value * u_f_n.dp)
            )
            * N_q
            + (hf_q.dp + hf_over_rho * rho_f_n.dp) * q_D * B_q
            + grad_hf_eff * dqD_dp_direct
            + (hf_q.dp * grad_rhof / rho_f_n.value - hf_over_rho * grad_rhof / rho_f_n.value * rho_f_n.dp)
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


# ---- Integration functions


def _integrate_fracture_coupling_residual(  # noqa
    e: int,  # fracture element
    enodes_f: list[int],  # fracture element connectivity
    nodes_f: npt.NDArray[np.float64],  # fracture element nodes
    N_q: npt.NDArray[np.float64],  # fracture shape function
    wJe_q: npt.NDArray[np.float64],
    mesh: dict[str, Any],
    fluid: dict[str, Any],  # fluid mat props object
    kin_n: ElementKinematics,  # current-iterate slice
    pe_lag: npt.NDArray[np.float64],  # fracture element lagged pressures - upwind decision only
    p: npt.NDArray[np.float64],  # Full pressure vector
    p_lag: npt.NDArray[np.float64],  # Full pressure vector, lagged - upwind decision only
    T: npt.NDArray[np.float64],  # Full temperature vector
    T_flow: npt.NDArray[np.float64],  # Temperature vector for property evaluation
    kf_perp: PropertyResult,  # fracture transverse permeability at quad points
    mu_q: PropertyResult,  # viscosity at quad points
    rho_f: PropertyState,  # fluid density; `.q_n` is the current-iterate value used here
    heat_transfer_factor: float,  # heat transfer coefficient multiplier
    lambda_f: float,  # fluid thermal conductivity
    aperture: float,
    *,
    pressure_coupling: bool,
    thermal_coupling: bool,
    advection_on: bool,
    T_flow_lagged: bool,
) -> tuple[dict[str, npt.NDArray[np.float64]], list[tuple[int, dict[str, npt.NDArray[np.float64]]]]]:
    """Integrate the fracture-matrix leak-off/convective-exchange flux into residual vectors.

    Returns `(fracture_vectors, matches)`: `fracture_vectors` accumulates onto this fracture
    element's own residual, and `matches` is a list of `(nne_m, matrix_vectors)`, one entry per
    matched matrix element.
    """
    fracture_vectors: dict[str, npt.NDArray[np.float64]] = {}
    matches: list[tuple[int, dict[str, npt.NDArray[np.float64]]]] = []
    if not pressure_coupling and not thermal_coupling:
        return fracture_vectors, matches

    rhof_q = rho_f.q_n
    p_f_q = N_q @ kin_n.pe
    p_f_q_lag = N_q @ pe_lag  # lagged fracture pressure, upwind decision only
    T_f_q = N_q @ kin_n.Te
    Rf_p = np.zeros((len(enodes_f), 1))
    Rf_T = np.zeros((len(enodes_f), 1))
    Rf_Tadv = np.zeros((len(enodes_f), 1))

    for _e_m, enodes_m, nne_m, N_qm in _iter_fracture_matrix_matches(e, enodes_f, nodes_f, N_q, mesh):
        matrix_vectors: dict[str, npt.NDArray[np.float64]] = {}
        if pressure_coupling or advection_on:
            p_m_q = N_qm @ p[enodes_m]  # matrix pressure
            flux_p_q, _, _, _, _ = _evaluate_fracture_leakoff_flux(
                rhof_q, kf_perp, mu_q, p_f_q, p_m_q
            )  # mass flux

        # F_Pdiffusion and F_Tadvection hold inflows, so the leak-off outflow enters negatively
        if pressure_coupling:
            Rf_p = Rf_p + _vctrzd_qp_vctr_integrate(N_q, -flux_p_q, wJe_q, True)  # Fracture
            matrix_vectors["F_Pdiffusion_mm"] = _vctrzd_qp_vctr_integrate(N_qm, flux_p_q, wJe_q, True)

        if thermal_coupling:
            T_m_q = N_qm @ T[enodes_m]  # matrix temperature
            flux_hc_T_q, _, _ = _evaluate_fracture_conduction_flux(
                heat_transfer_factor, lambda_f, aperture, T_f_q, T_m_q
            )
            Rf_T = Rf_T + _vctrzd_qp_vctr_integrate(N_q, flux_hc_T_q, wJe_q, True)
            matrix_vectors["F_Tdiffusion_mm"] = _vctrzd_qp_vctr_integrate(N_qm, -flux_hc_T_q, wJe_q, True)

            if advection_on:
                T_flow_fq = N_q @ kin_n.Te_flow
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
                    T_flow_lagged=T_flow_lagged,
                )
                Rf_Tadv = Rf_Tadv + _vctrzd_qp_vctr_integrate(N_q, -flux_adv_T_q, wJe_q, True)
                matrix_vectors["F_Tadvection_mm"] = _vctrzd_qp_vctr_integrate(N_qm, flux_adv_T_q, wJe_q, True)

        matches.append((nne_m, matrix_vectors))

    if pressure_coupling:
        fracture_vectors["F_Pdiffusion"] = Rf_p
    if thermal_coupling:
        fracture_vectors["F_Tdiffusion"] = Rf_T
        if advection_on:
            fracture_vectors["F_Tadvection"] = Rf_Tadv
    return fracture_vectors, matches


def _integrate_fracture_coupling_tangent(
    e: int,
    enodes_f: list[int],
    nodes_f: npt.NDArray[np.float64],
    N_q: npt.NDArray[np.float64],
    wJe_q: npt.NDArray[np.float64],
    mesh: dict[str, Any],
    fluid: dict[str, Any],
    kin_n: ElementKinematics,  # current-iterate slice
    pe_lag: npt.NDArray[np.float64],  # fracture element lagged pressures - upwind decision only
    p: npt.NDArray[np.float64],  # Full pressure vector
    p_lag: npt.NDArray[np.float64],  # Full pressure vector, lagged - upwind decision only
    T: npt.NDArray[np.float64],  # Full temperature vector
    T_flow: npt.NDArray[np.float64],  # Temperature vector for property evaluation
    kf_perp: PropertyResult,
    mu_q: PropertyResult,  # viscosity at quad points
    rho_f: PropertyState,  # fluid density; `.q_n` is the current-iterate value used here
    heat_transfer_factor: float,  # heat transfer coefficient multiplier
    lambda_f: float,  # fluid thermal conductivity
    aperture: float,
    *,
    pressure_coupling: bool,
    thermal_coupling: bool,
    fully_coupled: bool,
    advection_on: bool,
    T_flow_lagged: bool,
) -> list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Integrate one fracture element's coupling tangent contribution to its neighboring matrix elements."""
    entries: list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]] = []

    rhof_q = rho_f.q_n
    p_f_q = N_q @ kin_n.pe  # fracture pressure
    p_f_q_lag = N_q @ pe_lag  # lagged fracture pressure, upwind decision only

    for _e_m, enodes_m, nne_m, N_qm in _iter_fracture_matrix_matches(e, enodes_f, nodes_f, N_q, mesh):
        ff_matrices: dict[str, npt.NDArray[np.float64]] = {}
        fm_matrices: dict[str, npt.NDArray[np.float64]] = {}
        mf_matrices: dict[str, npt.NDArray[np.float64]] = {}
        mm_matrices: dict[str, npt.NDArray[np.float64]] = {}

        if pressure_coupling or advection_on:
            p_m_q = N_qm @ p[enodes_m]  # matrix pressure
            flux_p_q, dfluxp_dpf, dfluxp_dpm, dfluxp_dTf, dfluxp_dTm = _evaluate_fracture_leakoff_flux(
                rhof_q, kf_perp, mu_q, p_f_q, p_m_q, want_tangent_dp=True, want_tangent_dT=thermal_coupling
            )  # Mass flux

        if pressure_coupling:
            ff_matrices["Kpp_ff"] = _vctrzd_quad_point_integration(N_q, dfluxp_dpf, N_q, wJe_q, True)
            fm_matrices["Kpp_fm"] = _vctrzd_quad_point_integration(N_qm, -dfluxp_dpf, N_q, wJe_q, True)
            mf_matrices["Kpp_mf"] = _vctrzd_quad_point_integration(N_q, dfluxp_dpm, N_qm, wJe_q, True)
            mm_matrices["Kpp_mm"] = _vctrzd_quad_point_integration(N_qm, -dfluxp_dpm, N_qm, wJe_q, True)

        if thermal_coupling:
            T_f_q = N_q @ kin_n.Te
            T_m_q = N_qm @ T[enodes_m]
            _, dfluxc_dTf, dfluxc_dTm = _evaluate_fracture_conduction_flux(
                heat_transfer_factor, lambda_f, aperture, T_f_q, T_m_q, want_tangent_dT=True
            )  # convection contribution

            RTT_ff = _vctrzd_quad_point_integration(N_q, dfluxc_dTf, N_q, wJe_q, True)
            RTT_fm = _vctrzd_quad_point_integration(N_qm, dfluxc_dTm, N_q, wJe_q, True)
            RTT_mf = _vctrzd_quad_point_integration(N_q, -dfluxc_dTf, N_qm, wJe_q, True)
            RTT_mm = _vctrzd_quad_point_integration(N_qm, -dfluxc_dTm, N_qm, wJe_q, True)

            if advection_on:
                T_flow_fq = N_q @ kin_n.Te_flow
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
                    dfluxp_dpf=dfluxp_dpf,
                    dfluxp_dpm=dfluxp_dpm,
                    want_tangent=True,
                    T_flow_lagged=T_flow_lagged,
                )

                # fm/mf coefficients use the derivative wrt the column's temperature
                RTT_ff = RTT_ff + _vctrzd_quad_point_integration(N_q, dqadv_dTf, N_q, wJe_q, True)
                RTT_fm = RTT_fm + _vctrzd_quad_point_integration(N_qm, -dqadv_dTf, N_q, wJe_q, True)
                RTT_mf = RTT_mf + _vctrzd_quad_point_integration(N_q, dqadv_dTm, N_qm, wJe_q, True)
                RTT_mm = RTT_mm + _vctrzd_quad_point_integration(N_qm, -dqadv_dTm, N_qm, wJe_q, True)

                if fully_coupled:
                    ff_matrices["KpT_ff"] = _vctrzd_quad_point_integration(N_q, dfluxp_dTf, N_q, wJe_q, True)
                    fm_matrices["KpT_fm"] = _vctrzd_quad_point_integration(
                        N_qm, -dfluxp_dTf, N_q, wJe_q, True
                    )
                    mf_matrices["KpT_mf"] = _vctrzd_quad_point_integration(N_q, dfluxp_dTm, N_qm, wJe_q, True)
                    mm_matrices["KpT_mm"] = _vctrzd_quad_point_integration(
                        N_qm, -dfluxp_dTm, N_qm, wJe_q, True
                    )

                    ff_matrices["KTp_ff"] = _vctrzd_quad_point_integration(N_q, dqadv_dpf, N_q, wJe_q, True)
                    fm_matrices["KTp_fm"] = _vctrzd_quad_point_integration(N_qm, -dqadv_dpf, N_q, wJe_q, True)
                    mf_matrices["KTp_mf"] = _vctrzd_quad_point_integration(N_q, dqadv_dpm, N_qm, wJe_q, True)
                    mm_matrices["KTp_mm"] = _vctrzd_quad_point_integration(
                        N_qm, -dqadv_dpm, N_qm, wJe_q, True
                    )

            ff_matrices["KTT_ff"] = RTT_ff
            fm_matrices["KTT_fm"] = RTT_fm
            mf_matrices["KTT_mf"] = RTT_mf
            mm_matrices["KTT_mm"] = RTT_mm
        entries.append((nne_m, ff_matrices, fm_matrices, mf_matrices, mm_matrices))

    return entries


def _integrate_well_coupling_residual(
    enodes_m: list[int],  # matrix nodes coincident with the well element nodes
    N_q: npt.NDArray[np.float64],  # well shape functions, shared by the coincident matrix nodes
    wJe_q: npt.NDArray[np.float64],
    fluid: dict[str, Any],  # fluid mat props object
    kin_n: ElementKinematics,  # current-iterate slice
    pe_lag: npt.NDArray[np.float64],  # well element lagged pressures - upwind decision only
    p: npt.NDArray[np.float64],  # Full pressure vector
    p_lag: npt.NDArray[np.float64],  # Full pressure vector, lagged - upwind decision only
    T: npt.NDArray[np.float64],  # Full temperature vector
    T_flow: npt.NDArray[np.float64],  # Temperature vector for property evaluation
    perforation_conductance: float,  # well-to-rock conductance per unit wall area (m)
    mu_q: PropertyResult,  # viscosity at quad points
    rho_f: PropertyState,  # fluid density; `.q_n` is the current-iterate value used here
    heat_transfer_factor: float,  # heat transfer coefficient multiplier
    lambda_f: float,  # fluid thermal conductivity
    radius: float,
    roughness: float,
    mu_c: PropertyResult,  # element midpoint viscosity
    Re_c: npt.NDArray[np.float64],  # element midpoint Reynolds number
    hf: NodalEnthalpyState,  # enthalpy; nodal `cp` used only when thermal_coupling
    *,
    pressure_coupling: bool,
    thermal_coupling: bool,
    advection_on: bool,
    T_flow_lagged: bool,
) -> tuple[dict[str, npt.NDArray[np.float64]], dict[str, npt.NDArray[np.float64]]]:
    """Integrate the well-matrix exchange flux into well and matrix residual vectors.

    Returns `(well_vectors, matrix_vectors)`; the exchange is integrated once per well element.
    """
    well_vectors: dict[str, npt.NDArray[np.float64]] = {}
    matrix_vectors: dict[str, npt.NDArray[np.float64]] = {}
    if not pressure_coupling and not thermal_coupling:
        return well_vectors, matrix_vectors

    # Well and coincident matrix states at the well quadrature points
    rhof_q = rho_f.q_n
    p_w_q = N_q @ kin_n.pe
    p_m_q = N_q @ p[enodes_m]

    # Leak-off mass flux
    if pressure_coupling or advection_on:
        flux_p_q, _, _, _, _ = _evaluate_well_leakoff_flux(
            rhof_q, perforation_conductance, radius, mu_q, p_w_q, p_m_q
        )

    # F_Pdiffusion and F_Tadvection hold inflows, so the leak-off outflow enters negatively
    if pressure_coupling:
        well_vectors["F_Pdiffusion"] = _vctrzd_qp_vctr_integrate(N_q, -flux_p_q, wJe_q, True)
        matrix_vectors["F_Pdiffusion_mm"] = _vctrzd_qp_vctr_integrate(N_q, flux_p_q, wJe_q, True)

    if thermal_coupling:
        # Conduction exchange
        T_w_q = N_q @ kin_n.Te
        T_m_q = N_q @ T[enodes_m]
        flux_hc_T_q, _, _, _ = _evaluate_well_conduction_flux(
            heat_transfer_factor,
            lambda_f,
            radius,
            roughness,
            Re_c,
            mu_c,
            _N_MID_L2 @ hf.e_n.cp,  # midpoint cp from the nodal values
            _N_MID_L2,
            N_q,
            T_w_q,
            T_m_q,
        )
        well_vectors["F_Tdiffusion"] = _vctrzd_qp_vctr_integrate(N_q, flux_hc_T_q, wJe_q, True)
        matrix_vectors["F_Tdiffusion_mm"] = _vctrzd_qp_vctr_integrate(N_q, -flux_hc_T_q, wJe_q, True)

        # Advective exchange, upwinded on lagged pressures
        if advection_on:
            flux_adv_T_q, _, _, _, _ = _evaluate_well_advection_flux(
                p_w_q,
                p_m_q,
                N_q @ pe_lag,
                N_q @ p_lag[enodes_m],
                T_w_q,
                T_m_q,
                N_q @ kin_n.Te_flow,
                N_q @ T_flow[enodes_m],
                flux_p_q,
                fluid,
                perforation_conductance,
                radius,
                mu_q,
                rhof_q,
                T_flow_lagged=T_flow_lagged,
            )
            well_vectors["F_Tadvection"] = _vctrzd_qp_vctr_integrate(N_q, -flux_adv_T_q, wJe_q, True)
            matrix_vectors["F_Tadvection_mm"] = _vctrzd_qp_vctr_integrate(N_q, flux_adv_T_q, wJe_q, True)

    return well_vectors, matrix_vectors


def _integrate_well_coupling_tangent(  # noqa
    enodes_m: list[int],  # matrix nodes coincident with the well element nodes
    N_q: npt.NDArray[np.float64],  # well shape functions, shared by the coincident matrix nodes
    wJe_q: npt.NDArray[np.float64],
    fluid: dict[str, Any],
    kin_n: ElementKinematics,  # current-iterate slice
    pe_lag: npt.NDArray[np.float64],  # well element lagged pressures - upwind decision only
    p: npt.NDArray[np.float64],  # Full pressure vector
    p_lag: npt.NDArray[np.float64],  # Full pressure vector, lagged - upwind decision only
    T: npt.NDArray[np.float64],  # Full temperature vector
    T_flow: npt.NDArray[np.float64],  # Temperature vector for property evaluation
    perforation_conductance: float,  # well-to-rock conductance per unit wall area (m)
    mu_q: PropertyResult,  # viscosity at quad points
    rho_f: PropertyState,  # fluid density; `.q_n` is the current-iterate value used here
    heat_transfer_factor: float,  # heat transfer coefficient multiplier
    lambda_f: float,  # fluid thermal conductivity
    radius: float,
    roughness: float,
    mu_c: PropertyResult,  # element midpoint viscosity
    Re_c: npt.NDArray[np.float64],  # element midpoint Reynolds number
    dRe_dp: npt.NDArray[
        np.float64
    ],  # element Reynolds number pressure tangent; `_EMPTY` unless fully_coupled
    dRe_dT: npt.NDArray[np.float64],  # element Reynolds number temperature tangent
    hf: NodalEnthalpyState,  # enthalpy; nodal `cp` (lagged) used only when thermal_coupling
    *,
    pressure_coupling: bool,
    thermal_coupling: bool,
    fully_coupled: bool,
    advection_on: bool,
    T_flow_lagged: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Integrate one well element's coupling tangent contribution to its coincident matrix nodes."""
    ww_matrices: dict[str, npt.NDArray[np.float64]] = {}
    wm_matrices: dict[str, npt.NDArray[np.float64]] = {}
    mw_matrices: dict[str, npt.NDArray[np.float64]] = {}
    mm_matrices: dict[str, npt.NDArray[np.float64]] = {}

    # Well and coincident matrix states at the well quadrature points
    rhof_q = rho_f.q_n
    p_w_q = N_q @ kin_n.pe
    p_m_q = N_q @ p[enodes_m]

    # Leak-off mass flux and tangents
    if pressure_coupling or advection_on:
        flux_p_q, dfluxp_dpw, dfluxp_dpm, dfluxp_dTw, dfluxp_dTm = _evaluate_well_leakoff_flux(
            rhof_q,
            perforation_conductance,
            radius,
            mu_q,
            p_w_q,
            p_m_q,
            want_tangent_dp=True,
            want_tangent_dT=thermal_coupling,
        )

    if pressure_coupling:
        ww_matrices["Kpp_ww"] = _vctrzd_quad_point_integration(N_q, dfluxp_dpw, N_q, wJe_q, True)
        wm_matrices["Kpp_wm"] = _vctrzd_quad_point_integration(N_q, -dfluxp_dpw, N_q, wJe_q, True)
        mw_matrices["Kpp_mw"] = _vctrzd_quad_point_integration(N_q, dfluxp_dpm, N_q, wJe_q, True)
        mm_matrices["Kpp_mm"] = _vctrzd_quad_point_integration(N_q, -dfluxp_dpm, N_q, wJe_q, True)

    if thermal_coupling:
        # Conduction exchange
        T_w_q = N_q @ kin_n.Te
        T_m_q = N_q @ T[enodes_m]
        _, dfluxc_dTw, dfluxc_dTm, dfluxc_dpw = _evaluate_well_conduction_flux(
            heat_transfer_factor,
            lambda_f,
            radius,
            roughness,
            Re_c,
            mu_c,
            _N_MID_L2 @ hf.e_n.cp,  # midpoint cp from the nodal values
            _N_MID_L2,
            N_q,
            T_w_q,
            T_m_q,
            dRe_dp=dRe_dp,
            dRe_dT=dRe_dT,
            want_tangent_dT=True,
            want_tangent_dp=fully_coupled,
        )
        # Tangent arrays carry the column node shape functions; well rows +flux, matrix rows -flux
        RTT_ww = _vctrzd_qp_vctr_integrate(N_q, dfluxc_dTw, wJe_q, False)
        RTT_wm = _vctrzd_qp_vctr_integrate(N_q, -dfluxc_dTw, wJe_q, False)
        RTT_mw = _vctrzd_qp_vctr_integrate(N_q, dfluxc_dTm, wJe_q, False)
        RTT_mm = _vctrzd_qp_vctr_integrate(N_q, -dfluxc_dTm, wJe_q, False)

        # Conduction depends on well pressure through h
        if fully_coupled:
            ww_matrices["KTp_ww"] = _vctrzd_qp_vctr_integrate(N_q, dfluxc_dpw, wJe_q, False)
            wm_matrices["KTp_wm"] = _vctrzd_qp_vctr_integrate(N_q, -dfluxc_dpw, wJe_q, False)

        # Advective exchange, upwinded on lagged pressures
        if advection_on:
            _, dqadv_dTw, dqadv_dTm, dqadv_dpw, dqadv_dpm = _evaluate_well_advection_flux(
                p_w_q,
                p_m_q,
                N_q @ pe_lag,
                N_q @ p_lag[enodes_m],
                T_w_q,
                T_m_q,
                N_q @ kin_n.Te_flow,
                N_q @ T_flow[enodes_m],
                flux_p_q,
                fluid,
                perforation_conductance,
                radius,
                mu_q,
                rhof_q,
                dfluxp_dpw=dfluxp_dpw,
                dfluxp_dpm=dfluxp_dpm,
                want_tangent=True,
                T_flow_lagged=T_flow_lagged,
            )
            RTT_ww = RTT_ww + _vctrzd_quad_point_integration(N_q, dqadv_dTw, N_q, wJe_q, True)
            RTT_wm = RTT_wm + _vctrzd_quad_point_integration(N_q, -dqadv_dTw, N_q, wJe_q, True)
            RTT_mw = RTT_mw + _vctrzd_quad_point_integration(N_q, dqadv_dTm, N_q, wJe_q, True)
            RTT_mm = RTT_mm + _vctrzd_quad_point_integration(N_q, -dqadv_dTm, N_q, wJe_q, True)

            # Off-diagonal pressure/temperature coupling
            if fully_coupled:
                ww_matrices["KpT_ww"] = _vctrzd_quad_point_integration(N_q, dfluxp_dTw, N_q, wJe_q, True)
                wm_matrices["KpT_wm"] = _vctrzd_quad_point_integration(N_q, -dfluxp_dTw, N_q, wJe_q, True)
                mw_matrices["KpT_mw"] = _vctrzd_quad_point_integration(N_q, dfluxp_dTm, N_q, wJe_q, True)
                mm_matrices["KpT_mm"] = _vctrzd_quad_point_integration(N_q, -dfluxp_dTm, N_q, wJe_q, True)

                ww_matrices["KTp_ww"] += _vctrzd_quad_point_integration(N_q, dqadv_dpw, N_q, wJe_q, True)
                wm_matrices["KTp_wm"] += _vctrzd_quad_point_integration(N_q, -dqadv_dpw, N_q, wJe_q, True)
                mw_matrices["KTp_mw"] = _vctrzd_quad_point_integration(N_q, dqadv_dpm, N_q, wJe_q, True)
                mm_matrices["KTp_mm"] = _vctrzd_quad_point_integration(N_q, -dqadv_dpm, N_q, wJe_q, True)

        ww_matrices["KTT_ww"] = RTT_ww
        wm_matrices["KTT_wm"] = RTT_wm
        mw_matrices["KTT_mw"] = RTT_mw
        mm_matrices["KTT_mm"] = RTT_mm

    return ww_matrices, wm_matrices, mw_matrices, mm_matrices


def integrate_residuals(  # noqa
    mesh: dict[str, Any],
    material: dict[str, Any],
    primary_variables: dict[str, Any],
    controls: dict[str, Any],
    cv: npt.NDArray[np.float64],  # rate coefficients
    T_flow: npt.NDArray[np.float64],  # temperature field used for property evaluations
    *,
    is_fracture: bool = False,
    is_well: bool = False,
    physics: Physics,  # fields solved in this call; `off` fields are not assembled
    T_flow_lagged: bool = False,
) -> dict[str, Any]:
    """Integrate residual vectors for conservation of mass and energy."""
    p = primary_variables["p"]
    T = primary_variables["T"]

    p_0 = material["p_0"]
    T_0 = material["T_0"]

    axisymmetry = bool(controls["axisymmetric_on"])
    fracture_on = bool(controls["fracture_on"])
    wells_on = bool(controls["wells_on"])
    on_submesh = is_fracture or is_well  # fracture/well pass: no solid, global DoF numbering
    domain = physics.domain(is_fracture=is_fracture, is_well=is_well)
    solve_p, solve_T, advection_on = domain.solve_flow, domain.solve_heat, domain.advection
    need_flux = solve_p or (solve_T and advection_on)
    # matrix coupling is assembled for a field if either side solves (or advects) it
    pressure_coupling = solve_p or physics.matrix.solve_flow
    thermal_coupling = solve_T or physics.matrix.solve_heat
    coupling_advection = advection_on or physics.matrix.advection
    volumetric_strain_on = bool(controls["volumetric_strain_on"])
    steady_pres = float(domain.steady_flow)
    steady_temp = float(domain.steady_heat)

    fluid = material["fluid"]
    lambda_f = fluid["therm_cond"]

    # fixed accumulator buckets; new nonlinear terms just sum onto one
    list_of_matrices = ["F_Pstorage", "F_Pdiffusion"]
    if physics.solves("heat"):
        list_of_matrices = ["F_Tstorage", "F_Tdiffusion", "F_Tadvection", "F_Tstab"] + list_of_matrices

    # ----- mesh data: bulk (matrix) mesh, or the fracture/well mesh on the recursive passes
    mesh_data = _get_region_mesh_data(mesh, is_fracture=is_fracture, is_well=is_well)

    if is_fracture:
        fmesh = mesh_data.submesh
        conn_global = fmesh["conn_global"]
        nn_m = mesh["nn"]

        # Connectivity-only fracture/matrix coupling index vectors, cached on fmesh.
        row_vec_flux_m = _get_fracture_coupling_vectors(mesh, mesh_data)["row_vec_flux_m"]

    if is_well:
        wmesh = mesh_data.submesh
        conn_global = wmesh["conn_global"]
        nn_m = mesh["nn"]

        # Connectivity-only well/matrix coupling index vectors, cached on wmesh.
        row_vec_flux_m = _get_well_coupling_vectors(mesh_data)["row_vec_flux_m"]

    if on_submesh:
        coupling_flux_names = (["F_Pdiffusion_mm"] if pressure_coupling else []) + (
            ["F_Tdiffusion_mm"] if thermal_coupling else []
        )
        if thermal_coupling and coupling_advection:
            coupling_flux_names = coupling_flux_names + ["F_Tadvection_mm"]

        vec_flux_m = {name: np.zeros(row_vec_flux_m.size) for name in coupling_flux_names}
        vec_counter_flux_m = 0

    # ----- gravity vector
    g = _get_gravity(mesh["nsd"], controls["gravity"])

    vec = _initiate_vectors(list_of_matrices, mesh_data.nne_list, mesh_data.elem_indx, ismatrix=False)
    vec_counter = 0  # vectorized storage index counter

    # ----- Nodal states, evaluated once per node rather than once per element node
    dofs = _get_pass_dofs(mesh, mesh_data.nn, is_fracture=is_fracture, is_well=is_well)
    p_nodes, T_flow_nodes = p[dofs], T_flow[dofs]
    p_0_nodes, T_0_nodes = p_0[dofs].reshape(-1, 1), T_0[dofs].reshape(-1, 1)
    nodal_fluid = _evaluate_nodal_fluid_state(
        T[dofs], T_flow_nodes, p_nodes, fluid, T_flow_lagged=T_flow_lagged, want_energy=thermal_coupling
    )

    # ----- Matrix integration element loop; skipped when nothing is solved or coupled here
    active = solve_p or solve_T or on_submesh
    for r in range(mesh_data.num_regions if active else 0):
        flow = _get_region_flow_data(mesh, material, mesh_data, r, is_fracture=is_fracture, is_well=is_well)
        # Gravity projected onto the fracture's own local (tangential) coordinate, matching
        # the reduced dimensionality of the fracture's own gradP/B_q. Wells project per element.
        g_r = (flow.u.T @ g) if is_fracture else g
        isotropic_cond, heat_transfer_factor = _get_region_thermal_data(
            material, r, is_fracture=is_fracture, is_well=is_well
        )
        solid = _get_region_solid_data(material, r, no_solid=on_submesh)
        # Porosity depends on region solid properties, so nodes shared across regions differ
        nodal_phi_L = _evaluate_nodal_porosity(
            p_nodes,
            p_0_nodes,
            T_flow_nodes,
            T_0_nodes,
            solid,
            flow.w0,
            no_solid=on_submesh,
            T_flow_lagged=T_flow_lagged,
            volumetric_strain_on=volumetric_strain_on,
        )
        nodal_state = _stack_nodal_state(nodal_phi_L, nodal_fluid)

        # find element type of first element in region
        elem_type_counter = -1  # force an update on the first element

        for e in mesh_data.region_lists[r]:
            new_type_idx = bisect.bisect_left(mesh_data.elem_indx, e)
            if new_type_idx != elem_type_counter:
                # Update for different element type
                # Evaluate quadrature points and get weights and shape functions.
                W, nq, N_q, GN_q = _get_quad_points_and_shape_functions(
                    mesh_data.nq_list[new_type_idx], mesh_data.nsd, mesh_data.etype_list[new_type_idx]
                )
                nne = mesh_data.nne_list[new_type_idx]
                elem_type_counter = new_type_idx

            enodes = mesh_data.conn[e]  # local element nodes
            enodes_dof = conn_global[e] if on_submesh else enodes  # global DoF index, for T/p
            if is_well:
                # Project onto the element tangent for 1D local coordinates and gravity
                t_e = flow.u[e].reshape(-1, 1)
                xI = flow.nodes_r[enodes, :] @ t_e
                g_r = t_e.T @ g
            else:
                xI = flow.nodes_r[enodes, :]
            B_q, detJe_q = getB(GN_q, xI, mesh_data.nsd)
            wJe_q = W * detJe_q  # Weight * det(Jacobian), used in every integrand
            if axisymmetry:
                # Add radial component to integrand for axisymmetry
                r_coord = N_q @ xI[:, 0]
                wJe_q = wJe_q * r_coord.T

            # Element vectors initialize
            element_vectors: dict[str, npt.NDArray[np.float64]] = {}

            # Evaluate primary variables at quad points and nodes
            kin = _evaluate_element_kinematics(
                N_q, B_q, xI, T, T_flow, p, T_0, p_0, enodes_dof, want_lengths=advection_on
            )
            kin_n = kin.n  # current-iterate (column 0) slice, reused throughout below

            # Evaluate dependent variables at quad points and nodes
            kf, mu_q, rho_f, phi_L, u_f, hf = _evaluate_flow_and_storage_state(
                flow.kf,
                kin_n,
                N_q,
                fluid,
                nodal_state,
                np.asarray(enodes),
                T_flow_lagged=T_flow_lagged,
                want_energy=thermal_coupling,
            )
            if is_well:
                # Element flow from the midpoint state: L2 gradients are constant and T is linear
                mu_c = get_viscosity(_N_MID_L2 @ kin_n.Te_flow, _N_MID_L2 @ kin_n.pe, fluid, T_flow_lagged)
                _, q_c, _, _, Re_c, _, _ = _evaluate_well_flow(
                    _N_MID_L2, B_q[0:1], kin.gradP[0:1], g_r, mu_c, rho_f.e_n, flow.radius, flow.roughness
                )
                q_D = np.repeat(q_c, nq, axis=0)  # q_D = mass flow rate, constant along the element
            elif need_flux:
                _, q_D, _, _ = _evaluate_darcy_flux(
                    N_q, B_q, kin.gradP, g_r, kf, mu_q, rho_f, isotropic_perm=flow.isotropic_perm
                )  # q_D = mass flux
            else:
                q_D = _EMPTY

            # Storage terms use mass lumping to prevent oscillations, so use nodal values.
            m_lump = _lumped_mass(N_q, wJe_q).reshape(-1, 1)

            # ------ Conservation of mass
            if solve_p:
                # Mass storage vector
                mass_dot_e, _, _ = _evaluate_mass_storage_integrands(phi_L, rho_f, cv, steady_pres)
                element_vectors["F_Pstorage"] = m_lump * mass_dot_e * (1 - is_well)  # wells are quasi-steady

                # Diffusion vector
                element_vectors["F_Pdiffusion"] = _vctrzd_qp_vctr_integrate(
                    B_q, q_D, wJe_q, flow.isotropic_perm
                )

            # ------ Conservation of energy
            if solve_T:
                # thermal conductivity
                lambda_e = _evaluate_effective_conductivity(
                    phi_L,
                    solid,
                    flow.w0,
                    lambda_f,
                    mesh_data.nsd,
                    no_solid=on_submesh,
                    isotropic_cond=isotropic_cond,
                )
                _, energy_dot_e, _, _ = _evaluate_energy_storage_integrands(
                    kin, phi_L, rho_f, u_f, solid, cv, steady_temp, no_solid=on_submesh
                )
                # Energy storage
                element_vectors["F_Tstorage"] = m_lump * energy_dot_e

                # Energy Diffusion vector
                element_vectors["F_Tdiffusion"] = _vctrzd_qp_vctr_integrate(
                    B_q, lambda_e * kin.gradT, wJe_q, isotropic_cond
                )

                # Advection and SUPG stabilization
                if advection_on:
                    integrand_advection, integrand_SUPG, _, _, _, _ = _evaluate_advection_terms(
                        N_q,
                        B_q,
                        mesh_data.nsd,
                        nq,
                        kin,
                        kin_n,
                        solid,
                        q_D,
                        rho_f,
                        phi_L,
                        u_f,
                        hf,
                        lambda_e,
                        cv,
                        steady_temp,
                        no_solid=on_submesh,
                        isotropic_cond=isotropic_cond,
                    )
                    # Raw avection
                    element_vectors["F_Tadvection"] = _vctrzd_qp_vctr_integrate(
                        B_q, integrand_advection, wJe_q, flow.isotropic_perm
                    )
                    # Stabilization
                    element_vectors["F_Tstab"] = _vctrzd_qp_vctr_integrate(
                        B_q, integrand_SUPG, wJe_q, flow.isotropic_perm
                    )

            # ----- Fracture-matrix leak-off/convective-exchange coupling
            if is_fracture:
                fracture_flux_vectors, matches = _integrate_fracture_coupling_residual(
                    e,
                    enodes,
                    mesh_data.submesh_nodes,
                    N_q,
                    wJe_q,
                    mesh,
                    fluid,
                    kin_n,
                    kin.pe[:, 1:2],
                    p[:, 0:1],
                    p[:, 1:2],
                    T[:, 0:1],
                    T_flow[:, 0:1],
                    flow.kf_perp,
                    mu_q,
                    rho_f,
                    heat_transfer_factor,
                    lambda_f,
                    flow.w0,
                    pressure_coupling=pressure_coupling,
                    thermal_coupling=thermal_coupling,
                    advection_on=coupling_advection,
                    T_flow_lagged=T_flow_lagged,
                )
                _accumulate_element_matrices(element_vectors, fracture_flux_vectors)

                for nne_m, matrix_vectors in matches:
                    stop = vec_counter_flux_m + nne_m  # vectorized index count end
                    _store_vctrzd_element_matrix(vec_flux_m, matrix_vectors, vec_counter_flux_m, stop)
                    vec_counter_flux_m = stop

            # ----- Well-matrix leak-off/convective-exchange coupling
            if is_well and wmesh["matrix_conn"][e] != [-1]:  # none outside the domain
                well_flux_vectors, matrix_vectors = _integrate_well_coupling_residual(
                    wmesh["matrix_node_conn"][e],
                    N_q,
                    wJe_q,
                    fluid,
                    kin_n,
                    kin.pe[:, 1:2],
                    p[:, 0:1],
                    p[:, 1:2],
                    T[:, 0:1],
                    T_flow[:, 0:1],
                    flow.perforation_conductance,
                    mu_q,
                    rho_f,
                    heat_transfer_factor,
                    lambda_f,
                    flow.radius,
                    flow.roughness,
                    mu_c,
                    Re_c,
                    hf,
                    pressure_coupling=pressure_coupling,
                    thermal_coupling=thermal_coupling,
                    advection_on=coupling_advection,
                    T_flow_lagged=T_flow_lagged,
                )
                _accumulate_element_matrices(element_vectors, well_flux_vectors)

                stop = vec_counter_flux_m + nne  # vectorized index count end
                _store_vctrzd_element_matrix(vec_flux_m, matrix_vectors, vec_counter_flux_m, stop)
                vec_counter_flux_m = stop

            # ----- Assemble element contributions into global matrices
            stop = vec_counter + nne  # vectorized index count end
            _store_vctrzd_element_matrix(vec, element_vectors, vec_counter, stop)
            vec_counter = stop

    # ----- Return and convert to sparse matrix
    integrated_vectors = _assemble_global_matrices(
        vec,
        mesh_data.conn,
        mesh_data.nn,
        mesh_data.nne_list,
        mesh_data.elem_indx,
        np.concatenate(mesh_data.region_lists),
        isvector=True,
    )

    if on_submesh:  # Fracture/well residuals to be updated
        # Leak-off contributions into matrix. Assembled in the recursion call.
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
            physics=physics,
            T_flow_lagged=T_flow_lagged,
        )
        # Fold the matrix-side "_mm" flux from the fracture pass into the matrix residual.
        for name in ("F_Pdiffusion", "F_Tdiffusion", "F_Tadvection"):
            if name + "_mm" in integrated_fracture_vectors:
                flux_mm = integrated_fracture_vectors[name + "_mm"]
                integrated_vectors[name] = integrated_vectors[name] + flux_mm
        # Combine fracture and matrix vectors into one
        integrated_vectors = _stack_fracture_vectors(integrated_vectors, integrated_fracture_vectors)

    if wells_on:
        # Recursion call
        integrated_well_vectors = integrate_residuals(
            mesh,
            material,
            primary_variables,
            controls,
            cv,
            T_flow,
            is_well=True,
            physics=physics,
            T_flow_lagged=T_flow_lagged,
        )
        # Fold the matrix-side "_mm" flux from the well pass into the matrix residual.
        for name in ("F_Pdiffusion", "F_Tdiffusion", "F_Tadvection"):
            if name + "_mm" in integrated_well_vectors:
                flux_mm = _pad_sparse(integrated_well_vectors[name + "_mm"], integrated_vectors[name].shape)
                integrated_vectors[name] = integrated_vectors[name] + flux_mm
        # Combine well and (matrix + fracture) vectors into one
        integrated_vectors = _stack_well_vectors(integrated_vectors, integrated_well_vectors)

    return integrated_vectors


def integrate_tangent_matrices(  # noqa
    mesh: dict[str, Any],
    material: dict[str, Any],
    primary_variables: dict[str, Any],
    controls: dict[str, Any],
    cv: npt.NDArray[np.float64],  # rate coefficients
    T_flow: npt.NDArray[np.float64],  # temperature field used for flow-property evaluations
    *,
    is_fracture: bool = False,
    is_well: bool = False,
    physics: Physics,  # fields solved in this call; `off` fields are not assembled
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
    wells_on = bool(controls["wells_on"])
    on_submesh = is_fracture or is_well  # fracture/well pass: no solid, global DoF numbering
    domain = physics.domain(is_fracture=is_fracture, is_well=is_well)
    solve_p, solve_T, advection_on = domain.solve_flow, domain.solve_heat, domain.advection
    coupled = solve_p and solve_T  # off-diagonal KpT/KTp blocks
    need_flux = solve_p or (solve_T and advection_on)
    # matrix coupling is assembled for a field if either side solves (or advects) it
    pressure_coupling = solve_p or physics.matrix.solve_flow
    thermal_coupling = solve_T or physics.matrix.solve_heat
    coupling_cross = pressure_coupling and thermal_coupling
    coupling_advection = advection_on or physics.matrix.advection
    volumetric_strain_on = bool(controls["volumetric_strain_on"])
    steady_pres = float(domain.steady_flow)
    steady_temp = float(domain.steady_heat)

    fluid = material["fluid"]
    lambda_f = fluid["therm_cond"]

    # fixed accumulator buckets; new nonlinear terms just sum onto one
    list_of_matrices = ["Kpp"]
    if physics.solves("heat"):
        list_of_matrices = ["KTT", "KpT", "KTp"] + list_of_matrices
    # ----- mesh data: bulk (matrix) mesh, or the fracture/well mesh on the recursive passes
    mesh_data = _get_region_mesh_data(mesh, is_fracture=is_fracture, is_well=is_well)

    if is_fracture:
        fmesh = mesh_data.submesh
        conn_global = fmesh["conn_global"]

        # Connectivity-only index vectors, cached on fmesh.
        cached_cv = _get_fracture_coupling_vectors(mesh, mesh_data)
        row_vec_ff, col_vec_ff = cached_cv["row_vec_ff"], cached_cv["col_vec_ff"]
        row_vec_fm, col_vec_fm = cached_cv["row_vec_fm"], cached_cv["col_vec_fm"]
        row_vec_mf, col_vec_mf = cached_cv["row_vec_mf"], cached_cv["col_vec_mf"]
        row_vec_mm, col_vec_mm = cached_cv["row_vec_mm"], cached_cv["col_vec_mm"]

        ktp_on = coupling_cross and coupling_advection
        coupling_ff_names = (
            (["Kpp_ff"] if pressure_coupling else [])
            + (["KTT_ff"] if thermal_coupling else [])
            + (["KTp_ff"] if ktp_on else [])
            + (["KpT_ff"] if ktp_on else [])
        )
        coupling_fm_names = (
            (["Kpp_fm"] if pressure_coupling else [])
            + (["KTT_fm"] if thermal_coupling else [])
            + (["KTp_fm"] if ktp_on else [])
            + (["KpT_fm"] if ktp_on else [])
        )
        coupling_mf_names = (
            (["Kpp_mf"] if pressure_coupling else [])
            + (["KTT_mf"] if thermal_coupling else [])
            + (["KTp_mf"] if ktp_on else [])
            + (["KpT_mf"] if ktp_on else [])
        )
        coupling_mm_names = (
            (["Kpp_mm"] if pressure_coupling else [])
            + (["KTT_mm"] if thermal_coupling else [])
            + (["KTp_mm"] if ktp_on else [])
            + (["KpT_mm"] if ktp_on else [])
        )
        vec_ff = {name: np.zeros(row_vec_ff.size) for name in coupling_ff_names}
        vec_fm = {name: np.zeros(row_vec_fm.size) for name in coupling_fm_names}
        vec_mf = {name: np.zeros(row_vec_mf.size) for name in coupling_mf_names}
        vec_mm = {name: np.zeros(row_vec_mm.size) for name in coupling_mm_names}

        vec_counter_ff = 0
        vec_counter_fm = 0
        vec_counter_mf = 0
        vec_counter_mm = 0

    if is_well:
        wmesh = mesh_data.submesh
        conn_global = wmesh["conn_global"]

        # Connectivity-only index vectors, cached on wmesh.
        cached_wv = _get_well_coupling_vectors(mesh_data)
        row_vec_ww, col_vec_ww = cached_wv["row_vec_ww"], cached_wv["col_vec_ww"]
        row_vec_wm, col_vec_wm = cached_wv["row_vec_wm"], cached_wv["col_vec_wm"]
        row_vec_mw, col_vec_mw = cached_wv["row_vec_mw"], cached_wv["col_vec_mw"]
        row_vec_mm, col_vec_mm = cached_wv["row_vec_mm"], cached_wv["col_vec_mm"]

        ktp_on = coupling_cross and coupling_advection
        well_ktp_on = coupling_cross  # conduction h depends on well pressure
        coupling_ww_names = (
            (["Kpp_ww"] if pressure_coupling else [])
            + (["KTT_ww"] if thermal_coupling else [])
            + (["KTp_ww"] if well_ktp_on else [])
            + (["KpT_ww"] if ktp_on else [])
        )
        coupling_wm_names = (
            (["Kpp_wm"] if pressure_coupling else [])
            + (["KTT_wm"] if thermal_coupling else [])
            + (["KTp_wm"] if well_ktp_on else [])
            + (["KpT_wm"] if ktp_on else [])
        )
        coupling_mw_names = (
            (["Kpp_mw"] if pressure_coupling else [])
            + (["KTT_mw"] if thermal_coupling else [])
            + (["KTp_mw"] if well_ktp_on else [])
            + (["KpT_mw"] if ktp_on else [])
        )
        coupling_mm_names = (
            (["Kpp_mm"] if pressure_coupling else [])
            + (["KTT_mm"] if thermal_coupling else [])
            + (["KTp_mm"] if well_ktp_on else [])
            + (["KpT_mm"] if ktp_on else [])
        )
        vec_ww = {name: np.zeros(row_vec_ww.size) for name in coupling_ww_names}
        vec_wm = {name: np.zeros(row_vec_wm.size) for name in coupling_wm_names}
        vec_mw = {name: np.zeros(row_vec_mw.size) for name in coupling_mw_names}
        vec_mm = {name: np.zeros(row_vec_mm.size) for name in coupling_mm_names}

        vec_counter_ww = 0
        vec_counter_wm = 0
        vec_counter_mw = 0
        vec_counter_mm = 0

    # ----- gravity vector
    g = _get_gravity(mesh["nsd"], controls["gravity"])

    vec = _initiate_vectors(list_of_matrices, mesh_data.nne_list, mesh_data.elem_indx)
    vec_counter = 0  # vectorized storage index counter

    # ----- Nodal states (current iterate), evaluated once per node rather than once per element node
    dofs = _get_pass_dofs(mesh, mesh_data.nn, is_fracture=is_fracture, is_well=is_well)
    p_nodes, T_flow_nodes = p[dofs, 0:1], T_flow[dofs, 0:1]
    p_0_nodes, T_0_nodes = p_0[dofs].reshape(-1, 1), T_0[dofs].reshape(-1, 1)
    nodal_fluid = _evaluate_nodal_fluid_state(
        T[dofs, 0:1],
        T_flow_nodes,
        p_nodes,
        fluid,
        calc_derivatives=True,
        T_flow_lagged=T_flow_lagged,
        solve_p=solve_p,
        solve_T=solve_T,
        want_energy=thermal_coupling,
    )

    # ----- Matrix integration element loop; skipped when nothing is solved or coupled here
    active = solve_p or solve_T or on_submesh
    for r in range(mesh_data.num_regions if active else 0):
        flow = _get_region_flow_data(mesh, material, mesh_data, r, is_fracture=is_fracture, is_well=is_well)
        # Gravity projected onto the fracture's own local (tangential) coordinate, matching
        # the reduced dimensionality of the fracture's own gradP/B_q. Wells project per element.
        g_r = (flow.u.T @ g) if is_fracture else g
        isotropic_cond, heat_transfer_factor = _get_region_thermal_data(
            material, r, is_fracture=is_fracture, is_well=is_well
        )
        solid = _get_region_solid_data(material, r, no_solid=on_submesh)
        # Porosity depends on region solid properties, so nodes shared across regions differ
        nodal_phi_L = _evaluate_nodal_porosity(
            p_nodes,
            p_0_nodes,
            T_flow_nodes,
            T_0_nodes,
            solid,
            flow.w0,
            no_solid=on_submesh,
            T_flow_lagged=T_flow_lagged,
            volumetric_strain_on=volumetric_strain_on,
        )
        nodal_state = _stack_nodal_state(nodal_phi_L, nodal_fluid)

        # find element type of first element in region
        elem_type_counter = -1  # force an update on the first element

        for e in mesh_data.region_lists[r]:
            new_type_idx = bisect.bisect_left(mesh_data.elem_indx, e)
            if new_type_idx != elem_type_counter:
                # Update for different element type
                # Evaluate quadrature points and get weights and shape functions.
                W, nq, N_q, GN_q = _get_quad_points_and_shape_functions(
                    mesh_data.nq_list[new_type_idx], mesh_data.nsd, mesh_data.etype_list[new_type_idx]
                )
                nne = mesh_data.nne_list[new_type_idx]
                elem_type_counter = new_type_idx

            enodes = mesh_data.conn[e]  # local element nodes
            enodes_dof = conn_global[e] if on_submesh else enodes  # global DoF index, for T/p
            if is_well:
                # Project onto the element tangent for 1D local coordinates and gravity
                t_e = flow.u[e].reshape(-1, 1)
                xI = flow.nodes_r[enodes, :] @ t_e
                g_r = t_e.T @ g
            else:
                xI = flow.nodes_r[enodes, :]
            B_q, detJe_q = getB(GN_q, xI, mesh_data.nsd)
            wJe_q = W * detJe_q  # Weight * det(Jacobian), used in every integrand
            if axisymmetry:
                # Add radial component to integrand for axisymmetry
                r_coord = N_q @ xI[:, 0]
                wJe_q = wJe_q * r_coord.T

            # Element matrices initialize
            element_matrices: dict[str, npt.NDArray[np.float64]] = {}

            # Evaluate primary variables at quad points and nodes
            kin = _evaluate_element_kinematics(
                N_q, B_q, xI, T, T_flow, p, T_0, p_0, enodes_dof, want_lengths=advection_on
            )
            kin_n = kin.n  # current-iterate (column 0) slice, reused throughout below

            # Evaluate dependent variables at quad points and nodes
            kf, mu_q, rho_f, phi_L, u_f, hf = _evaluate_flow_and_storage_state(
                flow.kf,
                kin_n,
                N_q,
                fluid,
                nodal_state,
                np.asarray(enodes),
                T_flow_lagged=T_flow_lagged,
                calc_derivatives=True,
                solve_p=solve_p,
                solve_T=solve_T,
                want_energy=thermal_coupling,
            )
            if is_well:
                # Element flow from the midpoint state: L2 gradients are constant and T is linear.
                # dT tangents whenever temperature is solved: conduction h depends on Re.
                mu_c = get_viscosity(
                    _N_MID_L2 @ kin_n.Te_flow,
                    _N_MID_L2 @ kin_n.pe,
                    fluid,
                    T_flow_lagged,
                    want_dp=solve_p,
                    want_dT=solve_T,
                )
                _, q_c, dqD_dp_c, dqD_dT_c, Re_c, dRe_dp, dRe_dT = _evaluate_well_flow(
                    _N_MID_L2,
                    B_q[0:1],
                    kin.gradP[0:1],
                    g_r,
                    mu_c,
                    rho_f.e_n,
                    flow.radius,
                    flow.roughness,
                    want_tangent_dp=solve_p,
                    want_tangent_dT=solve_T,
                )
                # Broadcast the element flow and its tangents to the quad points
                q_D = np.repeat(q_c, nq, axis=0)
                dqD_dp = np.repeat(dqD_dp_c, nq, axis=0) if solve_p else _EMPTY
                dqD_dT = np.repeat(dqD_dT_c, nq, axis=0) if solve_T else _EMPTY
            elif need_flux:
                _, q_D, dqD_dp, dqD_dT = _evaluate_darcy_flux(
                    N_q,
                    B_q,
                    kin.gradP,
                    g_r,
                    kf,
                    mu_q,
                    rho_f,
                    isotropic_perm=flow.isotropic_perm,
                    want_tangent_dp=solve_p,
                    want_tangent_dT=solve_T and (advection_on or solve_p),
                )
            else:
                q_D = dqD_dp = dqD_dT = _EMPTY

            # Storage terms use mass lumping to prevent oscillations, so use nodal values.
            phi_L_n = phi_L.q_n
            m_lump = _lumped_mass(N_q, wJe_q)

            _, integrand_dFsdp_e, integrand_dFsdT_e = _evaluate_mass_storage_integrands(
                phi_L,
                rho_f,
                cv,
                steady_pres,
                want_tangent_dp=solve_p,
                want_tangent_dT=coupled,
            )

            # ------------------------------------ Conservation of mass wrt pressure
            if solve_p:
                # Storage vector - Fs (lumped: nodal dFs/dp times the lumped mass weight)
                element_matrices["Kpp"] = _diag_matrix(m_lump * integrand_dFsdp_e.ravel()) * (1 - is_well)

                # Diffusion vector - Fk
                # dFk/dp
                # dqD_dp already contains the RHS matrices, integrate like a vector
                element_matrices["Kpp"] += _vctrzd_qp_vctr_integrate(B_q, dqD_dp, wJe_q, False)

            if solve_T:
                # -------------------------------- Conservation of energy wrt temperature
                lambda_e = _evaluate_effective_conductivity(
                    phi_L,
                    solid,
                    flow.w0,
                    lambda_f,
                    mesh_data.nsd,
                    no_solid=on_submesh,
                    isotropic_cond=isotropic_cond,
                )

                # Energy storage vector - FC (lumped: nodal dFC/dT times the lumped mass weight)
                _, _, integrand_dFCdT_e, integrand_dFCdp_e = _evaluate_energy_storage_integrands(
                    kin,
                    phi_L,
                    rho_f,
                    u_f,
                    solid,
                    cv,
                    steady_temp,
                    no_solid=on_submesh,
                    kin_n=kin_n,
                    want_tangent_dT=True,
                    want_tangent_dp=coupled,
                )
                element_matrices["KTT"] = _diag_matrix(m_lump * integrand_dFCdT_e.ravel())

                # Themal diffusion vector
                integrand_dFlambdadT = (
                    lambda_e * B_q + phi_L_n.dT * (lambda_f - solid.lambda_s) * kin.gradT * N_q
                )
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
                        N_q,
                        B_q,
                        mesh_data.nsd,
                        nq,
                        kin,
                        kin_n,
                        solid,
                        q_D,
                        rho_f,
                        phi_L,
                        u_f,
                        hf,
                        lambda_e,
                        cv,
                        steady_temp,
                        no_solid=on_submesh,
                        isotropic_cond=isotropic_cond,
                        dqD_dT=dqD_dT,
                        dqD_dp=dqD_dp,
                        want_tangent_dT=True,
                        want_tangent_dp=coupled,
                    )
                    element_matrices["KTT"] -= _vctrzd_qp_vctr_integrate(
                        B_q, integrand_advection_dT, wJe_q, False
                    )
                    element_matrices["KTT"] += _vctrzd_qp_vctr_integrate(B_q, integrand_SUPG_dT, wJe_q, False)

            # Fully coupled - off diagonal matrices
            if coupled:
                # ----------- Conservation of mass wrt temperature
                # Storage vector - Fs
                element_matrices["KpT"] = _diag_matrix(m_lump * integrand_dFsdT_e.ravel())

                # Diffusion vector - Fk
                integrand_dFkdT = dqD_dT
                element_matrices["KpT"] += _vctrzd_qp_vctr_integrate(B_q, integrand_dFkdT, wJe_q, False)

                # ----------- Conservation of energy wrt pressure
                # Energy storage vector - FC
                element_matrices["KTp"] = _diag_matrix(m_lump * integrand_dFCdp_e.ravel())

                # Themal diffusion vector
                integrand_dFlambdadT = phi_L_n.dp * (lambda_f - solid.lambda_s) * kin.gradT * N_q
                element_matrices["KTp"] += _vctrzd_qp_vctr_integrate(B_q, integrand_dFlambdadT, wJe_q, False)

                # Advection and stabilization
                if advection_on:
                    element_matrices["KTp"] -= _vctrzd_qp_vctr_integrate(
                        B_q, integrand_advection_dp, wJe_q, False
                    )
                    element_matrices["KTp"] += _vctrzd_qp_vctr_integrate(B_q, integrand_SUPG_dp, wJe_q, False)

            # ----- Fracture-matrix leak-off/convective-exchange coupling
            if is_fracture:
                coupling_entries = _integrate_fracture_coupling_tangent(
                    e,
                    enodes,
                    mesh_data.submesh_nodes,
                    N_q,
                    wJe_q,
                    mesh,
                    fluid,
                    kin_n,
                    kin.pe[:, 1:2],
                    p[:, 0:1],
                    p[:, 1:2],
                    T[:, 0:1],
                    T_flow[:, 0:1],
                    flow.kf_perp,
                    mu_q,
                    rho_f,
                    heat_transfer_factor,
                    lambda_f,
                    flow.w0,
                    pressure_coupling=pressure_coupling,
                    thermal_coupling=thermal_coupling,
                    fully_coupled=coupling_cross,
                    advection_on=coupling_advection,
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

            # ----- Well-matrix leak-off/convective-exchange coupling
            if is_well and wmesh["matrix_conn"][e] != [-1]:  # none outside the domain
                ww_matrices, wm_matrices, mw_matrices, mm_matrices = _integrate_well_coupling_tangent(
                    wmesh["matrix_node_conn"][e],
                    N_q,
                    wJe_q,
                    fluid,
                    kin_n,
                    kin.pe[:, 1:2],
                    p[:, 0:1],
                    p[:, 1:2],
                    T[:, 0:1],
                    T_flow[:, 0:1],
                    flow.perforation_conductance,
                    mu_q,
                    rho_f,
                    heat_transfer_factor,
                    lambda_f,
                    flow.radius,
                    flow.roughness,
                    mu_c,
                    Re_c,
                    dRe_dp,
                    dRe_dT,
                    hf,
                    pressure_coupling=pressure_coupling,
                    thermal_coupling=thermal_coupling,
                    fully_coupled=coupling_cross,
                    advection_on=coupling_advection,
                    T_flow_lagged=T_flow_lagged,
                )
                # Direct node coupling: every block is nne x nne
                stop = vec_counter_ww + nne * nne  # vectorized index count end
                _store_vctrzd_element_matrix(vec_ww, ww_matrices, vec_counter_ww, stop)
                vec_counter_ww = stop
                stop = vec_counter_wm + nne * nne  # vectorized index count end
                _store_vctrzd_element_matrix(vec_wm, wm_matrices, vec_counter_wm, stop)
                vec_counter_wm = stop
                stop = vec_counter_mw + nne * nne  # vectorized index count end
                _store_vctrzd_element_matrix(vec_mw, mw_matrices, vec_counter_mw, stop)
                vec_counter_mw = stop
                stop = vec_counter_mm + nne * nne  # vectorized index count end
                _store_vctrzd_element_matrix(vec_mm, mm_matrices, vec_counter_mm, stop)
                vec_counter_mm = stop

            # ----- Assemble element contributions into global matrices
            stop = vec_counter + nne * nne  # vectorized index count end
            _store_vctrzd_element_matrix(vec, element_matrices, vec_counter, stop)
            vec_counter = stop

    # ----- Return and convert to sparse matrix
    integrated_matrices = _assemble_global_matrices(
        vec,
        mesh_data.conn,
        mesh_data.nn,
        mesh_data.nne_list,
        mesh_data.elem_indx,
        np.concatenate(mesh_data.region_lists),
    )

    if is_fracture:
        # Assembly of coupling vectors in recursion call
        nn, nn_m = mesh_data.nn, mesh["nn"]
        for name in vec_ff:
            integrated_matrices[name] = _make_sparse(vec_ff[name], row_vec_ff, col_vec_ff, nn, nn)
        for name in vec_fm:
            integrated_matrices[name] = _make_sparse(vec_fm[name], row_vec_fm, col_vec_fm, nn_m, nn)
        for name in vec_mf:
            integrated_matrices[name] = _make_sparse(vec_mf[name], row_vec_mf, col_vec_mf, nn, nn_m)
        for name in vec_mm:
            integrated_matrices[name] = _make_sparse(vec_mm[name], row_vec_mm, col_vec_mm, nn_m, nn_m)
        return integrated_matrices

    if is_well:
        # Assembly of coupling vectors in recursion call
        nn, nn_m = mesh_data.nn, mesh["nn"]
        for name in vec_ww:
            integrated_matrices[name] = _make_sparse(vec_ww[name], row_vec_ww, col_vec_ww, nn, nn)
        for name in vec_wm:
            integrated_matrices[name] = _make_sparse(vec_wm[name], row_vec_wm, col_vec_wm, nn_m, nn)
        for name in vec_mw:
            integrated_matrices[name] = _make_sparse(vec_mw[name], row_vec_mw, col_vec_mw, nn, nn_m)
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
            physics=physics,
            T_flow_lagged=T_flow_lagged,
        )
        # Combine fracture and matrix matrices into one
        integrated_matrices = _stack_fracture_matrices(
            integrated_matrices, integrated_fracture_matrices, mesh_data.nn, mesh["fracture_mesh"]["nn"]
        )

    if wells_on:
        # Recursion call
        integrated_well_matrices = integrate_tangent_matrices(
            mesh,
            material,
            primary_variables,
            controls,
            cv,
            T_flow,
            is_well=True,
            physics=physics,
            T_flow_lagged=T_flow_lagged,
        )
        # Combine well and (matrix + fracture) matrices into one
        nn_f = mesh["fracture_mesh"]["nn"] if fracture_on else 0
        integrated_matrices = _stack_well_matrices(
            integrated_matrices, integrated_well_matrices, mesh_data.nn, nn_f, mesh["well_mesh"]["nn"]
        )

    return integrated_matrices


def integrate_enthlpy_bndry_residual(
    mesh: dict[str, Any],
    material: dict[str, Any],
    primary_variables: dict[str, Any],
    controls: dict[str, Any],
    T_flow: npt.NDArray[np.float64],  # temperature field used for flow-property evaluations
    BC_config: list[dict[str, Any]],
    *,
    T_flow_lagged: bool = False,
) -> dict[str, Any]:
    """Integrate the advective enthalpy flux boundary vector.

    Only applies to fluid BC without an associated temperature BC.
    We use the existing assembled Jacobians and vectors rather than
    re-performing the matrix integration.
    """
    advection_nodes = controls["physics"].advection_node_mask(mesh)
    nn_total = mesh["nn_total"]

    if not advection_nodes.any():
        return {"F_enth_bnd": csc_matrix((nn_total, 1), dtype=float)}

    F_enth_bnd = np.zeros((nn_total, 1))
    fluid = material["fluid"]
    T = primary_variables["T"][:, 0:1]
    p = primary_variables["p"][:, 0:1]
    q = primary_variables["q"][:, 0:1]
    node_value: dict[int, float] = {}  # keyed by node so shared nodes aren't double-counted

    for group in BC_config:
        if "hydraulics" in group and "thermal" not in group:
            nodes_i = np.asarray(group["nodes"], dtype=np.int32)
            nodes_i = nodes_i[advection_nodes[nodes_i]]  # no enthalpy outflow where advection is off
            if nodes_i.size == 0:
                continue
            q_i = q[nodes_i, 0]
            T_i = T[nodes_i, 0]
            T_flow_i = T_flow[nodes_i, 0]
            p_i = p[nodes_i, 0]
            _, _, hf_result = get_EOS(
                T_i, T_flow_i, p_i, fluid, T_flow_lagged=T_flow_lagged, want_enthalpy=True
            )
            hf_i = hf_result.value
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

    Only applies to fluid BC without an associated temperature BC.
    d(F_enth_bnd)/dT is q*dhf_dT (diagonal) plus hf_i*dq/dT;

    the latter is nonzero only for a Dirichlet-pressure node under
    `fully_coupled`, via the Kpp/KpT cross-terms.
    """
    advection_nodes = controls["physics"].advection_node_mask(mesh)
    nn_total = mesh["nn_total"]

    if not advection_nodes.any():
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
        if "hydraulics" in group and "thermal" not in group:
            nodes_i = np.asarray(group["nodes"], dtype=np.int32)
            nodes_i = nodes_i[advection_nodes[nodes_i]]  # no enthalpy outflow where advection is off
            if nodes_i.size == 0:
                continue
            q_i = q[nodes_i, 0]
            T_i = T[nodes_i, 0]
            T_flow_i = T_flow[nodes_i, 0]
            p_i = p[nodes_i, 0]
            _, _, hf_result = get_EOS(
                T_i, T_flow_i, p_i, fluid, T_flow_lagged=T_flow_lagged, want_enthalpy=True, want_dT=True
            )
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
