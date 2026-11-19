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

"""Linear (thermo-poro)elastic force balance of the matrix, with fracture face loads, contact, and cohesion.

Voigt order: 2D [xx, yy, xy]; axisymmetric [rr, zz, rz, tt]; 3D [xx, yy, zz, xz, yz, xy].
Shear strains are engineering strains; shear stresses are tensor components.

The fracture faces are the split matrix nodes on either side of each fracture element. The jump
[[u]] = u+ - u- is taken between the positive and negative matrix elements of each fracture
element (`fracture_mesh["matrix_conn"]`), with the unit normal pointing from the negative to the
positive side, so g_n = n . [[u]] is the opening.
"""

import bisect
from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix, csc_matrix, diags
from scipy.sparse.linalg import splu

from .dofmap import get_scatter
from .matrixassembly import (
    RegionMeshData,
    _assemble_global_matrices,
    _get_quad_points_and_shape_functions,
    _get_region_mesh_data,
    _get_region_value,
    _iter_fracture_matrix_matches,
    _lumped_mass,
    _pad_sparse,
    _vctrzd_quad_point_integration,
)
from .mechanicsdata import CohesiveParameters, FractureAperture, FractureInterface, MechanicsMatrices
from .mechconstitutive import (
    bandis_barton_aperture,
    cohesive_scales,
    cohesive_traction,
    elasticity_matrix,
    penalty_contact,
)
from .shapefunctions import GNMATRIX_FUNCS, NMATRIX_FUNCS, getB

# Stress component names in Voigt order, for output
STRESS_NAMES = {
    (2, False): ("xx", "yy", "xy"),
    (2, True): ("rr", "zz", "rz", "tt"),
    (3, False): ("xx", "yy", "zz", "xz", "yz", "xy"),
}


def identity_voigt(nsd: int, axisymmetric: bool) -> npt.NDArray[np.float64]:
    """Voigt form of the identity tensor: 1 on normal components, 0 on shear."""
    if nsd == 3:
        return np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    if axisymmetric:
        return np.array([1.0, 1.0, 0.0, 1.0])
    return np.array([1.0, 1.0, 0.0])


def _voigt_traction(sigma: npt.NDArray[np.float64], n: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Traction sigma . n from a Voigt stress (tensor shear components), in the dimension of n."""
    if n.size == 2:  # [xx, yy, xy], or axisymmetric [rr, zz, rz, tt]
        return np.array([sigma[0] * n[0] + sigma[2] * n[1], sigma[2] * n[0] + sigma[1] * n[1]])
    # [xx, yy, zz, xz, yz, xy]
    tensor = np.array(
        [
            [sigma[0], sigma[5], sigma[3]],
            [sigma[5], sigma[1], sigma[4]],
            [sigma[3], sigma[4], sigma[2]],
        ]
    )
    result: npt.NDArray[np.float64] = tensor @ n
    return result


def strain_displacement(
    GN_q: npt.NDArray[np.float64],
    N_q: npt.NDArray[np.float64],
    xI: npt.NDArray[np.float64],
    nsd: int,
    axisymmetric: bool,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Strain-displacement matrix B_u at quadrature points and the integration weight factor det(J) (* r)."""
    B_u, detJe_q = getB(GN_q, xI, nsd, nsd)
    if not axisymmetric:
        return B_u, detJe_q

    # hoop strain u_r / r appended as the last Voigt row; radius weights the integral
    N = N_q.reshape(N_q.shape[0], -1)
    r = N @ xI[:, 0]
    hoop = np.zeros((B_u.shape[0], 1, B_u.shape[2]))
    hoop[:, 0, 0::nsd] = N / r[:, None]
    return np.concatenate((B_u, hoop), axis=1), detJe_q * r


# Parent coordinates of the element nodes, in node order
_ELEMENT_NODE_COORDS = {
    "T3": [[0, 0], [1, 0], [0, 1]],
    "Q4": [[-1, -1], [1, -1], [1, 1], [-1, 1]],
    "T4": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "B8": [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
}


def _element_node_shape_functions(etype: str) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Shape functions and parent gradients at the element nodes, shaped like quadrature-point arrays."""
    coords = np.asarray(_ELEMENT_NODE_COORDS[etype], dtype=np.float64)
    N = np.array([NMATRIX_FUNCS[etype](xi) for xi in coords])
    GN = np.array([GNMATRIX_FUNCS[etype](xi) for xi in coords])
    return N, GN


def _region_elastic_data(material: dict[str, Any], r: int) -> tuple[float, float, float, float]:
    """E, nu, Biot coefficient and drained thermal stress coefficient K_d * alpha_s of a region."""
    solid = material["solid"]
    E = solid["youngs_mod"][r]
    nu = _get_region_value(solid["poisson_ratio"], r)
    biot = _get_region_value(solid["biot_coeff"], r)
    thermal = solid["drained_bulk_mod"][r] * _get_region_value(solid["therm_expansion"], r)
    return E, nu, biot, thermal


def _nodal_strain_operator(
    mesh_data: RegionMeshData,
    r: int,
    strain_entries: list[npt.NDArray[np.float64]],
    lumped_mass: npt.NDArray[np.float64],
) -> csc_matrix:
    """Lumped L2 projection of the volumetric strain in one region: M_lump^-1 int N m^T B, (nn, nsd*nn)."""
    integral = _assemble_global_matrices(
        {"G": np.concatenate(strain_entries)},
        mesh_data.conn,
        mesh_data.nn,
        mesh_data.nne_list,
        mesh_data.elem_indx,
        np.asarray(mesh_data.region_lists[r], dtype=np.int32),
        col_ndof=mesh_data.nsd,
    )["G"]
    # nodes outside the region have no mass and no strain
    inverse_mass = np.divide(1.0, lumped_mass, out=np.zeros_like(lumped_mass), where=lumped_mass > 0.0)
    return csc_matrix(diags(inverse_mass) @ integral)


def integrate_mechanics_matrices(
    mesh: dict[str, Any], material: dict[str, Any], controls: dict[str, Any]
) -> MechanicsMatrices:
    """Integrate the constant force-balance matrices and the nodal strain operators over the matrix."""
    mesh_data = _get_region_mesh_data(mesh, is_fracture=False, is_well=False)
    nsd = mesh_data.nsd
    axisymmetric = bool(controls["axisymmetric_on"])
    plane = controls["mechanics_controls"]["plane"]
    m = identity_voigt(nsd, axisymmetric)

    # element matrices, raveled in element order
    entries: dict[str, list[npt.NDArray[np.float64]]] = {"K_uu": [], "K_up": [], "K_uT": []}
    nodal_strain: list[csc_matrix] = []

    for r in range(mesh_data.num_regions):
        E, nu, biot, thermal = _region_elastic_data(material, r)
        D = elasticity_matrix(E, nu, nsd, plane, axisymmetric)
        strain_entries: list[npt.NDArray[np.float64]] = []
        lumped_mass = np.zeros(mesh_data.nn)

        elem_type_counter = -1  # force an update on the first element
        for e in mesh_data.region_lists[r]:
            # quadrature and shape functions per element type
            new_type_idx = bisect.bisect_left(mesh_data.elem_indx, e)
            if new_type_idx != elem_type_counter:
                W, _, N_q, GN_q = _get_quad_points_and_shape_functions(
                    mesh_data.nq_list[new_type_idx], nsd, mesh_data.etype_list[new_type_idx]
                )
                elem_type_counter = new_type_idx

            xI = mesh["nodes"][mesh_data.conn[e], :]
            B_u, detJe_q = strain_displacement(GN_q, N_q, xI, nsd, axisymmetric)
            wJ = W * detJe_q  # weight * det(J) (* r)

            # K_uu = int B^T D B
            entries["K_uu"].append(_vctrzd_quad_point_integration(B_u, D, B_u, wJ, False).ravel())
            # K_up, K_uT = -int B^T m N (scaled by Biot and K_d alpha_s)
            BtmN = _vctrzd_quad_point_integration(B_u, m.reshape(-1, 1), N_q, wJ, False)
            entries["K_up"].append((-biot * BtmN).ravel())
            entries["K_uT"].append((-thermal * BtmN).ravel())
            # int N m^T B and int N, for the region's nodal strain
            strain_entries.append(BtmN.T.ravel())
            np.add.at(lumped_mass, mesh_data.conn[e], _lumped_mass(N_q, wJ))

        nodal_strain.append(_nodal_strain_operator(mesh_data, r, strain_entries, lumped_mass))

    # assemble over node-interleaved displacement DoFs
    elem_order = np.concatenate(mesh_data.region_lists).astype(np.int32)
    args = (mesh_data.conn, mesh_data.nn, mesh_data.nne_list, mesh_data.elem_indx, elem_order)
    K_uu = _assemble_global_matrices(
        {"K_uu": np.concatenate(entries["K_uu"])}, *args, row_ndof=nsd, col_ndof=nsd
    )["K_uu"]
    coupling = _assemble_global_matrices(
        {name: np.concatenate(entries[name]) for name in ("K_up", "K_uT")}, *args, row_ndof=nsd, col_ndof=1
    )

    # p and T live on all nodes; pad the coupling columns to nn_total
    shape = (nsd * mesh_data.nn, mesh["nn_total"])
    K_up = _pad_sparse(coupling["K_up"], shape)

    # fracture faces: fracture pressure in the fracture columns of K_up, and the in-situ stress face load
    F_fracture_0 = np.zeros((nsd * mesh_data.nn, 1))
    interface = FractureInterface.empty()
    if controls["fracture_on"]:
        interface, K_upf, F_fracture_0 = integrate_fracture_interface(mesh, material, controls)
        K_up = csc_matrix(K_up + K_upf)

    return MechanicsMatrices(
        K_uu, K_up, _pad_sparse(coupling["K_uT"], shape), tuple(nodal_strain), F_fracture_0, interface
    )


def internal_force(
    matrices: MechanicsMatrices, primary_variables: dict[str, Any], material: dict[str, Any], nn: int
) -> npt.NDArray[np.float64]:
    """Internal force K_uu u + K_up (p - p_0) + K_uT (T - T_0) - F_fracture_0 at the current iterate.

    The matrix columns of K_up act on the change from p_0 and the fracture columns on the total
    fracture pressure, which loads the faces. Fracture contact and cohesion are added separately
    (`fracture_interface_force`).
    """
    u = primary_variables["u"][:, 0:1]
    dp = primary_variables["p"][:, 0:1] - material["p_0"].reshape(-1, 1)
    dp[nn:] = primary_variables["p"][nn:, 0:1]
    dT = primary_variables["T"][:, 0:1] - material["T_0"].reshape(-1, 1)
    force: npt.NDArray[np.float64] = (
        matrices.K_uu @ u + matrices.K_up @ dp + matrices.K_uT @ dT - matrices.F_fracture_0
    )
    return force


def insitu_stress(
    config_stress: dict[str, Any], nodes: npt.NDArray[np.float64], nsd: int, axisymmetric: bool
) -> npt.NDArray[np.float64]:
    """Nodal in-situ total stress (nn, nv) in Voigt order; zero when not given.

    Each component is `values` + `gradient` * distance along `grad_direction` from the shallowest node.
    """
    names = STRESS_NAMES[(nsd, axisymmetric and nsd == 2)]
    # axisymmetric components map to the Cartesian input keys (r -> x, z -> y, t -> z)
    keys = {"rr": "xx", "zz": "yy", "rz": "xy", "tt": "zz"} if axisymmetric and nsd == 2 else {}
    values = config_stress.get("values", {})
    gradient = config_stress.get("gradient", {})

    # distance along the gradient direction, vertically downward by default
    default_direction = [0.0, -1.0] if nsd == 2 else [0.0, 0.0, -1.0]
    direction = np.asarray(config_stress.get("grad_direction", default_direction), dtype=float)[:nsd]
    s = np.zeros(len(nodes))
    if np.linalg.norm(direction) > 0:
        s = nodes @ (direction / np.linalg.norm(direction))
        s = s - s.min()

    sigma_0 = np.zeros((len(nodes), len(names)))
    for i, name in enumerate(names):
        key = "S" + keys.get(name, name)
        sigma_0[:, i] = values.get(key, 0.0) + gradient.get(key, 0.0) * s
    return sigma_0


def element_volumetric_strain(
    mesh: dict[str, Any],
    material: dict[str, Any],
    controls: dict[str, Any],
    primary_variables: dict[str, Any],
) -> npt.NDArray[np.float64]:
    """Element-mean volumetric strain (ne,) of the current iterate, from the reference displacement u_0."""
    mesh_data = _get_region_mesh_data(mesh, is_fracture=False, is_well=False)
    nsd = mesh_data.nsd
    axisymmetric = bool(controls["axisymmetric_on"])
    m = identity_voigt(nsd, axisymmetric)
    u = primary_variables["u"][:, 0] - material["u_0"]

    strain = np.zeros(mesh["ne"])
    elem_type_counter = -1  # force an update on the first element
    for e, enodes in enumerate(mesh_data.conn):
        new_type_idx = bisect.bisect_left(mesh_data.elem_indx, e)
        if new_type_idx != elem_type_counter:
            _, _, N_q, GN_q = _get_quad_points_and_shape_functions(
                mesh_data.nq_list[new_type_idx], nsd, mesh_data.etype_list[new_type_idx]
            )
            elem_type_counter = new_type_idx
        B_u, _ = strain_displacement(GN_q, N_q, mesh["nodes"][enodes, :], nsd, axisymmetric)
        strain[e] = np.mean((B_u @ u[get_scatter(enodes, nsd)]) @ m)
    return strain


def _element_stresses(
    mesh: dict[str, Any],
    material: dict[str, Any],
    controls: dict[str, Any],
    primary_variables: dict[str, Any],
    *,
    at_nodes: bool,
) -> Iterator[
    tuple[int, list[int], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
]:
    """Stress change of each matrix element at its quadrature points, or at its nodes when `at_nodes`.

    Yields (element, element nodes, N, stress (npts, nv), weight * det(J) (* r)); the weights are
    those of the quadrature rule and are only meaningful at quadrature points.
    """
    mesh_data = _get_region_mesh_data(mesh, is_fracture=False, is_well=False)
    nsd = mesh_data.nsd
    axisymmetric = bool(controls["axisymmetric_on"])
    plane = controls["mechanics_controls"]["plane"]
    m = identity_voigt(nsd, axisymmetric)
    nn = mesh_data.nn

    # change of the current iterate from the reference state, on matrix nodes
    u = primary_variables["u"][:, 0] - material["u_0"]
    dp = primary_variables["p"][:nn, 0] - material["p_0"][:nn]
    dT = primary_variables["T"][:nn, 0] - material["T_0"][:nn]

    for r in range(mesh_data.num_regions):
        E, nu, biot, thermal = _region_elastic_data(material, r)
        D = elasticity_matrix(E, nu, nsd, plane, axisymmetric)

        elem_type_counter = -1  # force an update on the first element
        for e in mesh_data.region_lists[r]:
            new_type_idx = bisect.bisect_left(mesh_data.elem_indx, e)
            if new_type_idx != elem_type_counter:
                etype = mesh_data.etype_list[new_type_idx]
                if at_nodes:
                    N_q, GN_q = _element_node_shape_functions(etype)
                    W = np.zeros(len(N_q))
                else:
                    W, _, N_q, GN_q = _get_quad_points_and_shape_functions(
                        mesh_data.nq_list[new_type_idx], nsd, etype
                    )
                N = N_q.reshape(len(N_q), -1)
                elem_type_counter = new_type_idx

            enodes = mesh_data.conn[e]
            xI = mesh["nodes"][enodes, :]
            with np.errstate(divide="ignore", invalid="ignore"):  # hoop strain at nodes on the axis
                B_u, detJe = strain_displacement(GN_q, N_q, xI, nsd, axisymmetric)
                strain = B_u @ u[get_scatter(enodes, nsd)]
            if axisymmetric and at_nodes:
                # on the axis the hoop strain u_r / r tends to the radial strain
                on_axis = np.abs(xI[:, 0]) < 1e-12
                strain[on_axis, -1] = strain[on_axis, 0]

            # D eps - Biot dp m - K_d alpha_s dT m
            stress = strain @ D.T - np.outer(biot * (N @ dp[enodes]) + thermal * (N @ dT[enodes]), m)
            yield e, enodes, N, stress, W * detJe


def element_stress(
    mesh: dict[str, Any],
    material: dict[str, Any],
    controls: dict[str, Any],
    primary_variables: dict[str, Any],
) -> npt.NDArray[np.float64]:
    """Element total stress (ne, nv): the mean of the quadrature-point stresses, plus the in-situ stress."""
    sigma_0 = material["sigma_0"]
    stress = np.zeros((mesh["ne"], sigma_0.shape[1]))
    for e, enodes, _, stress_q, _ in _element_stresses(
        mesh, material, controls, primary_variables, at_nodes=False
    ):
        stress[e] = sigma_0[enodes].mean(axis=0) + stress_q.mean(axis=0)
    return stress


def nodal_stress(
    mesh: dict[str, Any],
    material: dict[str, Any],
    controls: dict[str, Any],
    primary_variables: dict[str, Any],
    method: str = "average",
) -> npt.NDArray[np.float64]:
    """Nodal total stress (nn, nv).

    `method` "average": stress evaluated at each element's nodes, averaged over the elements
    sharing each node. "L2": consistent L2 projection of the quadrature-point stress,
    M s = int N^T sigma with M = int N^T N.
    """
    mesh_data = _get_region_mesh_data(mesh, is_fracture=False, is_well=False)
    nn = mesh_data.nn
    projection = method == "L2"
    stresses = _element_stresses(mesh, material, controls, primary_variables, at_nodes=not projection)

    stress_sum = np.zeros((nn, material["sigma_0"].shape[1]))  # sum of nodal stresses, or int N^T sigma
    if projection:
        mass_entries = []
        for _, enodes, N, stress, wJ in stresses:
            stress_sum[enodes] += N.T @ (stress * wJ[:, None])
            N_q = N[:, None, :]
            mass_entries.append(_vctrzd_quad_point_integration(N_q, 1.0, N_q, wJ, True).ravel())
        elem_order = np.concatenate(mesh_data.region_lists).astype(np.int32)
        mass = _assemble_global_matrices(
            {"M": np.concatenate(mass_entries)},
            mesh_data.conn,
            nn,
            mesh_data.nne_list,
            mesh_data.elem_indx,
            elem_order,
        )["M"]
        change = splu(csc_matrix(mass)).solve(stress_sum)
    else:
        count = np.zeros(nn)
        for _, enodes, _, stress, _ in stresses:
            stress_sum[enodes] += stress
            count[enodes] += 1
        change = stress_sum / np.maximum(count, 1)[:, None]

    # in-situ stress plus the recovered change
    result: npt.NDArray[np.float64] = material["sigma_0"] + change
    return result


# ------------------------ Fracture face kinematics, loads and contact ----------------
def fracture_face_nodes(mesh: dict[str, Any]) -> npt.NDArray[np.int64]:
    """Matrix nodes on the positive and negative face at each fracture element node, (ne_f, nne_f, 2).

    Taken element by element from `matrix_conn`, which also resolves junctions. Tip nodes are unsplit
    and appear on both faces. Cached on the fracture mesh.
    """
    fmesh = mesh["fracture_mesh"]
    if "face_nodes" in fmesh:
        cached: npt.NDArray[np.int64] = fmesh["face_nodes"]
        return cached

    nodes_m = mesh["nodes"]
    nodes_f = fmesh["nodes"]
    conn_f = np.asarray(fmesh["conn"])
    face_nodes = np.empty((conn_f.shape[0], conn_f.shape[1], 2), dtype=np.int64)
    for e, enodes_f in enumerate(conn_f):
        sides = fmesh["matrix_conn"][e]
        if len(sides) != 2:
            raise ValueError(
                f"Fracture element {e} has a matrix element on only one side; "
                "fractures with mechanics must lie inside the matrix."
            )
        for s, e_m in enumerate(sides):
            enodes_m = np.asarray(mesh["conn"][e_m])
            distance = np.linalg.norm(nodes_m[enodes_m][None, :, :] - nodes_f[enodes_f][:, None, :], axis=2)
            face_nodes[e, :, s] = enodes_m[np.argmin(distance, axis=1)]
    fmesh["face_nodes"] = face_nodes
    return face_nodes


def element_penalty(
    mesh: dict[str, Any], material: dict[str, Any], controls: dict[str, Any]
) -> npt.NDArray[np.float64]:
    """Contact penalty eps_N = c_N E / h per fracture element, (ne_f,).

    E is the mean Young's modulus of the two adjacent matrix elements and h the mean element length of
    the fracture. A per-fracture h keeps the penalty uniform where the cutter leaves uneven segments.
    """
    fmesh = mesh["fracture_mesh"]
    nodes_f = fmesh["nodes"]
    conn_f = np.asarray(fmesh["conn"])
    youngs = material["solid"]["youngs_mod"]
    regions = mesh["regions"]
    factor = controls["mechanics_controls"]["penalty_factor"]

    lengths = np.linalg.norm(nodes_f[conn_f[:, -1]] - nodes_f[conn_f[:, 0]], axis=1)
    h = np.empty(conn_f.shape[0])
    for elems in fmesh["region_lists"]:
        h[elems] = lengths[elems].mean()

    penalty = np.empty(conn_f.shape[0])
    for e in range(conn_f.shape[0]):
        E = np.mean([_get_region_value(youngs, regions[e_m]) for e_m in fmesh["matrix_conn"][e]])
        penalty[e] = factor * E / h[e]
    return penalty


def integrate_fracture_interface(
    mesh: dict[str, Any], material: dict[str, Any], controls: dict[str, Any]
) -> tuple[FractureInterface, csc_matrix, npt.NDArray[np.float64]]:
    """Fracture face operators, the fracture pressure coupling K_upf, (nsd*nn, nn_total), and the in-situ
    stress face load F_fracture_0, (nsd*nn, 1).

    K_upf p_f - F_fracture_0 = -int [[N_u]]^T (p_f n + sigma_0 . n), with K_upf = -int [[N_u]]^T n N_f.
    """
    nsd = mesh["nsd"]
    nn = mesh["nn"]
    axisymmetric = bool(controls["axisymmetric_on"])
    fmesh = mesh["fracture_mesh"]
    mesh_data = _get_region_mesh_data(mesh, is_fracture=True, is_well=False)
    nodes_f = fmesh["nodes"]
    sigma_0 = material["sigma_0"]
    penalty_e = element_penalty(mesh, material, controls)
    w_m_e = material["fracture"]["w_m"]
    ne_f, nne_f = np.asarray(fmesh["conn"]).shape
    points_e = np.zeros((ne_f, 0), dtype=np.int64)
    projection = np.zeros((ne_f, nne_f, 0))

    rows_G: list[npt.NDArray[np.int64]] = []
    cols_G: list[npt.NDArray[np.int64]] = []
    vals_G: list[npt.NDArray[np.float64]] = []
    rows_N: list[npt.NDArray[np.int64]] = []
    cols_N: list[npt.NDArray[np.int64]] = []
    vals_N: list[npt.NDArray[np.float64]] = []
    weights: list[npt.NDArray[np.float64]] = []
    penalty: list[npt.NDArray[np.float64]] = []
    element: list[npt.NDArray[np.int64]] = []
    w_m: list[npt.NDArray[np.float64]] = []
    K_w: list[npt.NDArray[np.float64]] = []
    normals: list[npt.NDArray[np.float64]] = []
    traction_0: list[npt.NDArray[np.float64]] = []

    k = 0  # first Gauss point of the current element
    for r in range(mesh_data.num_regions):
        n = np.asarray(fmesh["fracture_normal"][r], dtype=np.float64).ravel()[:nsd]
        tangent = np.asarray(fmesh["fracture_tangent"][r], dtype=np.float64).reshape(-1, 1)[:nsd]
        K_w_r = _get_region_value(material["fracture"]["normal_stiffness"], r)
        elem_type_counter = -1  # force an update on the first element
        for e in mesh_data.region_lists[r]:
            new_type_idx = bisect.bisect_left(mesh_data.elem_indx, e)
            if new_type_idx != elem_type_counter:
                W, nq, N_q, GN_q = _get_quad_points_and_shape_functions(
                    mesh_data.nq_list[new_type_idx], mesh_data.nsd, mesh_data.etype_list[new_type_idx]
                )
                elem_type_counter = new_type_idx
                if points_e.shape[1] != nq:  # size the per-element Gauss-point arrays
                    points_e = np.zeros((ne_f, nq), dtype=np.int64)
                    projection = np.zeros((ne_f, nne_f, nq))

            enodes_f = np.asarray(mesh_data.conn[e])
            _, detJ = getB(GN_q, nodes_f[enodes_f, :] @ tangent, mesh_data.nsd)
            wJ = W * detJ
            if axisymmetric:
                wJ = wJ * (N_q[:, 0, :] @ nodes_f[enodes_f, 0])

            matches = list(_iter_fracture_matrix_matches(e, list(enodes_f), nodes_f, N_q, mesh))
            if len(matches) != 2:
                raise ValueError(
                    f"Fracture element {e} has a matrix element on only one side; "
                    "fractures with mechanics must lie inside the matrix."
                )
            (_, enodes_pos, _, N_pos), (_, enodes_neg, _, N_neg) = matches

            # jump rows: positive-side shape functions minus negative-side ones, per component
            points = k + np.arange(nq)
            side_nodes = np.concatenate([enodes_pos, enodes_neg])
            side_values = np.hstack([N_pos[:, 0, :], -N_neg[:, 0, :]])
            for d in range(nsd):
                rows_G.append(np.repeat(nsd * points + d, side_nodes.size))
                cols_G.append(np.tile(nsd * side_nodes + d, nq))
                vals_G.append(side_values.ravel())
            rows_N.append(np.repeat(points, enodes_f.size))
            cols_N.append(np.tile(nn + enodes_f, nq))
            vals_N.append(N_q[:, 0, :].ravel())

            # in-situ stress face traction sigma_0 . n at the Gauss points
            sigma_0_q = N_pos[:, 0, :] @ sigma_0[enodes_pos]
            traction_0.append(np.array([_voigt_traction(sigma_0_q[q], n) for q in range(nq)]))
            weights.append(np.asarray(wJ, dtype=np.float64).ravel())
            penalty.append(np.full(nq, penalty_e[e]))
            element.append(np.full(nq, e, dtype=np.int64))
            w_m.append(np.full(nq, w_m_e[e]))
            K_w.append(np.full(nq, K_w_r))
            # lumped projection of Gauss values to the element nodes: N_a w_q / sum_q N_a w_q
            weighted = N_q[:, 0, :].T * np.asarray(wJ, dtype=np.float64).ravel()  # (nne, nq)
            projection[e] = weighted / weighted.sum(axis=1, keepdims=True)
            points_e[e] = points
            normals.append(np.tile(n, (nq, 1)))
            k += nq

    n_q = k
    jump = csc_matrix(
        (np.concatenate(vals_G), (np.concatenate(rows_G), np.concatenate(cols_G))),
        shape=(nsd * n_q, nsd * nn),
    )
    normal = np.vstack(normals)
    # normal jump: n . [[u]] at each Gauss point
    jump_normal = csc_matrix(diags(normal[:, 0]) @ jump[0::nsd, :])
    for d in range(1, nsd):
        jump_normal = csc_matrix(jump_normal + diags(normal[:, d]) @ jump[d::nsd, :])
    shape_f = csc_matrix(
        (np.concatenate(vals_N), (np.concatenate(rows_N), np.concatenate(cols_N))),
        shape=(n_q, mesh["nn_total"]),
    )
    w = np.concatenate(weights)
    element_q = np.concatenate(element)

    K_upf = csc_matrix(-(jump_normal.T @ diags(w) @ shape_f))
    F_fracture_0 = np.asarray(jump.T @ (w[:, None] * np.vstack(traction_0)).ravel()).reshape(-1, 1)
    interface = FractureInterface(
        jump,
        jump_normal,
        w,
        np.concatenate(penalty),
        element_q,
        np.concatenate(w_m),
        np.concatenate(K_w),
        points_e,
        projection,
        normal,
        _cohesive_parameters(material, controls, element_q),
    )
    return interface, K_upf, F_fracture_0


def _cohesive_parameters(
    material: dict[str, Any], controls: dict[str, Any], element: npt.NDArray[np.int64]
) -> CohesiveParameters:
    """Cohesive parameters at the fracture Gauss points of the given elements; empty without cohesion."""
    if "cohesive_elements" not in material["fracture"]:
        return CohesiveParameters.empty()
    cohesive = material["fracture"]["cohesive"]
    elements = material["fracture"]["cohesive_elements"]
    law = cohesive["law"]
    weakening_ratio = float(cohesive.get("weakening_ratio", 0.0))
    f_u = elements["f_u"][element]
    w_c, delta_p = cohesive_scales(law, f_u, elements["G_c"][element], weakening_ratio)
    c_eps = controls["mechanics_controls"]["damage_peak_smoothing_coefficient"]
    gamma = elements["gamma"][element]
    return CohesiveParameters(law, weakening_ratio, f_u, gamma, w_c, delta_p, c_eps * w_c)


def fracture_interface_force(
    interface: FractureInterface, u: npt.NDArray[np.float64], delta_max_c: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], csc_matrix, npt.NDArray[np.float64]]:
    """Face force int [[N_u]]^T t_f, (nsd*nn, 1), its tangent, and the cohesive effective displacement
    (nq,) at displacement u, with t_f = t_coh - sigma_c n.

    Without cohesion (empty delta_max_c) t_f is the contact traction and the effective displacement empty.
    """
    normal = interface.normal
    nq, nsd = normal.shape
    jump = (interface.jump @ u).reshape(nq, nsd)

    # contact: -sigma_c n and tangent -dsigma_c/dg_n n n^T
    g_n = np.sum(jump * normal, axis=1)
    sigma_c, dsigma_c_dg = penalty_contact(g_n, interface.penalty)
    traction = -sigma_c[:, None] * normal
    tangent_q = -dsigma_c_dg[:, None, None] * (normal[:, :, None] * normal[:, None, :])  # (nq, nsd, nsd)

    # cohesion
    delta = np.empty(0)
    if delta_max_c.size:
        t_coh, D_coh, delta = cohesive_traction(jump, normal, delta_max_c, interface.cohesive)
        traction = traction + t_coh
        tangent_q = tangent_q + D_coh

    # integrate through the jump operator with the block-diagonal point tangents
    w = interface.weights
    residual = (interface.jump.T @ (w[:, None] * traction).ravel()).reshape(-1, 1)
    rows = nsd * np.arange(nq)[:, None, None] + np.arange(nsd)[None, :, None]  # (nq, nsd, 1)
    cols = nsd * np.arange(nq)[:, None, None] + np.arange(nsd)[None, None, :]  # (nq, 1, nsd)
    rows, cols = np.broadcast_arrays(rows, cols)
    block = coo_matrix(
        ((w[:, None, None] * tangent_q).ravel(), (rows.ravel(), cols.ravel())), shape=(nsd * nq, nsd * nq)
    )
    tangent = csc_matrix(interface.jump.T @ block.tocsc() @ interface.jump)
    return residual, tangent, delta


def fracture_aperture(interface: FractureInterface, u: npt.NDArray[np.float64]) -> FractureAperture:
    """Bandis-Barton aperture at the Gauss points and element nodes for every column of u, and dw/du."""
    g_n = interface.jump_normal @ u  # (nq, n_levels)
    penalty = interface.penalty[:, None]
    sigma_c, dsigma_c_dg = penalty_contact(g_n, penalty)
    w, dw_dg = bandis_barton_aperture(
        g_n, sigma_c, dsigma_c_dg, interface.w_m[:, None], interface.K_w[:, None]
    )
    gauss = w[interface.points]  # (ne_f, nq_e, n_levels)
    element_nodal = interface.projection @ gauss  # (ne_f, nne_f, n_levels)
    D_w = csc_matrix(diags(dw_dg[:, 0]) @ interface.jump_normal)
    return FractureAperture(gauss, element_nodal, D_w, interface.points)
