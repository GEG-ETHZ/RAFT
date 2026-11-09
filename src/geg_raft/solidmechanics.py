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

"""Linear (thermo-poro)elastic force balance of the matrix.

Voigt order: 2D [xx, yy, xy]; axisymmetric [rr, zz, rz, tt]; 3D [xx, yy, zz, xz, yz, xy].
Shear strains are engineering strains; shear stresses are tensor components.
"""

import bisect
from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import splu

from .dofmap import get_scatter
from .matrixassembly import (
    RegionMeshData,
    _assemble_global_matrices,
    _get_quad_points_and_shape_functions,
    _get_region_mesh_data,
    _get_region_value,
    _lumped_mass,
    _pad_sparse,
    _vctrzd_quad_point_integration,
)
from .mechanicsdata import MechanicsMatrices
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


def elasticity_matrix(
    E: float, nu: float, nsd: int, plane: str, axisymmetric: bool
) -> npt.NDArray[np.float64]:
    """Isotropic elasticity matrix D (Voigt form of the elasticity tensor C)."""
    if nsd == 3:
        # 3D matrix
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        G = E / (2.0 * (1.0 + nu))
        D = np.zeros((6, 6))
        D[:3, :3] = lam
        D[[0, 1, 2], [0, 1, 2]] = lam + 2.0 * G
        D[[3, 4, 5], [3, 4, 5]] = G
        return D
    if axisymmetric:
        # axisymmetric matrix with 4 stress components
        c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return c * np.array(
            [
                [1.0 - nu, nu, 0.0, nu],
                [nu, 1.0 - nu, 0.0, nu],
                [0.0, 0.0, 0.5 - nu, 0.0],
                [nu, nu, 0.0, 1.0 - nu],
            ]
        )
    if plane == "stress":
        # plane stress
        c = E / (1.0 - nu**2)
        return c * np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]])
    # plane strain
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return c * np.array([[1.0 - nu, nu, 0.0], [nu, 1.0 - nu, 0.0], [0.0, 0.0, 0.5 - nu]])


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
    return MechanicsMatrices(
        K_uu, _pad_sparse(coupling["K_up"], shape), _pad_sparse(coupling["K_uT"], shape), tuple(nodal_strain)
    )


def internal_force(
    matrices: MechanicsMatrices, primary_variables: dict[str, Any], material: dict[str, Any]
) -> npt.NDArray[np.float64]:
    """Internal force K_uu u + K_up (p - p_0) + K_uT (T - T_0) of the current iterate, (nsd*nn, 1)."""
    u = primary_variables["u"][:, 0:1]
    dp = primary_variables["p"][:, 0:1] - material["p_0"].reshape(-1, 1)
    dT = primary_variables["T"][:, 0:1] - material["T_0"].reshape(-1, 1)
    force: npt.NDArray[np.float64] = matrices.K_uu @ u + matrices.K_up @ dp + matrices.K_uT @ dT
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
    mesh: dict[str, Any], controls: dict[str, Any], primary_variables: dict[str, Any]
) -> npt.NDArray[np.float64]:
    """Element-mean volumetric strain (ne,) of the current iterate."""
    mesh_data = _get_region_mesh_data(mesh, is_fracture=False, is_well=False)
    nsd = mesh_data.nsd
    axisymmetric = bool(controls["axisymmetric_on"])
    m = identity_voigt(nsd, axisymmetric)
    u = primary_variables["u"][:, 0]

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

    # current iterate on matrix nodes
    u = primary_variables["u"][:, 0]
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
