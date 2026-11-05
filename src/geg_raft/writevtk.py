# SPDX-FileCopyrightText: 2026 ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Write mesh data to vtk file."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt


def write_mesh2vtk(  # noqa: C901
    filename: str,
    description: str,
    nodes: npt.NDArray[np.float64],
    conn: Sequence[Sequence[int]],
    scalar_data_point: list[dict[str, Any]] | None = None,
    vector_data_point: list[dict[str, Any]] | None = None,
    scalar_data_cell: list[dict[str, Any]] | None = None,
    vector_data_cell: list[dict[str, Any]] | None = None,
) -> None:
    """Write FEM mesh data to a file in VTK format.

    Parameters:
    ----------
    filename : str
        Output file path, should end with .vtk
    description : str
        String describing the model
    nodes : np.ndarray
        Nodal coordinates, shape (n, nsd)
    conn :  sequence of sequence of int
        Ragged connectivity list - conn[e] is the list of node ids for element e.
        Elements are allowed to have differing lengths so that a mesh can mix
        element types (e.g. hex + tet), matching the ragged connectivity used
        elsewhere.
    scalar_data : list of dict, optional
        Each entry is {'name': str, 'data': np.ndarray of shape (nn,)}
        Example: [{'name': 'd0x', 'data': array}, {'name': 'd0y', 'data': array}]
    vector_data : list of dict, optional
        Each entry is {'name': str, 'data': np.ndarray of shape (nn, 2)}
        Example: [{'name': 'disp', 'data': array}]
    scalar_data_cell, vector_data_cell : list of dict, optional
        As above, with one entry per element: data of shape (ne,) or (ne, nsd)

    Notes:
    -----
    Supported element types:
        1D: L2, L3, L4
        2D: T3, Q4, Q9, Q8
        3D: B8, B27, B20
    """
    if scalar_data_point is None:
        scalar_data_point = []
    if vector_data_point is None:
        vector_data_point = []
    if scalar_data_cell is None:
        scalar_data_cell = []
    if vector_data_cell is None:
        vector_data_cell = []

    nodes = np.array(nodes, dtype=float)
    conn_list: list[npt.NDArray[np.int_]] = [np.asarray(enodes, dtype=np.int_) for enodes in conn]

    nn = nodes.shape[0]  # number of nodes
    nsd = nodes.shape[1]  # number of spatial dimensions
    ne = len(conn_list)  # number of elements
    nne_per_elem = np.array([len(enodes) for enodes in conn_list])  # nodes per element, may vary
    ndp = len(scalar_data_point)  # number of scalar fields
    ndc = len(scalar_data_cell) + len(vector_data_cell)  # number of cell fields

    # Pad 2D nodes with a zero z-column for VTK output
    if nsd == 2:
        nodes_out = np.hstack([nodes, np.zeros((nn, 1))])
    elif nsd == 3:
        nodes_out = nodes.copy()
    else:
        raise ValueError("Only 2D and 3D node coordinates are supported.")

    with open(filename, "w") as fid:
        # Header
        fid.write("# vtk DataFile Version 2.0\n")
        fid.write(f"HFX mesh description: {description}\n")
        fid.write("ASCII\n")

        # cell type array for look-up
        # nne x nsd
        cell_types = np.array(
            [
                [0, 0, 0],
                [3, 3, 3],  # L2
                [21, 5, 0],  # L3, T3
                [4, 9, 10],  # L4, Q4, T4
                [0, 0, 14],  # T5
                [0, 22, 13],  # T6, H6
                [0, 0, 0],
                [0, 23, 12],  # Q8, B8
                [0, 23, 0],  # Q9
                [0, 0, 24],  # T10
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 25],  # B20
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 25],  # B27
            ]
        )
        # cell type is looked up per element
        cell_type_per_elem = cell_types[nne_per_elem - 1, nsd - 1]
        bad = np.nonzero(cell_type_per_elem == 0)[0]
        if bad.size:
            e0 = int(bad[0])
            raise ValueError(
                f"WriteMesh2VTK: Element {e0} not supported (nsd={nsd}, nne={int(nne_per_elem[e0])})"
            )

        # Paraview can't render Q9 or B27, so strip the middle nodes. Currently
        # only perform this if the mesh is uniform, needs to be updated for
        # mixed mode higher dimensionsal meshes.
        uniform_nne = int(nne_per_elem[0]) if np.all(nne_per_elem == nne_per_elem[0]) else None
        if uniform_nne in (9, 27):
            conn_uniform = np.array(conn_list, dtype=np.int_)
            nodes_out, conn_uniform, scalar_data_point, vector_data_point = _strip_nodes(
                nodes_out, conn_uniform, scalar_data_point, vector_data_point, uniform_nne
            )
            conn_list = list(conn_uniform)
            nn = nodes_out.shape[0]

        # Begin writing
        # nodes
        fid.write("DATASET UNSTRUCTURED_GRID\n")
        fid.write(f"POINTS {nn} float\n")
        for row in nodes_out:
            fid.write(f"{row[0]:f} {row[1]:f} {row[2]:f}\n")

        # connectivity
        cells_size = int(sum(len(enodes) for enodes in conn_list)) + ne
        fid.write(f"CELLS {ne} {cells_size}\n")
        for enodes in conn_list:
            fid.write(str(len(enodes)) + " " + " ".join(str(v) for v in enodes) + "\n")

        # cell types
        fid.write(f"CELL_TYPES {ne}\n")
        for ct in cell_type_per_elem:
            fid.write(f"{ct}\n")

        # --- Scalar point data ---
        if ndp > 0:
            fid.write(f"POINT_DATA {nn}\n")

        for field in scalar_data_point:
            fid.write(f"SCALARS {field['name']} float 1\n")
            fid.write("LOOKUP_TABLE default\n")
            data = np.asarray(field["data"]).ravel()
            for val in data:
                fid.write(f"{val:.4e}\n")

        # --- Vector point data ---
        for field in vector_data_point:
            fid.write(f"VECTORS {field['name']} float\n")
            data = np.asarray(field["data"])

            nsv = np.size(data, axis=1)
            if nsd != 3:
                # append zeros to vectors
                data = np.hstack([data, np.zeros((nn, 3 - nsv))])

            for j in range(nn):
                fid.write(f"{data[j, 0]:.4e} {data[j, 1]:.4e} {data[j, 2]:.4e} \n")

        # --- Scalar cell data ---
        if ndc > 0:
            fid.write(f"CELL_DATA {ne}\n")

        for field in scalar_data_cell:
            fid.write(f"SCALARS {field['name']} float 1\n")
            fid.write("LOOKUP_TABLE default\n")
            data = np.asarray(field["data"]).ravel()
            for val in data:
                fid.write(f"{val:.4e}\n")

        # --- Vector cell data ---
        for field in vector_data_cell:
            fid.write(f"VECTORS {field['name']} float\n")
            data = np.asarray(field["data"])

            nsv = np.size(data, axis=1)
            if nsv != 3:
                # append zeros to vectors
                data = np.hstack([data, np.zeros((ne, 3 - nsv))])

            for j in range(ne):
                fid.write(f"{data[j, 0]:.4e} {data[j, 1]:.4e} {data[j, 2]:.4e} \n")


# Helper
def _strip_nodes(
    nodes_out: npt.NDArray[np.float64],
    conn0: npt.NDArray[np.int_],
    scalar_data: list[dict[str, Any]],
    vector_data: list[dict[str, Any]],
    nne: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int_], list[dict[str, Any]], list[dict[str, Any]]]:
    """ParaView cannot render Q9/27 elements. Remove higher order nodes to render as Q8/B20."""
    keep_cols = 8 if nne == 9 else 20

    # Identify nodes to remove from the discarded columns
    nodes_to_remove = np.unique(conn0[:, keep_cols:])

    # Trim connectivity to the retained columns
    conn0 = conn0[:, :keep_cols].copy()

    # Remove nodes from node list, scalar fields, and vector fields
    nodes_out = np.delete(nodes_out, nodes_to_remove, axis=0)

    for field in scalar_data:
        field["data"] = np.delete(np.asarray(field["data"]), nodes_to_remove, axis=0)

    for field in vector_data:
        field["data"] = np.delete(np.asarray(field["data"]), nodes_to_remove, axis=0)

    # Renumber remaining connectivity after node removal
    nodes_sorted = np.sort(nodes_to_remove)
    for mn in nodes_sorted:
        conn0[conn0 > mn] -= 1
        nodes_sorted[nodes_sorted > mn] -= 1

    return nodes_out, conn0, scalar_data, vector_data
