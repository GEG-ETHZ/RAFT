"""Write mesh data to vtk file."""

from typing import Any

import numpy as np
import numpy.typing as npt


def write_mesh2vtk(  # noqa: C901
    filename: str,
    description: str,
    nodes: npt.NDArray[np.float64],
    conn: npt.NDArray[np.float64],
    scalar_data_point: list[dict[str, Any]] | None = None,
    vector_data_point: list[dict[str, Any]] | None = None,
    scalar_data_cell: list[dict[str, Any]] | None = None,
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
    conn : np.ndarray
        Connectivity matrix, shape (ne, nne)
    scalar_data : list of dict, optional
        Each entry is {'name': str, 'data': np.ndarray of shape (nn,)}
        Example: [{'name': 'd0x', 'data': array}, {'name': 'd0y', 'data': array}]
    vector_data : list of dict, optional
        Each entry is {'name': str, 'data': np.ndarray of shape (nn, 2)}
        Example: [{'name': 'disp', 'data': array}]

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

    nodes = np.array(nodes, dtype=float)
    conn_int: npt.NDArray[np.int_] = np.asarray(conn, dtype=np.int_)

    nn = nodes.shape[0]  # number of nodes
    nsd = nodes.shape[1]  # number of spatial dimensions
    ne = conn_int.shape[0]  # number of elements
    nne = conn_int.shape[1]  # nodes per element
    ndp = len(scalar_data_point)  # number of scalar fields
    ndc = len(scalar_data_cell)  # number of scalar fields

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
                [3, 0, 0],  # L2
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
        cell_type = cell_types[nne - 1, nsd - 1]
        if cell_type == 0:
            raise ValueError(f"WriteMesh2VTK: Element not supported (nsd={nsd}, nne={nne})")

        if nne == 9 or nne == 27:
            nodes_out, conn_int, scalar_data, vector_data = _strip_nodes(
                nodes_out, conn_int, scalar_data_point, vector_data_point, nne
            )
            nn = nodes_out.shape[0]

        # Prepend node-count column to connectivity
        conn_vtk = np.hstack([np.full((ne, 1), nne, dtype=int), conn_int])

        fid.write("DATASET UNSTRUCTURED_GRID\n")
        fid.write(f"POINTS {nn} float\n")
        for row in nodes_out:
            fid.write(f"{row[0]:f} {row[1]:f} {row[2]:f}\n")

        fid.write(f"CELLS {ne} {(nne + 1) * ne}\n")
        for row in conn_vtk:
            fid.write(" ".join(str(v) for v in row) + "\n")

        fid.write(f"CELL_TYPES {ne}\n")
        for _ in range(ne):
            fid.write(f"{cell_type}\n")

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
            for j in range(nn):
                fid.write(f"{data[j, 0]:.4e} {data[j, 1]:.4e} {data[j, 2]:.4e} \n")

        # --- Scalar point data ---
        if ndc > 0:
            fid.write(f"CELL_DATA {ne}\n")

        for field in scalar_data_cell:
            fid.write(f"SCALARS {field['name']} float 1\n")
            fid.write("LOOKUP_TABLE default\n")
            data = np.asarray(field["data"]).ravel()
            for val in data:
                fid.write(f"{val:.4e}\n")


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
