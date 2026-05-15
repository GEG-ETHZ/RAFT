"""Import mesh and create mesh objects."""

import numpy as np
import numpy.typing as npt


def loadmesh(meshfile: str, nsd: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Reads mesh data from a Gmsh (.msh) file (Version 2.x format).

    Parameters:
    -----------
    meshfile : str
        Name/path of the mesh file.
    nsd : int
        Number of space dimensions to extract.

    Returns:
    --------
    nodes : NDArray[float64]
        Coordinates of mesh nodes.
    conn : NDArray[int64]
        Element connectivity matrix.
    """
    try:
        with open(meshfile) as f:
            lines: list[str] = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("Error: File {meshfile} not found.")
        return None, None

    # Find the start of nodes and elements sections
    try:
        n_str_idx: int = lines.index("$Nodes")
        e_str_idx: int = lines.index("$Elements")
    except ValueError:
        print("Error: $Nodes or $Elements section not found in file.")
        return None, None

    # Parse number of nodes and elements
    nnode: int = int(lines[n_str_idx + 1])
    nelem: int = int(lines[e_str_idx + 1])

    # Initialize nodes array
    nodes: npt.NDArray[np.float64] = np.zeros((nnode, nsd), dtype=np.float64)

    # Extract nodal coordinates
    for i in range(nnode):
        temp_line: list[str] = lines[n_str_idx + 2 + i].split()
        temp_data: list[float] = [float(x) for x in temp_line]
        nodes[i, :] = temp_data[1 : nsd + 1]

    conn_list: list[list[int]] = []

    # Extract element connectivity
    for i in range(nelem):
        temp_line = lines[e_str_idx + 2 + i].split()
        temp_data_int: list[int] = [int(x) for x in temp_line]

        elmtyp = temp_data_int[1]
        edg = 1  # edge element flag

        # Element type logic
        match (nsd, elmtyp):
            case (3, 5):  # B8
                nne, edg = 8, 0
            case (3, 12):  # B27
                nne, edg = 27, 0
            case (2, 3):  # Q4
                nne, edg = 4, 0
            case (2, 10):  # Q9
                nne, edg = 9, 0

        # Only append elements if they are not edge elements
        if edg == 0:
            conn_list.append(temp_data_int[-nne:])

    # Convert list to numpy array
    conn: npt.NDArray[np.int64] = (
        np.array(conn_list, dtype=np.int64) if conn_list else np.empty((0, 0), dtype=np.int64)
    )

    return nodes, conn


# script_dir = Path(__file__).parent
# config_path = script_dir.parent / "config" / "box3D_B27.msh"
# [nodes, conn] = loadmesh(str(config_path), 3)
