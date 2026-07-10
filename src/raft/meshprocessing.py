"""Import mesh, assign regions, extract relevant mesh properties."""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


def load_Gmsh(meshfile: str, nsd: int) -> tuple[npt.NDArray[np.float64] | None, npt.NDArray[np.int64] | None]:  # noqa
    """Reads mesh data from a Gmsh (.msh) file (Version 2.x format).

    meshfile : name of mesh file
    nsd : number of space dimensions of the analysis

    nodes : nn x nsd array with coordinates of mesh nodes.
    conn : ne x nne array with element connectivity.
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
            case (3, 4):  # T4
                nne, edg = 4, 0
            case (3, 12):  # B27
                nne, edg = 27, 0
            case (2, 3):  # Q4
                nne, edg = 4, 0
            case (2, 10):  # Q9
                nne, edg = 9, 0
            case (2, 2):  # T3
                nne, edg = 3, 0

        # Only append elements if they are not edge elements
        if edg == 0:
            conn_list.append(temp_data_int[-nne:])

    # Convert list to numpy array
    conn: npt.NDArray[np.int64] = (
        np.array(conn_list, dtype=np.int64) if conn_list else np.empty((0, 0), dtype=np.int64)
    )

    return nodes, conn


def _extract_exterior(mesh: dict[str, Any]) -> dict[str, Any]:  # noqa
    """Find the nodes along the exterior for applying boundary conditions.

    Note that this extraction assumes the reservoir has 6 sides that do not have to be planar.
    """
    conn = mesh["conn"]
    nodes = mesh["nodes"]
    nne = mesh["nne"]
    ne = mesh["ne"]
    nsd = mesh["nsd"]

    # Define element faces. Node numberings should have outwards pointing normals.
    if nne == 8:  # B8 elements
        faces = {
            "face0": [0, 3, 2, 1],
            "face1": [4, 5, 6, 7],
            "face2": [0, 1, 5, 4],
            "face3": [0, 4, 7, 3],
            "face4": [1, 2, 6, 5],
            "face5": [2, 3, 7, 6],
        }
    elif nne == 4 and nsd == 3:  # T4
        faces = {
            "face0": [0, 1, 3],
            "face1": [0, 3, 2],
            "face2": [0, 2, 1],
            "face3": [1, 2, 3],
        }
    elif nne == 4 and nsd == 2:  # Q4
        faces = {
            "face0": [0, 1],
            "face1": [1, 2],
            "face2": [2, 3],
            "face3": [3, 0],
        }
    elif nne == 3 and nsd == 2:  # T3
        faces = {
            "face0": [0, 1],
            "face1": [1, 2],
            "face2": [2, 0],
        }
    else:
        raise NotImplementedError(f"Boundary extraction for {nne}-node elements in {nsd} dimensions.")

    # Loop through all elements, collect a list of faces in global coordinates,
    # then find unique faces.

    # use dictionary to track faces and keep original order for calculating normals.
    # dictionary structure: sorted_face: [count, original face]
    face_tracker: dict[tuple[int, ...], list[Any]] = {}

    for elem in range(ne):
        for face_name, face_local in faces.items():
            original_face = tuple(conn[elem, face_local])  # for normals later
            sorted_face = tuple(sorted(original_face))  # for uniqueness
            if sorted_face in face_tracker:
                # Increment count
                face_tracker[sorted_face][0] += 1
            else:
                # Store count and original
                face_tracker[sorted_face] = [1, original_face]

    # sory through the faces and extract all unique faces
    boundary_faces = [data[1] for data in face_tracker.values() if data[0] == 1]

    # Loop through boundary faces, calculate the normal of each face and sort it based on its closest axis
    boundaries: dict[str, Any] = {}
    boundaries["x_min"] = set()
    boundaries["x_max"] = set()
    boundaries["y_min"] = set()
    boundaries["y_max"] = set()
    boundaries["z_min"] = set()
    boundaries["z_max"] = set()

    boundaries["x_min_conn"] = []
    boundaries["x_max_conn"] = []
    boundaries["y_min_conn"] = []
    boundaries["y_max_conn"] = []
    boundaries["z_min_conn"] = []
    boundaries["z_max_conn"] = []

    for face in boundary_faces:
        # calculate plane normal
        xI = nodes[face, :]
        if nsd == 3:
            n = np.cross(xI[1, :] - xI[0, :], xI[2, :] - xI[0, :])
            # project n onto axis normals
            ex = n @ [1, 0, 0]
            ey = n @ [0, 1, 0]
            ez = n @ [0, 0, 1]

            # Find which component has the largest magnitude
            components = {"x": ex, "y": ey, "z": ez}
        else:
            # 2D vectors
            n = -np.array([-(xI[1, 1] - xI[0, 1]), xI[1, 0] - xI[0, 0]])
            # project n onto axis normals
            ex = n @ [1, 0]
            ey = n @ [0, 1]

            # Find which component has the largest magnitude
            components = {"x": ex, "y": ey}

        max_component = max(components, key=lambda k: abs(components[k]))
        # Append to appropriate list based on sign
        side = "max" if components[max_component] > 0 else "min"

        boundaries[f"{max_component}_{side}"].update(face)
        boundaries[f"{max_component}_{side}_conn"].append(list(face))

    mesh["boundaries"] = boundaries

    x_all = boundaries["x_max"] | boundaries["x_min"]
    y_all = boundaries["y_max"] | boundaries["y_min"]
    z_all = boundaries["z_max"] | boundaries["z_min"]

    mesh["boundaries"]["x_all"] = x_all
    mesh["boundaries"]["y_all"] = y_all
    mesh["boundaries"]["z_all"] = z_all
    mesh["boundaries"]["n_all"] = np.arange(0, mesh["nn"])

    if nsd == 2:
        # Remove the z faces
        mesh["boundaries"].pop("z_max")
        mesh["boundaries"].pop("z_min")
        mesh["boundaries"].pop("z_max_conn")
        mesh["boundaries"].pop("z_min_conn")
        mesh["boundaries"].pop("z_all")

    return mesh


def _assign_regions(config_regions: list[dict[str, Any]], mesh: dict[str, Any]) -> list[int]:
    ne = mesh["ne"]
    nsd = mesh["nsd"]
    nodes = mesh["nodes"]
    conn = mesh["conn"]
    tol = 1.0e-6

    # Step 1: Initialize all elements to region 0
    regions_list = np.array([0] * ne)

    # Step 2: Calculate all element centroids
    centroids = np.mean(nodes[conn, :], 1)
    if nsd == 2:
        centroids = np.hstack((centroids, np.zeros((ne, 1))))

    # Step 3: Assign regions in order of specification
    supported_shapes = ["polyhedron", "sphere", "cylinder"]

    for region_shape in config_regions:
        shape_type = region_shape.get("shape")
        values = region_shape["values"]
        region_number = region_shape["region_number"]

        if shape_type == "polyhedron":
            from scipy.spatial import ConvexHull

            values = np.array(values)  # turn value into array of points
            hull = ConvexHull(values)  # calculate the convex shape formed by the points
            A, b = (
                hull.equations[:, :-1],
                hull.equations[:, -1:],
            )  # extract the equations defining the 0-level set
            # The level set defines the hull such that all points Ax + b = 0 form the shell of the hull.
            # If Ax + b < 0, then the point is within the polyhedron.
            elem_in_region = np.all(centroids @ A.T + b.T < tol, axis=1)

        elif shape_type == "sphere":
            center = values[0]
            radius = values[1]
            d = np.linalg.norm(centroids - center, axis=1)
            elem_in_region = d <= (radius + tol)

        elif shape_type == "cylinder":
            center = values[0]
            radius = values[1]
            unit_vector = values[2]
            unit_vector = unit_vector / np.linalg.norm(unit_vector)
            try:
                height = values[3] + tol
            except IndexError:
                height = 1.0e9

            # First, check height
            centroids_shift = centroids - center
            axial_distance = np.dot(centroids_shift, unit_vector)
            axial_inside = abs(axial_distance) <= height

            # Second, check radial distance using perpendicular components
            centroids_projected = centroids - axial_distance.reshape(-1, 1) * unit_vector
            d = np.linalg.norm(centroids_projected - center, axis=1)
            radial_inside = d <= (radius + tol)

            elem_in_region = axial_inside & radial_inside

        else:
            raise ValueError(
                f"Unknown shape type: '{shape_type}'\n"
                f"Supported shapes: {', '.join(supported_shapes)}\n"
                f"Region spec: {region_shape}"
            )

        regions_list[elem_in_region] = region_number

    return list(regions_list)


def _create_region_lists(mesh: dict[str, Any]) -> dict[str, Any]:
    """Create lists of all elements associated with each region."""
    regions = np.asarray(mesh["regions"])
    num_regions = max(regions) + 1
    region_lists = []
    for r in range(0, num_regions):
        region_lists.append(np.nonzero(np.asarray(regions == r))[0])

    mesh["region_lists"] = region_lists
    mesh["num_regions"] = num_regions
    return mesh


def _import_regions(config_mesh: dict[str, Any], mesh: dict[str, Any]) -> dict[str, Any]:
    config_regions = config_mesh["regions"]

    if len(config_regions) == 0:
        # Case 3: If regions are not defined, apply homogeneity
        mesh["regions"] = [0] * mesh["ne"]
    elif "list" in config_regions:
        # Case 1: regions are applied via a premade file list
        region_path = Path(config_mesh["config_dir"]) / config_mesh["regions"]

        with open(region_path) as file:
            element_regions = [int(line.strip()) for line in file]

        mesh["regions"] = element_regions
    elif len(config_regions) > 0:
        # Case 2: construct regions from applied geometries

        mesh["regions"] = _assign_regions(config_regions, mesh)

    mesh = _create_region_lists(mesh)

    return mesh


def load_mesh(config_mesh: dict[str, Any], verbose: bool) -> dict[str, Any]:
    """Load mesh and define mesh structures."""
    config_path = Path(config_mesh["config_dir"]) / config_mesh["file"]
    [nodes, conn] = load_Gmsh(str(config_path), config_mesh["dimensions"])

    if nodes is None or conn is None:
        raise RuntimeError(f"Failed to load mesh from {config_path}")

    mesh: dict[str, Any] = {
        "nodes": nodes,
        "conn": conn - 1,
    }

    # Extract key values
    mesh["nn"] = np.shape(nodes)[0]  # number of nodes
    mesh["ne"] = np.shape(conn)[0]  # number of elements
    mesh["nne"] = np.shape(conn)[1]  # number of nodes per element
    mesh["nsd"] = np.shape(nodes)[1]  # number of space dimensions

    # Import regions
    mesh = _import_regions(config_mesh, mesh)

    # Define number of quadrature points
    match (mesh["nsd"], mesh["nne"]):
        case (3, 8):  # B8
            nq, etyp = 2, "B8"
        case (3, 4):  # T4
            nq, etyp = 2, "T4"
        case (3, 27):  # B27
            nq, etyp = 3, "B27"
        case (2, 4):  # Q4
            nq, etyp = 2, "Q4"
        case (2, 9):  # Q9
            nq, etyp = 3, "Q9"
        case (2, 3):  # T3
            nq, etyp = 1, "T3"

    mesh["n_quadrature"] = nq
    mesh["element_type"] = etyp

    # Extract sets of exterior nodes
    mesh = _extract_exterior(mesh)

    if verbose:
        print("----- Mesh statistics")
        print(f"  File            : {config_mesh['file']}")
        print(f"  Element type    : {etyp}")
        print(f"  Dimensions      : {mesh['nsd']}D")
        print(f"  Nodes           : {mesh['nn']}")
        print(f"  Elements        : {mesh['ne']}")
        print(f"  Quadrature pts  : {nq}^{mesh['nsd']} = {nq ** mesh['nsd']}\n")

    return mesh
