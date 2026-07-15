"""Import mesh, assign regions, extract relevant mesh properties."""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy as sp


def load_Gmsh(meshfile: str, nsd: int) -> tuple[npt.NDArray[np.float64] | None, list[list[int]] | None]:  # noqa
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
            node_list = temp_data_int[-nne:]
            node_list = [n - 1 for n in node_list]
            conn_list.append(node_list)

    return nodes, conn_list


def _get_element_face_lists(nne: int, nsd: int) -> dict[str, list[int]]:
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
    return faces


def _get_all_edges(
    conn: list[list[int]], ne: int, nne: list[int], nsd: int, elem_indx: list[int]
) -> dict[tuple[int, ...], list[Any]]:
    # use dictionary to track faces and keep original order for calculating normals.
    # dictionary structure: sorted_face: [count, original face]
    face_tracker: dict[tuple[int, ...], list[Any]] = {}

    faces = _get_element_face_lists(nne[0], nsd)
    counter = 0

    for elem in range(ne):
        if elem > elem_indx[counter]:
            # Update faces for new element type
            counter += 1
            faces = _get_element_face_lists(nne[counter], nsd)

        # extract the element connectivity
        elem_conn = conn[elem]

        for face_name, face_local in faces.items():
            original_face = tuple(elem_conn[i] for i in face_local)  # for normals later
            sorted_face = tuple(sorted(original_face))  # for uniqueness
            if sorted_face in face_tracker:
                # Increment count
                face_tracker[sorted_face][0] += 1
            else:
                # Store count and original
                face_tracker[sorted_face] = [1, original_face]
    return face_tracker


def _extract_exterior(mesh: dict[str, Any]) -> dict[str, Any]:  # noqa
    """Find the nodes along the exterior for applying boundary conditions.

    Note that this extraction assumes the reservoir has 6 sides that do not have to be planar.
    """
    conn = mesh["conn"]
    nodes = mesh["nodes"]
    nne = mesh["nne"]
    ne = mesh["ne"]
    nsd = mesh["nsd"]
    elem_indx = mesh["elem_indx"]

    # Loop through all elements, collect a list of faces in global coordinates,
    # then find unique faces.
    face_tracker = _get_all_edges(conn, ne, nne, nsd, elem_indx)

    # sort through the faces and extract all unique faces
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
    elem_indx = mesh["elem_indx"]
    tol = 1.0e-6

    # Step 1: Initialize all elements to region 0
    regions_list = np.array([0] * ne)

    # Step 2: Calculate all element centroids
    centroids = np.empty((len(conn), nodes.shape[1]))
    start = 0
    for i in range(len(elem_indx)):
        end = elem_indx[i] + 1
        block_conn = np.array(conn[start:end])
        centroids[start:end, :] = np.mean(nodes[block_conn, :], 1)
        start = end + 1
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


def _get_fracture_geometry_2D(
    fracture_spec: dict[str, Any],
) -> tuple[npt.NDArray[np.float64], float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Extract geometric parameters from a 2D fracture specification."""
    center = fracture_spec["values"][0]  # [xc, yc, zc]
    center_2d = np.array([center[0], center[1]], dtype=np.float64)

    # Extents: for line/ellipse, first value is half-length
    half_length = fracture_spec["values"][1][0]

    # Angles: first value is theta (degrees) with respect to x-axis
    angle_deg = fracture_spec["values"][2][0]
    theta = np.deg2rad(angle_deg)

    # Unit vectors
    u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)  # Tangent direction
    n = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)  # Normal direction

    return center_2d, half_length, theta, u, n


def _compute_node_levels(
    nodes: npt.NDArray[np.float64],
    center_2d: npt.NDArray[np.float64],
    u: npt.NDArray[np.float64],
    n: npt.NDArray[np.float64],
    half_length: float,
    tol: float = 1e-8,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], set[int]]:
    """Compute the normal (phi) and tangential (psi) level sets,

    along with tangent projection (s) for all nodes, identifying indices
    that lie exactly on the fracture line segment.
    """
    # Signed level set
    phi = np.dot(nodes[:, :2] - center_2d, n)
    s = np.dot(nodes[:, :2] - center_2d, u)

    # Tangential level set (negative inside the crack, positive past the tips)
    psi = np.abs(s) - half_length

    # Identify nodes directly on the physical fracture segment (phi = 0 and psi <= 0)
    on_frac_mask = (np.abs(phi) < tol) & (psi <= tol)
    on_frac_nodes = set(np.where(on_frac_mask)[0])

    return phi, psi, s, on_frac_nodes


def _create_fracture_nodes_conn(  # noqa
    mesh: dict[str, Any], fracture: dict[str, Any], tol: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], list[int]]:

    conn = mesh["conn"]
    nodes = mesh["nodes"]
    nne = mesh["nne"]
    ne = mesh["ne"]
    nsd = mesh["nsd"]
    elem_indx = mesh["elem_indx"]

    # Geometric properties
    center_2d, half_length, theta, u, n = _get_fracture_geometry_2D(fracture)

    # Create a signed level set indicating nodal distance from the fracture
    phi, psi, s, on_frac_nodes = _compute_node_levels(nodes, center_2d, u, n, half_length, tol)
    # phi = normal level set
    # psi = tangential level set

    # Perturb near-zero values to avoid sign issues with float
    phi_perturbed = phi.copy()
    for idx in range(len(phi)):
        if idx not in on_frac_nodes and np.abs(phi_perturbed[idx]) < 1e-8:
            phi_perturbed[idx] = 1e-8 if phi_perturbed[idx] >= 0 else -1e-8

    # list of all boundary faces in the mesh. Each key is a unique boundary.
    boundary_list = _get_all_edges(conn, ne, nne, nsd, elem_indx)

    frac_nodes = np.empty((0, 2))  # Initialize

    for boundary in boundary_list:
        phi_a = phi_perturbed[boundary[0]]
        phi_b = phi_perturbed[boundary[1]]
        psi_a = psi[boundary[0]]
        psi_b = psi[boundary[1]]
        nodes_a = nodes[boundary[0], :]
        nodes_b = nodes[boundary[1], :]
        if np.sign(phi_a) != np.sign(phi_b):
            nodes_c = (phi_a * nodes_b - phi_b * nodes_a) / (phi_a - phi_b)
            # check tangential level set
            psi_c = psi_a - (psi_b - psi_a) * np.linalg.norm(nodes_a - nodes_c) / np.linalg.norm(
                nodes_a - nodes_b
            )
            if psi_c < 0:
                # If within bounds, add new node
                d_a = np.linalg.norm(nodes_c - nodes_a)
                d_b = np.linalg.norm(nodes_c - nodes_b)
                if d_a < tol:
                    new_node = nodes_a
                elif d_b < tol:
                    new_node = nodes_b
                else:
                    new_node = nodes_c

                frac_nodes = np.vstack((frac_nodes, new_node))

    # Now check tip elements
    tips = [-1.0, 1.0]
    tip_nodes = []  # Track which ones are tip nodes for special treatment
    for i in tips:
        tip_coords = center_2d + half_length * u * i

        # Check if tip_coords already exists
        d = np.linalg.norm(frac_nodes - tip_coords, axis=1)
        if np.any(d < tol):
            # Node already exists, track node
            tip_nodes.append(int(np.argmin(d)))
            continue

        # Check if tip_coords is outside the bounds
        hull = sp.spatial.ConvexHull(nodes)
        A, b = (
            hull.equations[:, :-1],
            hull.equations[:, -1:],
        )  # extract the equations defining the 0-level set
        # The level set defines the hull such that all points Ax + b = 0 form the shell of the hull.
        # If Ax + b < 0, then the point is within the polyhedron.
        node_in_region = np.all(tip_coords @ A.T + b.T < tol)
        if not node_in_region:
            # Tip is outside the domain
            continue

        # Otherwise, add new node
        frac_nodes = np.vstack((frac_nodes, tip_coords))
        tip_nodes.append(int(np.size(frac_nodes, 0)) - 1)

    # Now create connectivity
    indx = np.argsort(frac_nodes @ u)
    frac_conn = np.vstack((indx[0:-1], indx[1:])).T
    return frac_nodes, frac_conn, tip_nodes


def _create_fracture_mesh(config_mesh: dict[str, Any], mesh: dict[str, Any]) -> dict[str, Any]:
    """Create mesh cutting for fracture elements."""
    fracture_specs = config_mesh["fractures"]
    nsd = config_mesh["dimensions"]
    nodes = mesh["nodes"]

    if nsd != 2:
        print("3D fractures not yet supported.")
        return mesh
    # Initialize
    # frac_matrix_conn = []  # List of fracture-matrix connectivity [[top, bottom], [top, bottom],...]
    # frac_frac_conn = []  # 1D fracture elements [node 1, node 2; node 2, node 3; ...]
    #  conn_new = conn.copy()
    # nodes_new = nodes.copy()

    # Set node distance tolerance based on domain size

    d_max = np.amax(nodes, 0)
    L = (d_max.prod()) ** (1 / nsd)
    tol = L * 0.005

    # Loop through each fracture specification
    for fracture in fracture_specs:
        # Create fracture nodes and connectivity
        frac_nodes, frac_conn, tip_nodes = _create_fracture_nodes_conn(mesh, fracture, tol)

        # Loop through elements and check whether the new nodes lie within each element, then split
        # mesh = _split_elements(mesh, frac_nodes, frac_conn, tip_nodes)

        # Loop through all elements
        # If element is cut (nodal level set values do not have the same sign)
        # cut element:
        # Case 1: fully cut element
        # Lookup geometric cut cases for Q4 and T3 elements for new mapping.
        # Add three new nodes for top, fracture, and bottom to nodes.
        # Update connectivity of existing elements, and add new element
        # connectivity to the list, including fracture elements
        # store top and bottom element connectivity
        # Case 2: boundary cut element
        # subdivide the element into small elements sharing the tip node
        # Case 3: boundary-boundary element
        # new nodes needed if they don't already exist
        # no new elements required
        # Case 4: no element crossing

    return mesh


def _get_element_type(nsd: int, nne: int) -> tuple[int, str]:
    etyp = ""
    nq = 0
    # Define number of quadrature points
    match (nsd, nne):
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
    return nq, etyp


def _exact_key_mesh_values(mesh: dict[str, Any]) -> dict[str, Any]:
    nodes = mesh["nodes"]
    conn = mesh["conn"]

    # Extract key values
    mesh["nn"] = np.shape(nodes)[0]  # number of nodes
    nsd = np.shape(nodes)[1]  # number of space dimensions
    ne = len(conn)  # number of elements
    mesh["nsd"] = nsd
    mesh["ne"] = ne

    mesh["nne"] = [len(conn[0])]  # number of nodes per element - list
    mesh["elem_indx"] = []  # indicates the index of the last element of each element type
    nq, etype = _get_element_type(nsd, len(conn[0]))
    mesh["n_quadrature"] = [nq]
    mesh["element_type"] = [etype]

    # element type index
    counter = 0
    for e in range(1, len(conn)):
        nne = len(conn[e])
        if nne != mesh["nne"][counter]:
            mesh["nne"].append(nne)
            mesh["elem_indx"].append(e - 1)
            nq, etype = _get_element_type(nsd, nne)
            mesh["n_quadrature"].append(nq)
            mesh["element_type"].append(etype)
            counter += 1
    mesh["elem_indx"].append(ne - 1)  # add final element to end

    return mesh


def load_mesh(config_mesh: dict[str, Any], verbose: bool) -> dict[str, Any]:
    """Load mesh and define mesh structures."""
    config_path = Path(config_mesh["config_dir"]) / config_mesh["file"]

    nsd = config_mesh["dimensions"]  # number of space dimensions
    # Load mesh
    [nodes, conn] = load_Gmsh(str(config_path), nsd)

    if nodes is None or conn is None:
        raise RuntimeError(f"Failed to load mesh from {config_path}")

    mesh: dict[str, Any] = {
        "nodes": nodes,
        "conn": conn,
    }

    mesh = _exact_key_mesh_values(mesh)
    if "fractures" in config_mesh:
        mesh = _create_fracture_mesh(config_mesh, mesh)

    # Import regions
    mesh = _import_regions(config_mesh, mesh)

    # Extract sets of exterior nodes
    mesh = _extract_exterior(mesh)

    if verbose:
        # Join multiple element types with commas
        elem_types_str = ", ".join(str(et) for et in mesh["element_type"])

        # Compute the quadrature string breakdown for each type in the list
        quad_pts_str = ", ".join(f"{nq}^{nsd} = {nq**nsd}" for nq in mesh["n_quadrature"])

        print("----- Mesh statistics")
        print(f"  File            : {config_mesh['file']}")
        print(f"  Element type    : {elem_types_str}")
        print(f"  Dimensions      : {nsd}D")
        print(f"  Nodes           : {mesh['nn']}")
        print(f"  Elements        : {mesh['ne']}")
        print(f"  Quadrature pts  : {quad_pts_str}\n")

    return mesh
