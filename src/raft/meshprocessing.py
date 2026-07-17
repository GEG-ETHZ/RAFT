"""Import mesh, assign regions, extract relevant mesh properties."""

from dataclasses import dataclass
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
    phi[on_frac_mask] = 0.0
    on_frac_nodes = set(np.where(on_frac_mask)[0])

    return phi, psi, s, on_frac_nodes


@dataclass
class FractureLevelSet:
    """Geometry and nodal level-set data for a single 2D fracture.

    Computed once per fracture via `_build_fracture_level_set` and shared across the
    mesh-cutting helpers, instead of each helper recomputing it independently.
    """

    # Geometric properties (from _get_fracture_geometry_2D)
    center_2d: npt.NDArray[np.float64]
    half_length: float
    theta: float
    u: npt.NDArray[np.float64]  # tangent direction
    n: npt.NDArray[np.float64]  # normal direction

    # Nodal distance tolerance used to build the level set
    tol: float

    # Nodal level sets (from _compute_node_levels)
    phi: npt.NDArray[np.float64]  # normal (signed) level set
    psi: npt.NDArray[np.float64]  # tangential level set
    s: npt.NDArray[np.float64]  # tangential projection
    on_frac_nodes: set[int]  # nodes lying exactly on the fracture segment

    phi_sign: npt.NDArray[np.float64]  # sign of phi


def _build_fracture_level_set(
    nodes: npt.NDArray[np.float64], fracture_spec: dict[str, Any], tol: float
) -> FractureLevelSet:
    """Compute fracture geometry and nodal level sets once for reuse across mesh-cutting steps."""
    # Geometric properties
    center_2d, half_length, theta, u, n = _get_fracture_geometry_2D(fracture_spec)

    # Create a signed level set indicating nodal distance from the fracture
    phi, psi, s, on_frac_nodes = _compute_node_levels(nodes, center_2d, u, n, half_length, tol)
    # phi = normal level set
    # psi = tangential level set

    phi_sign = np.sign(phi)

    return FractureLevelSet(
        center_2d=center_2d,
        half_length=half_length,
        theta=theta,
        u=u,
        n=n,
        tol=tol,
        phi=phi,
        psi=psi,
        s=s,
        on_frac_nodes=on_frac_nodes,
        phi_sign=phi_sign,
    )


def _create_fracture_nodes_conn(  # noqa
    mesh: dict[str, Any], frac_ls: FractureLevelSet
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], list[int]]:
    """Create fracture nodes and connectivity."""
    conn = mesh["conn"]
    nodes = mesh["nodes"]
    nne = mesh["nne"]
    ne = mesh["ne"]
    nsd = mesh["nsd"]
    elem_indx = mesh["elem_indx"]

    center_2d = frac_ls.center_2d
    half_length = frac_ls.half_length
    u = frac_ls.u
    tol = frac_ls.tol
    psi = frac_ls.psi
    phi = frac_ls.phi

    # list of all boundary faces in the mesh. Each key is a unique boundary.
    boundary_list = _get_all_edges(conn, ne, nne, nsd, elem_indx)

    frac_nodes = np.empty((0, 2))  # Initialize

    for boundary in boundary_list:
        phi_a = phi[boundary[0]]
        phi_b = phi[boundary[1]]
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
        # Check if tip_coords already exists in mesh
        d = np.linalg.norm(nodes - tip_coords, axis=1)
        indx = np.argmin(d)
        if d[indx] < tol:
            tip_coords = nodes[indx, :]

        # Check if fracture node at tip_coords already exists
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


def _sort_boundary_coords(
    node_labels: list[int] | npt.NDArray[np.int64],
    nodes: npt.NDArray[np.float64],
) -> list[int]:
    """Sorts a list of node labels in counter-clockwise (CCW) order
    based on their angles relative to their collective centroid.
    """
    # Quick return if sorting is trivial or empty
    if len(node_labels) <= 1:
        return list(node_labels)

    # Convert input labels to a NumPy array for fast vector indexing
    labels_arr = np.asarray(node_labels, dtype=np.int64)

    # Extract 2D coordinates for the given subset of nodes
    coords = nodes[labels_arr]

    # Calculate the collective centroid of the subset
    centroid = np.mean(coords, axis=0)

    # Translate coordinates to be relative to the centroid
    rel_coords = coords - centroid

    # Compute polar angles in the range [-pi, pi]
    # np.arctan2(y, x) handles quadrant signs automatically
    angles = np.arctan2(rel_coords[:, 1], rel_coords[:, 0])

    # Sort the original labels based on ascending angle values
    sorted_indices = np.argsort(angles)
    sorted_labels = labels_arr[sorted_indices]

    return sorted_labels.tolist()


def _split_elements(  # noqa
    mesh: dict[str, Any],
    frac_ls: FractureLevelSet,
    frac_nodes: npt.NDArray[np.float64],
    frac_conn: npt.NDArray[np.int64],
    tip_nodes: list[int],
) -> dict[str, Any]:
    """Loop through the fracture elements, find the cut elements, and split them accordingly."""
    conn = mesh["conn"]
    nodes = mesh["nodes"]
    nne = mesh["nne"]  # noqa
    ne = mesh["ne"]
    nn = mesh["nn"]
    nsd = mesh["nsd"]
    elem_indx = mesh["elem_indx"]  # noqa

    phi_sign = frac_ls.phi_sign
    psi = frac_ls.psi
    tol = frac_ls.tol

    ne_f = np.size(frac_conn, 0)

    new_nodes = []  # noqa
    new_elem = []
    new_node_index = nn

    # ---- Create a sublist for all elements with a change in sign
    matrix_elements_to_cut = []
    for e in range(ne):
        enodes = conn[e]
        signs = phi_sign[enodes]

        # A true interior cut requires both positive and negative signs.
        # This excludes all vertex-touching and edge-aligned cases.
        if np.any(signs > 0) and np.any(signs < 0):  # noqa
            # The element crosses the normal level set
            if np.any(psi[enodes] < tol):
                # The element crosses or is in the tangential level set
                matrix_elements_to_cut.append(e)

    elements_to_delete = matrix_elements_to_cut.copy()  # noqa # Keep a list to delete these elements at the end
    matrix_elements_to_cut = np.array(matrix_elements_to_cut)

    # ---- Sort the elements by distance from the fracture element to find
    # the corresponding matrix parent element and prevent excessive looping
    centroids = np.empty((len(matrix_elements_to_cut), nodes.shape[1]))
    for i in range(len(matrix_elements_to_cut)):
        enodes = conn[matrix_elements_to_cut[i]]
        centroids[i, :] = np.mean(nodes[enodes, :], 0)

    # ---- Loop through all fracture elements
    for e_f in range(ne_f):
        f_nodes = frac_conn[e_f, :]
        f_coords = frac_nodes[f_nodes, :]
        f_center = np.mean(f_coords, 0)
        tip_element = np.any(f_nodes == tip_nodes)

        d = np.linalg.norm(centroids - f_center, axis=1)
        indx = np.argsort(d)

        for i in indx:
            e_m = matrix_elements_to_cut[i]
            m_nodes = conn[e_m]
            m_coords = nodes[m_nodes]
            hull = sp.spatial.ConvexHull(m_coords)
            A, b = (
                hull.equations[:, :-1],
                hull.equations[:, -1:],
            )  # extract the equations defining the 0-level set
            # The level set defines the hull such that all points Ax + b = 0 form the shell of the hull.
            # If Ax + b < 0, then the point is within the polyhedron.
            elem_in_region = np.all(f_coords @ A.T + b.T < tol)
            if elem_in_region:
                # The fracture element is within the matrix element domain,
                # this is the element to cut.

                # Pop it out of the list to check next time
                matrix_elements_to_cut = np.delete(matrix_elements_to_cut, i)
                centroids = np.delete(centroids, i, 0)
                # Continue on to cutting
                break

        # ---- Create new nodes
        # If the node is already on the fracture path, we're going to
        # replace it with a new node
        m_nodes_orig = m_nodes.copy()
        m_nodes = [node for node in m_nodes if phi_sign[node] != 0.0]
        shifted_nodes = []
        frac_normal = frac_ls.n
        shift_dist = 0.5 * tol if tip_element else 0.0

        for i in range(len(f_nodes)):
            if np.any(f_nodes[i] == tip_nodes):
                # For tip nodes, add one node
                nodes = np.vstack((nodes, f_coords[i, :]))
                m_nodes.append(new_node_index)
                new_node_index += 1
                phi_sign = np.hstack((phi_sign, 0.0))
                psi = np.hstack((psi, 0.0))
            else:
                # Positive side shift
                pos_coord = f_coords[i, :] + shift_dist * frac_normal
                nodes = np.vstack((nodes, pos_coord))
                m_nodes.append(new_node_index)
                shifted_nodes.append((new_node_index, f_coords[i, :]))  # Store to revert
                new_node_index += 1
                phi_sign = np.hstack((phi_sign, 1.0))
                psi = np.hstack((psi, -1.0))

                # Negative side shift
                neg_coord = f_coords[i, :] - shift_dist * frac_normal
                nodes = np.vstack((nodes, neg_coord))
                m_nodes.append(new_node_index)
                shifted_nodes.append((new_node_index, f_coords[i, :]))  # Store to revert
                new_node_index += 1
                phi_sign = np.hstack((phi_sign, -1.0))
                psi = np.hstack((psi, -1.0))

        # ---- Triangulate by grouping nodes according to sign
        m_node_signs = phi_sign[m_nodes]
        # Create node groups for triangulation grouped by side
        m_nodes_pos = [node for node, sign in zip(m_nodes, m_node_signs) if sign >= 0]
        m_nodes_neg = [node for node, sign in zip(m_nodes, m_node_signs) if sign <= 0]
        m_node_groups = [m_nodes_pos, m_nodes_neg]
        if tip_element:
            # Create an additional group to cover the region beyond the tip
            # that the fracture does not actually cut through: the tip node
            # plus the nodes of the parent element's far face -- the face
            # that crosses the normal level set (phi = 0, i.e. has nodes on
            # both sides) but whose nodes lie entirely past the tip along
            # the fracture (psi >= tol, so the crack never reaches them).
            tip_group = []
            # add the tip node
            tip_local_idx = np.where(m_node_signs == 0)[0]
            tip_group.extend(np.array(m_nodes)[tip_local_idx].tolist())

            # Add the nodes from the opposing face cut by the normal level set
            faces = _get_element_face_lists(len(m_nodes_orig), nsd)
            for face_local in faces.values():
                face_nodes = [m_nodes_orig[i] for i in face_local]
                crosses_phi = np.any(phi_sign[face_nodes] > 0) and np.any(phi_sign[face_nodes] < 0)
                beyond_tip = np.all(psi[face_nodes] >= tol)
                if crosses_phi and beyond_tip:
                    tip_group.extend(n for n in face_nodes if n not in tip_group)

            if len(tip_group) > 1:
                m_node_groups.append(tip_group)

        # Perform Delaunay triangulation
        for group in m_node_groups:
            if len(group) < 5:
                # Keep as a Q4/T3, sort to ensure new nodes are CCW ordered
                global_conn = _sort_boundary_coords(group, nodes[group, :])

            else:
                # Perform triangulation
                tri = sp.spatial.Delaunay(nodes[group, :])
                group = np.array(group)
                local_conn = np.array(tri.simplices.tolist())
                global_conn = group[local_conn]

            new_elem.append(global_conn)

        # ---- Restore shifted nodes back to their original degenerate coordinates
        for node_idx, original_coord in shifted_nodes:
            nodes[node_idx, :] = original_coord

    # ---- Loop through all nodes and remove duplicates where phi_sign and nodal coordinates are equal.
    # Also remove elements where phi_sign = 0 that are not tip nodes.
    # Create a mapping of old node numbering to new node numbering.
    # Prioritize keeping the later numbered node.

    # ---- Update connectivity matrices with new elements and new node numberings.

    return mesh


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
        # Compute fracture geometry and nodal level sets once, shared by both helpers below
        frac_ls = _build_fracture_level_set(nodes, fracture, tol)

        # Create fracture nodes and connectivity
        frac_nodes, frac_conn, tip_nodes = _create_fracture_nodes_conn(mesh, frac_ls)

        # Loop through elements and check whether the new nodes lie within each element, then split
        mesh = _split_elements(mesh, frac_ls, frac_nodes, frac_conn, tip_nodes)

        # Loop through elements and map top and bottom connectivity.

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


def _extract_key_mesh_values(mesh: dict[str, Any]) -> dict[str, Any]:
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

    mesh = _extract_key_mesh_values(mesh)
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
