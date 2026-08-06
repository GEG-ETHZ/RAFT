"""Import mesh, assign regions, extract relevant mesh properties."""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy as sp
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.tri import Triangulation


def _element_colors(n_elems: int, cmap_name: str = "tab20") -> npt.NDArray[np.float64]:
    """Distinct-ish RGBA color per element, cycling through a qualitative colormap.

    Used (with alpha) so that overlapping elements blend into a visibly different
    color instead of one silently occluding the other.
    """
    cmap = plt.get_cmap(cmap_name)
    return cmap(np.arange(n_elems) % cmap.N)


def _label_element_ids(
    ax: Axes,
    nodes: npt.NDArray[np.float64],
    conn: list[list[int]],
    prefix: str = "E",
) -> None:
    for elem_id, elem_nodes in enumerate(conn):
        centroid = np.mean(nodes[elem_nodes], axis=0)
        ax.text(
            centroid[0],
            centroid[1],
            f"{prefix}{elem_id}",
            color="black",
            fontweight="bold",
            ha="center",
            va="center",
            fontsize=6,
            zorder=4,
            bbox=dict(  # noqa
                facecolor="white",
                alpha=0.8,
                edgecolor="none",
                boxstyle="round,pad=0.2",
            ),
        )


def debugging_mesh_plot(
    nodes: npt.NDArray[np.float64],
    conn: list[list[int]],
    nodal_values: npt.NDArray[np.float64] | None = None,
    plot_ids: bool = False,
) -> None:
    """Plot and label mesh elements and nodes, optionally showing interpolated scalar values."""
    fig, ax = plt.subplots()

    # 1. Handle optional nodal scalar field
    if nodal_values is not None:
        # Fan triangulation around the first vertex for smooth Gouraud shading
        triangles = [[elem[0], elem[i], elem[i + 1]] for elem in conn for i in range(1, len(elem) - 1)]

        triangulation = Triangulation(nodes[:, 0], nodes[:, 1], np.array(triangles))
        tc = ax.tripcolor(
            triangulation,
            nodal_values,
            shading="gouraud",
            cmap="viridis",
        )
        cbar = fig.colorbar(tc, ax=ax)
        cbar.set_label("Nodal Scalar Value")

        facecolor: str | npt.NDArray[np.float64] = "none"
        alpha = 1.0
        plot_title = "Interpolated Nodal Values Mesh Plot"
    else:
        facecolor = _element_colors(len(conn))
        alpha = 0.5
        plot_title = "Mesh Plot"

    # 2. Overlay element borders
    verts = [nodes[elem] for elem in conn]
    mesh = PolyCollection(
        verts,
        closed=True,
        edgecolor="black",
        facecolor=facecolor,
        alpha=alpha,
        linewidth=1.2,
    )
    ax.add_collection(mesh)

    # 3. Scatter node locations
    ax.scatter(nodes[:, 0], nodes[:, 1], color="red", zorder=3, s=2)

    # 4. Add Node IDs (and scalar values if available)
    if plot_ids:
        for i, (x, y) in enumerate(nodes):
            label = f"{i} ({nodal_values[i]:.1f})" if nodal_values is not None else f"{i}"
            ax.text(
                x + 0.05,
                y + 0.05,
                label,
                color="white" if nodal_values is not None else "darkred",
                fontsize=6,
                zorder=4,
                bbox=dict(  # noqa
                    facecolor="black" if nodal_values is not None else "white",
                    alpha=0.6,
                    pad=1,
                    edgecolor="none",
                ),
            )

    # 5. Add Element IDs at centroids
    if plot_ids:
        _label_element_ids(ax, nodes, conn, prefix="E")

    # 6. Final adjustments
    ax.autoscale_view()
    ax.set_aspect("equal")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.title(plot_title)
    plt.show()


def debugging_mesh_plot_matrix_and_fracture(
    matrix_nodes: npt.NDArray[np.float64],
    matrix_conn: list[list[int]],
    frac_nodes: npt.NDArray[np.float64],
    frac_conn: npt.NDArray[np.int64] | list[list[int]],
    plot_ids: bool = False,
) -> None:
    """Plot the matrix and fracture meshes together, each element individually colored.

    Matrix elements are filled polygons; fracture elements are drawn as line segments
    on top. Overlaps within either mesh blend into a visibly different color, and
    fracture segments that don't sit on a matrix element boundary stand out clearly.
    """
    fig, ax = plt.subplots()

    matrix_verts = [matrix_nodes[elem] for elem in matrix_conn]
    matrix_mesh = PolyCollection(
        matrix_verts,
        closed=True,
        edgecolor="black",
        facecolor=_element_colors(len(matrix_conn), "tab20"),
        alpha=0.5,
        linewidth=1.0,
        zorder=1,
    )
    ax.add_collection(matrix_mesh)

    frac_segments = [frac_nodes[elem] for elem in frac_conn]
    frac_mesh = LineCollection(
        frac_segments,
        colors=_element_colors(len(frac_segments), "gist_rainbow"),
        linewidth=1.5,
        zorder=3,
    )
    ax.add_collection(frac_mesh)

    ax.scatter(matrix_nodes[:, 0], matrix_nodes[:, 1], color="black", zorder=2, s=2, label="matrix nodes")
    ax.scatter(frac_nodes[:, 0], frac_nodes[:, 1], color="red", zorder=4, s=8, label="fracture nodes")

    if plot_ids:
        _label_element_ids(ax, matrix_nodes, matrix_conn, prefix="M")
        _label_element_ids(ax, frac_nodes, list(frac_conn), prefix="F")

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=6)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.title("Matrix + Fracture Mesh Plot")
    plt.show()


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
        print(f"Error: File {meshfile} not found.")
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

    # Remove fracture faces from exterior sets
    if "fracture_mesh" in mesh:
        matrix_conn = mesh["fracture_mesh"]["matrix_conn"]
        # Find the fracture node faces by sorted node tuple so they can
        # be matched against what's already in `boundaries`.
        fracture_faces: set[tuple[int, ...]] = set()
        faces_by_nne = {n: _get_element_face_lists(n, nsd) for n in set(nne)}

        for conn_fm in matrix_conn:
            for elem in conn_fm:
                if elem is None:
                    continue
                elem_conn = conn[elem]
                for face_local in faces_by_nne[len(elem_conn)].values():
                    sorted_face = tuple(sorted(elem_conn[i] for i in face_local))
                    if face_tracker.get(sorted_face, [0])[0] == 1:
                        fracture_faces.add(sorted_face)

        # Strip fracture faces from each boundary's connectivity, then rebuild
        # the node set from whatever faces remain.
        for key in ("x", "y", "z"):
            for side in ("min", "max"):
                conn_key = f"{key}_{side}_conn"
                if conn_key not in boundaries:
                    continue
                boundaries[conn_key] = [
                    face for face in boundaries[conn_key] if tuple(sorted(face)) not in fracture_faces
                ]
                boundaries[f"{key}_{side}"] = {n for face in boundaries[conn_key] for n in face}

    # Combine boundary sets
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
        start = end
    if nsd == 2:
        centroids = np.hstack((centroids, np.zeros((ne, 1))))

    # Step 3: Assign regions in order of specification
    supported_shapes = ["polyhedron", "sphere", "cylinder"]

    for region_shape in config_regions:
        shape_type = region_shape.get("shape")
        values = region_shape["values"]
        region_number = region_shape["region_number"]

        if shape_type == "polyhedron":
            values = np.array(values)  # turn value into array of points
            nsd_spec = np.size(values, axis=1)
            hull = sp.spatial.ConvexHull(values)  # calculate the convex shape formed by the points
            A, b = (
                hull.equations[:, :-1],
                hull.equations[:, -1:],
            )  # extract the equations defining the 0-level set
            # The level set defines the hull such that all points Ax + b = 0 form the shell of the hull.
            # If Ax + b < 0, then the point is within the polyhedron.
            elem_in_region = np.all(centroids[:, :nsd_spec] @ A.T + b.T < tol, axis=1)

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

    # Tangential level set (negative inside the crack, positive past the tips)
    s = np.dot(nodes[:, :2] - center_2d, u)
    psi = np.abs(s) - half_length

    # Snap any node numerically collinear with the (infinite) fracture line to phi = 0
    collinear_mask = np.all((np.abs(phi) < tol, psi < tol), axis=0)
    phi[collinear_mask] = 0.0

    # Nodes directly on the physical fracture segment (phi = 0 and psi <= 0) -- a
    # narrower set than collinear_mask, since it excludes collinear nodes beyond a tip.
    on_frac_mask = collinear_mask & (psi <= tol)
    on_frac_nodes = set(np.where(on_frac_mask)[0])

    return phi, psi, s, on_frac_nodes


# Geometry of one already-processed fracture, kept around so any node -- including
# ones created much later, mid-cut, for a different fracture -- can have its sign
# w.r.t. this fracture evaluated directly from coordinates. (center_2d, n, tol)
FracGeom = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]


def _evaluate_historical_phi_sign(
    coords: npt.NDArray[np.float64], frac_geoms: list[FracGeom]
) -> npt.NDArray[np.float64]:
    """phi_sign of one or more nodes against every previously-processed fracture.

    phi is affine in nodal coordinates (a signed distance to an infinite line), so this
    evaluates directly from a node's coordinates -- no interpolation or topology
    tracking needed, even for a node created moments ago mid-cut. Returns an
    (n_nodes, len(frac_geoms)) array; n_nodes == 1 for a single new node.
    """
    coords = np.atleast_2d(coords)
    if not frac_geoms:
        return np.empty((coords.shape[0], 0))
    cols = []
    for center_2d, n, tol in frac_geoms:
        phi = np.dot(coords[:, :2] - center_2d, n)
        phi[np.abs(phi) < tol] = 0.0
        cols.append(np.sign(phi))
    return np.column_stack(cols)


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

    phi_sign: npt.NDArray[np.float64]  # sign of phi for the current fracture
    # (center_2d, n, tol) for every fracture processed before this one, in order.
    frac_geoms: list[FracGeom]
    # phi_sign of every current node against every fracture processed so far,
    # NOT including the current one -- shape (nn, len(frac_geoms)).
    phi_sign_history: npt.NDArray[np.float64]


def _build_fracture_level_set(
    nodes: npt.NDArray[np.float64],
    fracture_spec: dict[str, Any],
    phi_sign_history: npt.NDArray[np.float64],
    frac_geoms: list[FracGeom],
    tol: float,
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
        frac_geoms=frac_geoms,
        phi_sign_history=phi_sign_history,
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
    phi = frac_ls.phi
    psi = frac_ls.psi

    # list of all boundary faces in the mesh. Each key is a unique boundary.
    boundary_list = _get_all_edges(conn, ne, nne, nsd, elem_indx)

    frac_nodes = np.empty((0, 2))  # Initialize

    for boundary in boundary_list:
        phi_a = phi[boundary[0]]
        phi_b = phi[boundary[1]]
        nodes_a = nodes[boundary[0], :]
        nodes_b = nodes[boundary[1], :]
        if np.sign(phi_a) != np.sign(phi_b):
            nodes_c = (phi_a * nodes_b - phi_b * nodes_a) / (phi_a - phi_b)
            # check tangential level set
            s_c = np.dot(nodes_c - center_2d, u)
            psi_c = np.abs(s_c) - half_length
            if psi_c < tol:
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

    frac_nodes = np.unique(np.round(frac_nodes, 9), axis=0)

    # Now check tip elements
    tips = [-1.0, 1.0]
    tip_nodes = []  # Track which ones are tip nodes for special treatment
    for i in tips:
        tip_coords = center_2d + half_length * u * i
        # Check if tip_coords already exists in mesh
        d = np.linalg.norm(nodes - tip_coords, axis=1)
        nearest_indx = int(np.argmin(d))
        if phi[nearest_indx] == 0.0 and psi[nearest_indx] < tol:
            tip_coords = nodes[nearest_indx, :]

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
        node_in_region = np.all(tip_coords @ A.T + b.T < 1e-8)
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
    coords: npt.NDArray[np.float64],
) -> list[int]:
    """Sorts a list of node labels in counter-clockwise (CCW) order
    based on their angles relative to their collective centroid.
    """
    # Quick return if sorting is trivial or empty
    if len(node_labels) <= 1:
        return list(node_labels)

    # Convert input labels to a NumPy array for fast vector indexing
    labels_arr = np.asarray(node_labels, dtype=np.int64)

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

    return cast(list[int], sorted_labels.tolist())


@dataclass
class _CutState:
    """Mutable node / level-set arrays threaded through the element-cutting helpers below.

    Cutting a matrix element appends new nodes to `nodes` and matching entries to
    `phi_sign`/`psi`, and advances `new_node_index`. Helpers mutate this in place
    (rather than returning and reassigning four arrays at every call site) since all
    of it is single, growing mesh state shared across one call to `_split_elements`.
    """

    nodes: npt.NDArray[np.float64]
    phi: npt.NDArray[np.float64]
    phi_sign: npt.NDArray[np.float64]
    # phi_sign against every fracture processed before the current one -- (nn, k).
    # Grown by one row alongside `nodes`/`phi_sign` for each newly created node.
    phi_sign_history: npt.NDArray[np.float64]
    frac_geoms: list[FracGeom]
    psi: npt.NDArray[np.float64]
    new_node_index: int


def _find_cut_elements(
    conn: list[list[int]],
    ne: int,
    phi: npt.NDArray[np.float64],
    phi_sign: npt.NDArray[np.float64],
    psi: npt.NDArray[np.float64],
    tol: float,
) -> list[int]:
    """Find matrix elements with a true interior cut: nodes of both signs, within the
    tangential extent of the fracture. Excludes vertex-touching and edge-aligned cases.

    A node sitting on the fracture's infinite line but beyond the tip (psi >= tol) never
    gets its `phi` snapped to exactly 0 by the level-set setup, so skip those nodes.
    """
    matrix_elements_to_cut = []
    for e in range(ne):
        enodes = conn[e]
        on_line = np.abs(phi[enodes]) < tol
        signs = phi_sign[enodes][~on_line]

        if np.any(signs > 0) and np.any(signs < 0):  # noqa
            # The element crosses the normal level set
            if np.any(psi[enodes] < tol):
                # The element crosses or is in the tangential level set
                matrix_elements_to_cut.append(e)
    return matrix_elements_to_cut


def _find_collinear_tip_elements(  # noqa
    conn: list[list[int]],
    nsd: int,
    phi_sign: npt.NDArray[np.float64],
    psi: npt.NDArray[np.float64],
    frac_ls: FractureLevelSet,
    tip_nodes: list[int],
    frac_nodes: npt.NDArray[np.float64],
    nodes: npt.NDArray[np.float64],
) -> list[tuple[int, int, int, int]]:
    """Find elements with a boundary edge collinear with the fracture line whose two
    endpoints straddle a crack tip tangentially (one within the fracture's tangential
    extent, one beyond the tip).

    Returns (element_index, near_node, far_node, tip_index) tuples: near_node lies
    within the fracture's tangential extent, far_node lies beyond the tip, and
    tip_index indexes into `tip_nodes`/`frac_nodes` for the matching crack tip.

    Skips a match if the tip itself already coincides with an existing matrix node --
    that case is already handled by the fracture element loop's own tip cut
    (`_cut_matrix_element`/`_group_split_nodes`), and inserting a second, independent
    tip node here would create a coincident duplicate that can leave a degenerate,
    zero-area element behind once nodes are deduplicated.
    """
    zero_nodes = set(np.flatnonzero(phi_sign == 0.0).tolist())
    if not zero_nodes:
        return []

    # Determine which elements have at least one node already classified as on the fracture
    candidate_elems = [e for e, enodes in enumerate(conn) if any(n in zero_nodes for n in enodes)]
    if not candidate_elems:
        return []

    center_2d = frac_ls.center_2d
    u = frac_ls.u
    s = frac_ls.s
    tol = frac_ls.tol
    phi = frac_ls.phi
    tip_s = {idx: float(np.dot(frac_nodes[idx] - center_2d, u)) for idx in tip_nodes}

    found: list[tuple[int, int, int, int]] = []
    for e in candidate_elems:
        enodes = conn[e]
        faces = _get_element_face_lists(len(enodes), nsd)
        for face_local in faces.values():
            a, b = enodes[face_local[0]], enodes[face_local[1]]
            if a in zero_nodes:
                near_candidate, other = a, b
            elif b in zero_nodes:
                near_candidate, other = b, a
            else:
                continue  # neither endpoint is confirmed on the fracture segment

            if abs(phi[other]) >= tol:
                continue  # other endpoint isn't collinear with the (infinite) fracture line

            if np.sign(psi[near_candidate]) == np.sign(psi[other]):
                continue  # both collinear, but doesn't straddle a tip tangentially

            near, far = (
                (near_candidate, other) if psi[near_candidate] < psi[other] else (other, near_candidate)
            )
            if abs(psi[near]) < tol:
                continue  # near node is essentially at the tip already, as a node

            tip_idx = min(tip_s, key=lambda k: abs(tip_s[k] - s[near]))

            tip_coord = frac_nodes[tip_idx, :]
            dist_to_existing = np.linalg.norm(nodes - tip_coord, axis=1)
            if np.any(dist_to_existing < tol):
                continue  # tip already lands on an existing matrix node -- handled elsewhere

            found.append((e, near, far, tip_idx))
    return found


def _cut_collinear_tip_elements(
    state: _CutState,
    conn: list[list[int]],
    matches: list[tuple[int, int, int, int]],
    frac_nodes: npt.NDArray[np.float64],
    tip_nodes: list[int],
    frac_normal: npt.NDArray[np.float64],
    tol: float,
) -> tuple[list[list[int]], list[int]]:
    """Cut the neighbouring collinear elements by inserting the new node at the edge."""
    new_elem: list[list[int]] = []
    consumed_elems: list[int] = []

    for e, near, far, tip_idx in matches:
        m_nodes_orig = conn[e]

        f_nodes_tip = np.array([tip_idx])
        f_coords_tip = frac_nodes[[tip_idx], :]
        m_nodes = [node for node in m_nodes_orig if state.phi_sign[node] != 0.0]
        m_nodes, _ = _insert_split_nodes(
            state, m_nodes, f_nodes_tip, f_coords_tip, tip_nodes, True, frac_normal, tol
        )
        new_tip_node = m_nodes[-1]  # node _insert_split_nodes just appended for the tip

        # Splice the tip node into the element's existing CCW boundary order, directly
        # between near and far (they're adjacent by construction, since they form one
        # of the element's own edges).
        ordered = list(m_nodes_orig)
        idx_near = ordered.index(near)
        insert_at = idx_near + 1 if ordered[(idx_near + 1) % len(ordered)] == far else ordered.index(far) + 1
        ordered.insert(insert_at, new_tip_node)

        new_elem.extend(_triangulate_node_groups([ordered], state.nodes))

        consumed_elems.append(e)

    return new_elem, consumed_elems


def _compute_element_centroids(
    elem_indx: npt.NDArray[np.int64], conn: list[list[int]], nodes: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute centroids of the given elements, used to match fracture elements to their parent."""
    centroids = np.empty((len(elem_indx), nodes.shape[1]))
    for i in range(len(elem_indx)):
        enodes = conn[elem_indx[i]]
        centroids[i, :] = np.mean(nodes[enodes, :], 0)
    return centroids


def _find_parent_element(
    f_coords: npt.NDArray[np.float64],
    matrix_elements_to_cut_arr: npt.NDArray[np.int64],
    centroids: npt.NDArray[np.float64],
    conn: list[list[int]],
    nodes: npt.NDArray[np.float64],
) -> tuple[int | None, npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Find the matrix element containing a fracture element, checking nearest-centroid first.

    Returns the parent element index (or None if the fracture element lies along a
    boundary and has no parent to cut), plus the candidate arrays with the match popped out.
    """
    f_center = np.mean(f_coords, 0)
    d = np.linalg.norm(centroids - f_center, axis=1)
    indx = np.argsort(d)

    e_m = None
    for i in indx:
        candidate = matrix_elements_to_cut_arr[i]
        m_coords = nodes[conn[candidate]]
        hull = sp.spatial.ConvexHull(m_coords)
        A, b = (
            hull.equations[:, :-1],
            hull.equations[:, -1:],
        )  # extract the equations defining the 0-level set
        # The level set defines the hull such that all points Ax + b = 0 form the shell of the hull.
        # If Ax + b < 0, then the point is within the polyhedron.
        elem_in_region = np.all(f_coords @ A.T + b.T < 1e-8)
        if elem_in_region:
            # The fracture element is within the matrix element domain, this is the element to cut.
            e_m = int(candidate)
            # Pop it out of the list to check next time
            matrix_elements_to_cut_arr = np.delete(matrix_elements_to_cut_arr, i)
            centroids = np.delete(centroids, i, 0)
            break

    return e_m, matrix_elements_to_cut_arr, centroids


def _insert_split_nodes(
    state: _CutState,
    m_nodes: list[int],
    f_nodes: npt.NDArray[np.int64],
    f_coords: npt.NDArray[np.float64],
    tip_nodes: list[int],
    tip_element: bool,
    frac_normal: npt.NDArray[np.float64],
    tol: float,
) -> tuple[list[int], list[tuple[int, npt.NDArray[np.float64]]]]:
    """Add new nodes for one cut element: a single node at each fracture tip, or a
    coincident pair shifted to either side of the fracture for interior fracture nodes.

    Appends to `state.nodes`/`phi_sign`/`phi_sign_history`/`psi` and returns the
    extended `m_nodes` list, together with the shifted (non-tip) nodes so they can be
    restored afterwards.

    `phi_sign_history` is evaluated at each node's unshifted coordinate `f_coords[i, :]`
    -- the position the node is ultimately restored to below -- rather than at the
    shifted pos/neg coordinate, so a node's history stays consistent with where it
    ends up for deduplication.

    A coordinate on an earlier fracture's line evaluates to a historical sign of 0,
    but `_resolve_zero_level_set_nodes` never leaves an existing node at 0 -- so a
    fresh node here must be resolved the same way (majority sign of its element's
    neighbors) or it won't dedup against that existing node.
    """
    shifted_nodes = []
    shift_dist = 0.01 * tol if tip_element else 0.0
    neighbor_nodes = list(m_nodes)

    for i in range(len(f_nodes)):
        hist_row = _evaluate_historical_phi_sign(f_coords[i, :], state.frac_geoms)
        zero_cols = np.flatnonzero(hist_row[0] == 0.0)
        if zero_cols.size and neighbor_nodes:
            neighbor_hist = state.phi_sign_history[neighbor_nodes, :]
            for col in zero_cols:
                side_sign = _majority_side_sign(neighbor_hist[:, col].tolist())
                if side_sign is not None:
                    hist_row[0, col] = side_sign

        if np.any(f_nodes[i] == tip_nodes):
            # For tip nodes, add one node
            state.nodes = np.vstack((state.nodes, f_coords[i, :]))
            m_nodes.append(state.new_node_index)
            state.new_node_index += 1
            state.phi_sign = np.hstack((state.phi_sign, 0.0))
            state.phi_sign_history = np.vstack((state.phi_sign_history, hist_row))
            state.psi = np.hstack((state.psi, 0.0))
        else:
            # Positive side shift
            pos_coord = f_coords[i, :] + shift_dist * frac_normal
            state.nodes = np.vstack((state.nodes, pos_coord))
            m_nodes.append(state.new_node_index)
            shifted_nodes.append((state.new_node_index, f_coords[i, :]))  # Store to revert
            state.new_node_index += 1
            state.phi_sign = np.hstack((state.phi_sign, 1.0))
            state.phi_sign_history = np.vstack((state.phi_sign_history, hist_row))
            state.psi = np.hstack((state.psi, -1.0))

            # Negative side shift
            neg_coord = f_coords[i, :] - shift_dist * frac_normal
            state.nodes = np.vstack((state.nodes, neg_coord))
            m_nodes.append(state.new_node_index)
            shifted_nodes.append((state.new_node_index, f_coords[i, :]))  # Store to revert
            state.new_node_index += 1
            state.phi_sign = np.hstack((state.phi_sign, -1.0))
            state.phi_sign_history = np.vstack((state.phi_sign_history, hist_row))
            state.psi = np.hstack((state.psi, -1.0))

    return m_nodes, shifted_nodes


def _group_split_nodes(
    m_nodes: list[int],
    m_nodes_orig: list[int],
    nodes: npt.NDArray[np.float64],
    phi: npt.NDArray[np.float64],
    phi_sign: npt.NDArray[np.float64],
    center_2d: npt.NDArray[np.float64],
    u: npt.NDArray[np.float64],
    half_length: float,
    tip_element: bool,
    nsd: int,
    tol: float,
) -> list[list[int]]:
    """Group a cut element's nodes by side of the fracture, ready for triangulation.

    Interior cuts produce a positive-side and negative-side group. Tip elements also
    get a third group covering the region beyond the tip that the fracture doesn't
    actually cut through.
    """
    m_node_signs = phi_sign[m_nodes]
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

            if crosses_phi:
                # Find the exact intersection point of the edge with phi = 0 and
                # recompute psi there directly from geometry.
                phi_a, phi_b = phi[face_nodes[0]], phi[face_nodes[1]]
                nodes_a, nodes_b = nodes[face_nodes[0], :], nodes[face_nodes[1], :]

                denom = phi_a - phi_b
                nodes_c = (
                    (phi_a * nodes_b - phi_b * nodes_a) / denom
                    if abs(denom) > 1e-12
                    else 0.5 * (nodes_a + nodes_b)
                )
                s_intersect = np.dot(nodes_c - center_2d, u)
                psi_intersect = np.abs(s_intersect) - half_length

                # If the intersection occurs ahead of or at the crack tip
                if psi_intersect >= tol:
                    tip_group.extend(n for n in face_nodes if n not in tip_group)

        if len(tip_group) > 1:
            m_node_groups.append(tip_group)

    return m_node_groups


def _triangulate_node_groups(
    m_node_groups: list[list[int]],
    nodes: npt.NDArray[np.float64],
    cut_boundary: bool = False,
) -> list[list[int]]:
    """Triangulate (or keep as a single Q4/T3) each node group, returning new element connectivity."""
    new_elem = []
    for group in m_node_groups:
        if len(group) < 3:
            # Degenerate element, skip
            continue
        elif len(group) < 5 and not cut_boundary:
            # Keep as a Q4/T3, sort to ensure new nodes are CCW ordered
            global_conn = [_sort_boundary_coords(group, nodes[group, :])]
        else:
            # Perform triangulation
            tri = sp.spatial.Delaunay(nodes[group, :])
            group_arr = np.array(group)
            local_conn = np.array(tri.simplices.tolist())
            global_conn = group_arr[local_conn].tolist()

        new_elem.extend(global_conn)
    return new_elem


def _cut_matrix_element(
    state: _CutState,
    e_m: int,
    conn: list[list[int]],
    f_nodes: npt.NDArray[np.int64],
    f_coords: npt.NDArray[np.float64],
    tip_nodes: list[int],
    tip_element: bool,
    frac_normal: npt.NDArray[np.float64],
    center_2d: npt.NDArray[np.float64],
    u: npt.NDArray[np.float64],
    half_length: float,
    nsd: int,
    tol: float,
) -> list[list[int]]:
    """Split one matrix element along a fracture element: insert new nodes, group them by
    side of the fracture, triangulate each group, then restore the shifted node coordinates.
    """
    # If the node is already on the fracture path, we're going to
    # replace it with a new node
    m_nodes_orig = conn[e_m].copy()
    m_nodes = [node for node in m_nodes_orig if (state.phi_sign[node] != 0.0 or state.psi[node] > tol)]

    m_nodes, shifted_nodes = _insert_split_nodes(
        state, m_nodes, f_nodes, f_coords, tip_nodes, tip_element, frac_normal, tol
    )

    m_node_groups = _group_split_nodes(
        m_nodes,
        m_nodes_orig,
        state.nodes,
        state.phi,
        state.phi_sign,
        center_2d,
        u,
        half_length,
        tip_element,
        nsd,
        tol,
    )
    new_elem = _triangulate_node_groups(m_node_groups, state.nodes)

    # Restore shifted nodes back to their original degenerate coordinates
    for node_idx, original_coord in shifted_nodes:
        state.nodes[node_idx, :] = original_coord

    return new_elem


def _replace_cut_elements(
    conn: list[list[int]],
    elements_to_delete: list[int],
    matrix_elements_to_cut_arr: npt.NDArray[np.int64],
    new_elem: list[list[int]],
) -> list[list[int]]:
    """Remove matrix elements that were actually cut and append their replacement fragments."""
    for e in matrix_elements_to_cut_arr.tolist():
        elements_to_delete.remove(e)  # these elements weren't cut

    elements_to_delete.sort(reverse=True)
    for e in elements_to_delete:
        conn.pop(e)

    conn.extend(new_elem)
    return conn


def _majority_side_sign(values: list[float] | npt.NDArray[np.float64]) -> float | None:
    """Resolve an on-line (zero) sign to a side using the majority of nonzero neighbor signs.

    Returns None if there is nothing nonzero to disambiguate against.
    """
    nonzero = [v for v in values if v != 0.0]
    if not nonzero:
        return None
    return 1.0 if max(set(nonzero), key=nonzero.count) > 0 else -1.0


def _resolve_zero_level_set_nodes(
    state: _CutState,
    conn: list[list[int]],
    tip_nodes: list[int],
    frac_nodes: npt.NDArray[np.float64],
    tol: float,
) -> None:
    """Handle elements with nodes that fall exactly on the fracture but weren't cut.

    Duplicates each such node once per side it's needed on, so every surviving matrix
    element ends up unambiguously on one side of the fracture.
    """
    # Find all matrix tip nodes
    matrix_tip_nodes = []
    for i in tip_nodes:
        tip_coords = frac_nodes[i, :]
        d = np.linalg.norm(state.nodes - tip_coords, axis=1)
        matrix_tip_nodes.extend((np.where(d < tol)[0]).tolist())
    tip_node_set = set(matrix_tip_nodes)

    zero_ls_nodes = [
        n
        for n in np.where(state.phi_sign == 0.0)[0].tolist()
        if n not in tip_node_set and state.psi[n] <= tol
    ]

    for node in zero_ls_nodes:
        elems_with_node = [e for e, enodes in enumerate(conn) if node in enodes]
        last_side_sign = None

        for e in elems_with_node:
            enodes = conn[e]
            other_signs = [state.phi_sign[n2] for n2 in enodes if n2 != node]
            side_sign = _majority_side_sign(other_signs)
            if side_sign is None:
                continue  # nothing to disambiguate against; leave this occurrence alone
            last_side_sign = side_sign

            state.nodes = np.vstack((state.nodes, state.nodes[node, :]))
            state.phi_sign = np.hstack((state.phi_sign, side_sign))
            # Duplicate shares the original's coordinates exactly, so its history
            # w.r.t. earlier fractures is identical -- copy rather than recompute.
            state.phi_sign_history = np.vstack((state.phi_sign_history, state.phi_sign_history[node, :]))
            state.psi = np.hstack((state.psi, state.psi[node]))
            conn[e] = [state.new_node_index if n2 == node else n2 for n2 in enodes]
            state.new_node_index += 1

        # Fold the original zero-sign node into one of its own duplicates so
        # the dedup pass below removes it rather than leaving it dangling.
        if last_side_sign is not None:
            state.phi_sign[node] = last_side_sign


def _deduplicate_split_nodes(
    nodes: npt.NDArray[np.float64],
    phi_sign_history: npt.NDArray[np.float64],
    conn: list[list[int]],
    nsd: int,
) -> tuple[npt.NDArray[np.float64], list[list[int]], npt.NDArray[np.float64]]:
    """Remove duplicate nodes introduced while cutting, matching on coordinates AND
    phi_sign against every fracture considered so far (the current one plus history).

    Keying on the full history -- not just the current fracture's phi_sign -- matters
    at fracture intersections: two nodes can coincide in space and share the current
    fracture's sign while sitting on opposite sides of an earlier fracture, in which
    case they must stay distinct rather than merge.
    """
    # Append phi_sign_history columns as extra nodal coordinates
    nodes_old = np.hstack((nodes, phi_sign_history))

    # Round nodes to avoid truncation errors
    decimals = 9
    nodes_rounded = np.round(nodes_old, decimals=decimals)

    # Map unique nodes
    nodes_new, nodes_map = np.unique(nodes_rounded, axis=0, return_inverse=True)

    # New list of nodes
    nodes = nodes_new[:, 0:nsd]

    # Update connectivity
    conn = [nodes_map[sublist].tolist() for sublist in conn]

    return nodes, conn, nodes_new[:, nsd:]


def _snap_point_to_near_edges(
    point: npt.NDArray[np.float64],
    node_ids: list[int],
    nodes: npt.NDArray[np.float64],
    tol: float,
) -> npt.NDArray[np.float64]:
    """Snap `point` onto a polygon edge it lies within `tol` of, if any.

    Existing snapping only checks point-to-node distance, so a point that's merely
    near an edge (not either endpoint) slips through and later triangulates into a
    real-but-tiny sliver. Projecting onto the edge makes that degeneracy exact.
    """
    n = len(node_ids)
    for i in range(n):
        a = nodes[node_ids[i], :]
        b = nodes[node_ids[(i + 1) % n], :]
        ab = b - a
        ab_len_sq = float(np.dot(ab, ab))
        if ab_len_sq < 1e-30:
            continue
        t = float(np.dot(point - a, ab) / ab_len_sq)
        t_clamped = min(max(t, 0.0), 1.0)
        proj = a + t_clamped * ab
        if np.linalg.norm(point - proj) < tol:
            return proj
    return point


def _cut_tip_boundary_elements(
    state: _CutState,
    conn: list[list[int]],
    matrix_elements_to_cut_arr: npt.NDArray[np.int64],
    centroids: npt.NDArray[np.float64],
    frac_nodes: npt.NDArray[np.float64],
    tip_nodes: list[int],
    frac_normal: npt.NDArray[np.float64],
    nsd: int,  # noqa
    tol: float,
) -> tuple[list[list[int]], npt.NDArray[np.int64]]:
    """Cut far-side elements left over when a fracture tip lands exactly on an existing
    matrix boundary -- an existing node, or a point partway along an edge -- shared
    with another element that also qualified for cutting in `_find_cut_elements`.
    """
    new_elem: list[list[int]] = []

    for tip in tip_nodes:
        f_nodes_tip = np.array([tip])
        f_coords_tip = frac_nodes[[tip], :]

        while len(matrix_elements_to_cut_arr) > 0:
            e_m, matrix_elements_to_cut_arr, centroids = _find_parent_element(
                f_coords_tip, matrix_elements_to_cut_arr, centroids, conn, state.nodes
            )
            if e_m is None:
                break

            # If the node is already on the fracture path, we're going to
            # replace it with a new node
            m_nodes_orig = conn[e_m].copy()
            m_nodes = [
                node for node in m_nodes_orig if (state.phi_sign[node] != 0.0 or state.psi[node] > tol)
            ]

            # Snap the tip onto a nearby edge so any resulting sliver has exactly-zero area.
            snapped_coord = _snap_point_to_near_edges(f_coords_tip[0, :], m_nodes_orig, state.nodes, tol)
            f_coords_tip_elem = snapped_coord.reshape(1, -1)

            m_nodes, _ = _insert_split_nodes(
                state, m_nodes, f_nodes_tip, f_coords_tip_elem, tip_nodes, True, frac_normal, tol
            )

            tri_elem = _triangulate_node_groups([m_nodes], state.nodes, True)

            # Drop the resulting zero-area triangles.
            for tri in tri_elem:
                if len(tri) != 3:
                    new_elem.append(tri)
                    continue
                p0, p1, p2 = state.nodes[tri[0]], state.nodes[tri[1]], state.nodes[tri[2]]
                area = 0.5 * abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))
                if area > 1e-8:
                    new_elem.append(tri)

    return new_elem, matrix_elements_to_cut_arr


def _prune_unreferenced_nodes(
    nodes: npt.NDArray[np.float64],
    conn: list[list[int]],
    phi_sign_history: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], list[list[int]], npt.NDArray[np.float64]]:
    """Drop nodes no longer referenced by any element and renumber connectivity.

    A node cut out of every element that used to reference it (e.g. an on-fracture
    node superseded by a fresh split-node pair everywhere it appeared) would
    otherwise survive as a phantom DOF and make the assembled system singular.
    """
    referenced = np.zeros(nodes.shape[0], dtype=bool)
    for enodes in conn:
        referenced[enodes] = True

    if referenced.all():
        return nodes, conn, phi_sign_history

    new_index = np.full(nodes.shape[0], -1, dtype=np.int64)
    new_index[referenced] = np.arange(int(referenced.sum()))

    nodes = nodes[referenced]
    phi_sign_history = phi_sign_history[referenced]
    conn = [new_index[enodes].tolist() for enodes in conn]

    return nodes, conn, phi_sign_history


def _split_elements(
    mesh: dict[str, Any],
    frac_ls: FractureLevelSet,
    frac_nodes: npt.NDArray[np.float64],
    frac_conn: npt.NDArray[np.int64],
    tip_nodes: list[int],
) -> tuple[dict[str, Any], FractureLevelSet]:
    """Loop through the fracture elements, find the cut elements, and split them accordingly."""
    conn = mesh["conn"]
    ne = mesh["ne"]
    nsd = mesh["nsd"]
    tol = frac_ls.tol
    frac_normal = frac_ls.n

    state = _CutState(
        nodes=mesh["nodes"],
        phi=frac_ls.phi,
        phi_sign=frac_ls.phi_sign,
        phi_sign_history=frac_ls.phi_sign_history,
        frac_geoms=frac_ls.frac_geoms,
        psi=frac_ls.psi,
        new_node_index=mesh["nn"],
    )

    # Find matrix elements that need cutting, and their centroids -- used below to match
    # each fracture element to its parent, nearest-centroid first, to avoid excessive looping.
    matrix_elements_to_cut = _find_cut_elements(conn, ne, state.phi, state.phi_sign, state.psi, tol)
    elements_to_delete = matrix_elements_to_cut.copy()  # Keep a list to delete these elements at the end
    matrix_elements_to_cut_arr = np.array(matrix_elements_to_cut)
    centroids = _compute_element_centroids(matrix_elements_to_cut_arr, conn, state.nodes)

    new_elem: list[list[int]] = []
    ne_f = np.size(frac_conn, 0)

    # ---- Handle edge case: an element's boundary edge is collinear with the fracture
    # line but the fracture stops short of spanning it (the tip lands mid-edge).
    # `_find_cut_elements` can never see these elements (see docstring), so find and
    # cut them separately, up front, using the original (pre-cut) connectivity.
    if tip_nodes:
        collinear_matches = _find_collinear_tip_elements(
            conn, nsd, state.phi_sign, state.psi, frac_ls, tip_nodes, frac_nodes, state.nodes
        )
        if collinear_matches:
            collinear_new_elem, collinear_consumed = _cut_collinear_tip_elements(
                state, conn, collinear_matches, frac_nodes, tip_nodes, frac_normal, tol
            )
            new_elem.extend(collinear_new_elem)
            for e in collinear_consumed:
                if e not in elements_to_delete:
                    elements_to_delete.append(e)

    # ---- Loop through all fracture elements, cutting each one's parent matrix element
    for e_f in range(ne_f):
        f_nodes = frac_conn[e_f, :]
        f_coords = frac_nodes[f_nodes, :]
        tip_element = bool(np.any(f_nodes == tip_nodes)) if len(tip_nodes) > 0 else False

        e_m, matrix_elements_to_cut_arr, centroids = _find_parent_element(
            f_coords, matrix_elements_to_cut_arr, centroids, conn, state.nodes
        )
        if e_m is None:
            # fracture element lies along a boundary
            continue

        new_elem.extend(
            _cut_matrix_element(
                state,
                e_m,
                conn,
                f_nodes,
                f_coords,
                tip_nodes,
                tip_element,
                frac_normal,
                frac_ls.center_2d,
                frac_ls.u,
                frac_ls.half_length,
                nsd,
                tol,
            )
        )

    # ---- Handle edge case: Fracture tip falls exactly on mesh edge and far element needs to be cut.
    if tip_nodes and len(matrix_elements_to_cut_arr) > 0:
        tip_new_elem, matrix_elements_to_cut_arr = _cut_tip_boundary_elements(
            state, conn, matrix_elements_to_cut_arr, centroids, frac_nodes, tip_nodes, frac_normal, nsd, tol
        )
        new_elem.extend(tip_new_elem)

    conn = _replace_cut_elements(conn, elements_to_delete, matrix_elements_to_cut_arr, new_elem)

    # ---- Handle edge case: elements with nodes that fall on the fracture
    # exactly but are not cut by the fracture.
    _resolve_zero_level_set_nodes(state, conn, tip_nodes, frac_nodes, tol)

    # ---- Remove duplicate nodes introduced while cutting. Key on this fracture's
    # phi_sign together with the full history so nodes that coincide but sit on
    # opposite sides of an earlier fracture (e.g. at an intersection) stay distinct.
    combined_history = np.hstack((state.phi_sign_history, state.phi_sign.reshape(-1, 1)))
    nodes, conn, phi_sign_history = _deduplicate_split_nodes(state.nodes, combined_history, conn, nsd)
    nodes, conn, phi_sign_history = _prune_unreferenced_nodes(nodes, conn, phi_sign_history)

    frac_ls.phi_sign = phi_sign_history[:, -1]
    frac_ls.phi_sign_history = phi_sign_history[:, :-1]
    mesh["nodes"] = nodes
    mesh["conn"] = sorted(conn, key=len)  # Resort conn by node length
    # Full history including the fracture just processed, for the next iteration
    # and for anything downstream that wants per-fracture sign of every node.
    mesh["phi_sign_history"] = phi_sign_history

    return mesh, frac_ls


def _create_global_fracture_connectivities(
    mesh: dict[str, Any],
    frac_ls: FractureLevelSet,
    frac_nodes: npt.NDArray[np.float64],
    frac_conn: npt.NDArray[np.int64],
) -> dict[str, Any]:
    """Accumulate one fracture's nodes, connectivity, and bookkeeping into the running
    `mesh["fracture_mesh"]` structure.
    """
    ne_f = np.size(frac_conn, 0)

    if "fracture_mesh" not in mesh:
        # First fracture: initialize empty accumulator structures
        mesh["fracture_mesh"] = {
            "nodes": np.empty((0, frac_nodes.shape[1])),
            "conn": np.empty((0, frac_conn.shape[1]), dtype=np.int64),
            "matrix_conn": [],
            "region_lists": [],  # In the fractures, regions correspond to the fracture number.
            "nn": 0,
            "ne": 0,
            "fracture_tangent": [],
            "fracture_normal": [],
            "tol": [],  # per-fracture tolerance, needed later for matrix-conn matching
        }
    frac_mesh = mesh["fracture_mesh"]

    # Number of fracture nodes/elements already accumulated from prior fractures,
    # used to offset this fracture's own (locally-zero-based) connectivity.
    prior_nn = frac_mesh["nn"]
    prior_ne = frac_mesh["ne"]

    # Local fracture nodes
    frac_mesh["nodes"] = np.vstack((frac_mesh["nodes"], frac_nodes))
    # Local fracture connectivity, offset past the already-accumulated fracture nodes
    frac_mesh["conn"] = np.vstack((frac_mesh["conn"], frac_conn + prior_nn))
    # Number of fracture nodes
    frac_mesh["nn"] = prior_nn + np.size(frac_nodes, 0)
    # Number of fracture elements
    frac_mesh["ne"] = prior_ne + ne_f
    # Fracture unit vectors
    frac_mesh["fracture_tangent"].append(frac_ls.u)
    frac_mesh["fracture_normal"].append(frac_ls.n)
    frac_mesh["tol"].append(frac_ls.tol)

    # conn_global (fracture connectivity in combined matrix+fracture global numbering)
    # is NOT computed here: mesh["nn"] (matrix node count) keeps growing as later
    # fractures cut further matrix elements, so any offset applied now would go stale
    # by the time the loop finishes. It's (re)computed once from frac_mesh["conn"] in
    # `_create_fracture_mesh`, after mesh["nn"] is final.

    # Record which fracture-element indices (in the combined, accumulated numbering)
    # belong to this fracture.
    frac_mesh["region_lists"].append(np.arange(prior_ne, prior_ne + ne_f))

    return mesh


def _find_fracture_matrix_conn(
    matrix_nodes: npt.NDArray[np.float64],
    conn: list[list[int]],
    node_to_elems: dict[int, set[int]],
    frac_nodes: npt.NDArray[np.float64],
    frac_conn: npt.NDArray[np.int64],
    phi_sign: npt.NDArray[np.float64],
    tol: float = 1e-8,
) -> list[list[int]]:
    """Match one fracture's elements to their neighboring positive/negative matrix
    elements. `conn`/`matrix_nodes`/`node_to_elems`/`phi_sign` must all correspond to
    the same (final) matrix element numbering.
    """
    ne_f = np.size(frac_conn, 0)
    frac_matrix_conn: list[list[int]] = []
    for e_f in range(ne_f):
        f_coords = frac_nodes[frac_conn[e_f, :]]

        pos_candidates: set[int] | None = None
        neg_candidates: set[int] | None = None

        for coord in f_coords:
            d = np.linalg.norm(matrix_nodes - coord, axis=1)
            coincident = np.where(d < tol)[0]

            pos_nodes = coincident[phi_sign[coincident] >= 0]
            neg_nodes = coincident[phi_sign[coincident] <= 0]

            pos_elems: set[int] = set()
            for node in pos_nodes:
                pos_elems |= node_to_elems.get(int(node), set())

            neg_elems: set[int] = set()
            for node in neg_nodes:
                neg_elems |= node_to_elems.get(int(node), set())

            pos_candidates = pos_elems if pos_candidates is None else pos_candidates & pos_elems
            neg_candidates = neg_elems if neg_candidates is None else neg_candidates & neg_elems

        pos_elem = pos_candidates.pop() if pos_candidates else None
        neg_elem = neg_candidates.pop() if neg_candidates else None

        if pos_elem is not None and neg_elem is not None:
            # append to frac_matrix_conn as [positive side, negative side]
            frac_matrix_conn.append([pos_elem, neg_elem])
        elif pos_elem is None and neg_elem is None:
            debugging_mesh_plot_matrix_and_fracture(matrix_nodes, conn, frac_nodes, frac_conn, True)
            xbar = np.mean(f_coords, axis=0)
            raise ValueError(
                f"Failed to find corresponding matrix element for fracture element located at {xbar}."
            )
        else:
            found_elem = pos_elem if pos_elem is not None else neg_elem
            assert found_elem is not None
            warnings.warn(
                f"Only one matrix element (of positive/negative side) found for fracture "
                f"element {e_f}; appending single-element matrix connectivity."
            )
            frac_matrix_conn.append([found_elem])

    return frac_matrix_conn


def _assemble_fracture_matrix_conn(mesh: dict[str, Any]) -> dict[str, Any]:
    """Match every fracture's elements to their parent matrix element(s), once, after
    every fracture has been cut.
    """
    frac_mesh = mesh["fracture_mesh"]
    conn = mesh["conn"]
    matrix_nodes = mesh["nodes"]
    phi_sign_history = mesh["phi_sign_history"]

    # Build the node -> matrix-element lookup once for all fractures, rather than
    # once per fracture as before.
    node_to_elems: dict[int, set[int]] = {}
    for e, enodes in enumerate(conn):
        for node in enodes:
            node_to_elems.setdefault(node, set()).add(e)

    matrix_conn: list[list[int]] = []
    for i, region in enumerate(frac_mesh["region_lists"]):
        # frac_mesh["conn"] is already offset into the accumulated frac_mesh["nodes"]
        # array, so this fracture's slice of it can be used directly.
        frac_conn_i = frac_mesh["conn"][region]
        matrix_conn.extend(
            _find_fracture_matrix_conn(
                matrix_nodes,
                conn,
                node_to_elems,
                frac_mesh["nodes"],
                frac_conn_i,
                phi_sign_history[:, i],
            )
        )

    frac_mesh["matrix_conn"] = matrix_conn
    return mesh


def _local_element_scale(
    nodes: npt.NDArray[np.float64],
    conn: list[list[int]],
    center: npt.NDArray[np.float64],
    half_length: float,
    u: npt.NDArray[np.float64],
    n: npt.NDArray[np.float64],
    pad: float,
    percentile: float = 1.0,
) -> float | None:
    """Low-percentile edge length among elements near a fracture's line.

    Elements are selected via a bounding box aligned to the fracture's own
    tangent/normal axes.Uses a low percentile rather than the strict minimum,
    to prevent sliver elements from creating more slivers.

    Used to scale the node-snapping tolerance to local mesh resolution rather than
    overall domain size. Returns None if no element centroids fall within the strip.
    """
    edge_lengths = []
    for elem in conn:
        coords = nodes[elem]
        centroid = coords.mean(axis=0)
        rel = centroid - center
        s = rel @ u  # tangential coordinate along the fracture
        d = rel @ n  # normal (perpendicular) distance from the fracture line
        if abs(s) > half_length + pad or abs(d) > pad:
            continue
        n_nodes = len(elem)
        for i in range(n_nodes):
            edge = coords[(i + 1) % n_nodes] - coords[i]
            edge_lengths.append(float(np.sqrt(edge @ edge)))

    if not edge_lengths:
        return None
    return float(np.percentile(edge_lengths, percentile))


def _node_to_elem_lookup(conn: list[list[int]]) -> dict[int, set[int]]:
    # Build a node -> matrix-element lookup
    node_to_elems: dict[int, set[int]] = {}
    for e, enodes in enumerate(conn):
        for node in enodes:
            node_to_elems.setdefault(node, set()).add(e)

    return node_to_elems


def _insert_fracture_junctions(
    mesh: dict[str, Any], fracture_specs: list[dict[str, Any]], tol: float
) -> dict[str, Any]:
    """Loop through the fractures and detect any crossings.

    If crossings exist, check if it intersects any existing nodes.
    If not, add new node and split the element.
    """
    nodes = mesh["nodes"]
    conn = mesh["conn"]
    node_to_elems = _node_to_elem_lookup(conn)
    n_fractures = len(fracture_specs)
    elements_to_delete = []

    for indx_1 in range(n_fractures - 1):
        fracture_1 = fracture_specs[indx_1]
        center_1, half_length_1, _, u_1, _ = _get_fracture_geometry_2D(fracture_1)

        for indx_2 in range(indx_1 + 1, n_fractures):
            new_elem = []

            fracture_2 = fracture_specs[indx_2]
            center_2, half_length_2, _, u_2, _ = _get_fracture_geometry_2D(fracture_2)

            # Check for intersections
            Px1 = center_1[0]
            Py1 = center_1[1]
            ux1 = u_1[0]
            uy1 = u_1[1]

            Px2 = center_2[0]
            Py2 = center_2[1]
            ux2 = u_2[0]
            uy2 = u_2[1]

            s = ((Px2 - Px1) * uy1 - (Py2 - Py1) * ux1) / (uy2 * ux1 - uy1 * ux2)

            junction = center_2 + u_2 * s
            L1 = np.linalg.norm(junction - center_1)
            L2 = np.linalg.norm(junction - center_2)
            if half_length_1 < L1 or half_length_2 < L2:
                # The fractures do not intersect
                continue

            d = np.linalg.norm(nodes - junction, axis=1)
            # 1. Check if the fracture exists in the mesh already
            if np.any(d < tol):
                continue

            # 2. Find which element it is in. A single nearest node isn't a reliable
            # proxy once an earlier junction has fan-triangulated the area: the true
            # containing element can belong to a slightly-farther node whose own
            # untouched wedge happens to cover this point. Search a small neighbourhood
            # of nearby nodes instead of just the closest one.
            k = min(5, len(nodes))
            nearby_nodes = np.argsort(d)[:k]
            candidate_elems: set[int] = set()
            for nn in nearby_nodes:
                candidate_elems |= node_to_elems[int(nn)]

            e_m = None
            for e in candidate_elems:
                enodes = conn[e]
                xI = nodes[enodes, :]

                hull = sp.spatial.ConvexHull(xI)
                A, b = (
                    hull.equations[:, :-1],
                    hull.equations[:, -1:],
                )  # extract the equations defining the 0-level set
                # The level set defines the hull such that all points Ax + b = 0 form the shell of the hull.
                # If Ax + b < 0, then the point is within the polyhedron.
                if np.all(junction @ A.T + b.T < 1e-8):
                    e_m = e  # The junction is in this element
                    break

            if e_m is None:
                # No element neighbouring the k nearest nodes to the junction was found
                # to contain it, therefore the junction must be outside the domain.
                continue

            # Otherwise, insert the node and split the element.
            nodes = np.vstack((nodes, junction))
            added_elem = _triangulate_node_groups([enodes + [np.size(nodes, 0) - 1]], nodes, True)
            new_elem.extend(added_elem)
            elements_to_delete = [e_m]
            conn = _replace_cut_elements(conn, elements_to_delete, np.empty(0, dtype=np.int64), new_elem)
            node_to_elems = _node_to_elem_lookup(conn)

    mesh["nodes"] = nodes
    mesh["conn"] = conn

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
    if fracture_specs is None:
        raise ValueError(
            """Fracture block included but no fractures specified.
            Remove fracture block from config if not needed."""
        )

    # Set node distance tolerance based on domain size, bounded by local mesh resolution
    # so the tolerance shrinks with the mesh instead of staying fixed at the domain scale.

    d_max = np.amax(nodes, 0)
    L = (d_max.prod()) ** (1 / nsd)
    tol_domain = L * 0.005
    tol_floor = L * 1.0e-9

    # Insert junction nodes before beginning the cutting
    mesh = _insert_fracture_junctions(mesh, fracture_specs, tol_domain)
    mesh = _extract_key_mesh_values(mesh)

    # Geometry (center_2d, n, tol) of every fracture processed so far -- used only to
    # evaluate phi_sign_history for brand-new nodes as they're created mid-cut.
    frac_geoms: list[FracGeom] = []
    # phi_sign of every current node against every fracture processed so far. This is
    # carried forward as-is (never re-evaluated from coordinates) because nodes split
    # into a coincident +1/-1 pair along an earlier fracture sit exactly on that
    # fracture's line.
    phi_sign_history = np.empty((mesh["nn"], 0))

    # Loop through each fracture specification
    for fracture in fracture_specs:
        center_2d, half_length, _, u, n = _get_fracture_geometry_2D(fracture)
        h_local = _local_element_scale(
            mesh["nodes"], mesh["conn"], center_2d, half_length, u, n, pad=tol_domain
        )
        tol = min(tol_domain, 0.5 * h_local) if h_local is not None else tol_domain
        tol = max(tol, tol_floor)

        # Compute fracture geometry and nodal level sets once, shared by both helpers below
        frac_ls = _build_fracture_level_set(mesh["nodes"], fracture, phi_sign_history, frac_geoms, tol)

        # Create fracture nodes and connectivity
        frac_nodes, frac_conn, tip_nodes = _create_fracture_nodes_conn(mesh, frac_ls)

        # Loop through elements and check whether the new nodes lie within each element, then split
        mesh, frac_ls = _split_elements(mesh, frac_ls, frac_nodes, frac_conn, tip_nodes)

        # Assemble global fracture connectivity and matrix-fracture connectivity
        mesh = _create_global_fracture_connectivities(mesh, frac_ls, frac_nodes, frac_conn)
        mesh = _extract_key_mesh_values(mesh)

        # This fracture is now part of the history every subsequent fracture (and
        # every node it creates) gets checked against. phi_sign_history is taken
        # straight from mesh -- it's already been resolved through this fracture's
        # dedup pass and is authoritative for the next iteration.
        frac_geoms.append((frac_ls.center_2d, frac_ls.n, frac_ls.tol))
        phi_sign_history = mesh["phi_sign_history"]

    if len(mesh["fracture_mesh"]["conn"]) == 0:
        raise ValueError(
            """No fracture elements generated from fracture specifications.
            Try changing the fracture definitions or use a different mesh."""
        )

    # Match every fracture element to its parent matrix element(s) now that
    # mesh["conn"] has its final, stable numbering -- see _assemble_fracture_matrix_conn.
    mesh = _assemble_fracture_matrix_conn(mesh)

    # Deduplicate any fracture nodes
    fmesh = mesh["fracture_mesh"]
    fmesh["nodes"], fmesh["conn"], _ = _deduplicate_split_nodes(
        fmesh["nodes"], np.zeros((fmesh["nn"], 1)), fmesh["conn"], nsd
    )
    # conn_global is fracture connectivity in combined matrix+fracture global numbering.
    fmesh["conn_global"] = [[node + mesh["nn"] for node in elem] for elem in fmesh["conn"]]

    # Re-evaluate key values
    mesh["fracture_mesh"] = _extract_key_mesh_values(mesh["fracture_mesh"])
    mesh["nn_total"] = mesh["nn"] + mesh["fracture_mesh"]["nn"]
    mesh["nodes_total"] = np.vstack([mesh["nodes"], mesh["fracture_mesh"]["nodes"]])

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
            nq, etyp = 2, "T3"
        case (2, 2):  # L2
            nq, etyp = 2, "L2"
    return nq, etyp


def _extract_key_mesh_values(mesh: dict[str, Any]) -> dict[str, Any]:
    nodes = mesh["nodes"]
    conn = mesh["conn"]

    # nodes_total includes matrix and fracture nodes for places where all
    # nodes are necessary. Updated when creating fractures.
    mesh["nodes_total"] = nodes

    # Extract key values
    # nn is the number of matrix nodes, nn_total includes fracture nodes as well.
    mesh["nn"] = np.shape(nodes)[0]  # number of nodes
    mesh["nn_total"] = mesh["nn"]

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
