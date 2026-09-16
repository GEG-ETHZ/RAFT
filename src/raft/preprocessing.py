"""Import mesh, calculate material properties, apply defaults."""

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse
import yaml
from scipy.sparse.csgraph import dijkstra

from .matprops import create_abstract_state
from .matrixassembly import integrate_boundary_flux_matrices
from .meshprocessing import _get_element_face_lists, load_mesh

# ----- Default values for config file
DEFAULTS: dict[str, Any] = {
    # ---- mesh ----------------------------------------------------------------
    "mesh": {
        "file": "box3D.msh",
        "dimensions": 3,
        "regions": "",
    },
    # ---- material ------------------------------------------------------------
    "material": {
        "solid": {
            "bulk_mod": 20.0e9,  # Pa
            "density": 2600.0,  # kg/m^3
            "therm_cond": 3.0,  # W/mK
            "specific_heat": 800.0,  # J/kgK
            "therm_expansion": 12.0e-6,  # 1/K
            "porosity": 0.01,  # -
            "biot_coeff": 0.5,  # -
            "permeability": 1.0e-16,  # m^2
            "rotation_x": 0.0,  # rotation of local coordinates around x-axis (degrees)
            "rotation_y": 0.0,  # rotation of local coordinates around y-axis (degrees)
            "rotation_z": 0.0,  # rotation of local coordinates around z-axis (degrees)
        },
        "fluid": {
            "fluid_type": "WATER",  # Fluid name for property lookup
            "bulk_mod": 2.0e9,  # Pa
            "density": 1000,  # kg/m^3
            "reference_temperature": 293,  # K
            "reference_pressure": 1.0e5,  # K
            "therm_cond": 0.6,  # W/mK
            "therm_expansion": 200.0e-6,  # 1/K
            "specific_heat": 4180.0,  # J/kgK
            "viscosity": 1e-3,  # Pa·s
        },
    },
    # ---- initial conditions --------------------------------------------------
    "initial_conditions": {
        "assignment": "constant",
        "T_ini": 20.0,  # deg C
        "p_ini": 1.0e5,  # Pa
        "T_grad": 0.0,  # deg C/m
        "p_grad": 0.0,  # Pa / m
        "grad_direction_T": [0, 0, -1],  # unit vector in gradient direction
        "grad_direction_p": [0, 0, -1],  # unit vector in gradient direction
    },
    # ---- boundary conditions -------------------------------------------------
    "boundary_conditions": [],
    # ---- simulation ----------------------------------------------------------
    "simulation": {
        "t_start": 0.0,
        "t_end": 100.0,
        "timestep": 1.0,
        "timestep_min": 1.0,
        "timestep_max": 1.0,
        "solver_type": "linearized",
        "output_frequency": 1,
        "output_vtk": 0,
        "output_log": 0,
        "output_viscosity": 0,
        "output_density": 0,
        "output_porosity": 0,
        "rate_order": 1,
        "advection_on": 0,
        "volumetric_strain_on": 1,
        "gravity": 0.0,
        "steady_pressure": 0,
        "steady_temperature": 0,
        "axisymmetric_on": 0,
        "rel_error_T": 1.0e-2,
        "iterative_controls": {
            "NR_tol": 1.0e-6,  # Newton-Raphson residual tolerance
            "iter_max": 100,  # Maximum number of iterations
            "aitken_relaxation": 1,  # Turn on/off Aitken's delta^2 relaxation
            "NR_step": 1,  # Newton-Raphson line step size
        },
    },
}


def _apply_defaults(
    config: dict[str, Any], defaults: dict[str, Any], path: str = "", verbose: bool = True
) -> dict[str, Any]:
    """Check for missing values in the config file. Apply and report default values."""
    for key, default_val in defaults.items():
        full_key = f"{path}.{key}" if path else key

        if key not in config or config[key] is None:
            config[key] = default_val
            if verbose:
                print(f"  [DEFAULT]  {full_key} = {default_val}")
        elif isinstance(default_val, dict) and isinstance(config[key], dict):
            _apply_defaults(config[key], default_val, full_key, verbose)
        else:
            if verbose:
                print(f"  [CONFIG]   {full_key} = {config[key]}")

    return config


def _determine_physics_controls(config: dict[str, Any], verbose: bool = True) -> dict[str, Any]:
    """Determine physics controls that are controlled through implicit inputs.

    This primarily includes non-constant material properties, but also includes
    non-feasible combinations of settings.
    """
    fluid_name = config["material"]["fluid"]["fluid_type"]
    # Viscosity
    mu = config["material"]["fluid"]["viscosity"]
    if not isinstance(mu, (int, float)):
        raise ValueError(f"Incompatible viscosity format: {mu}")
    elif mu <= 0:  # non-constant: real-fluid viscosity from CoolProp
        config["simulation"]["viscosity_nonconstant"] = 1
        if verbose:
            print(f"\nViscosity computed from CoolProp ({fluid_name}) as a function of T and p.")
    else:  # constant
        config["simulation"]["viscosity_nonconstant"] = 0

    # Density
    rho = config["material"]["fluid"]["density"]
    if not isinstance(rho, (int, float)):
        raise ValueError(f"Incompatible density format: {rho}")
    elif rho == 0:  # non-constant: real-fluid density from CoolProp
        config["simulation"]["density_nonconstant"] = 1
        if verbose:
            print(f"\nDensity computed from CoolProp ({fluid_name}) as a function of T and p.")
    else:  # constant
        config["simulation"]["density_nonconstant"] = 0

    # Axisymmetry
    axisymmetric_on = config["simulation"]["axisymmetric_on"]
    if axisymmetric_on and config["mesh"]["dimensions"] == 3:
        config["simulation"]["axisymmetric_on"] = 0
        if verbose:
            print("Axisymmetry turned off, not valid in 3D analysis.")

    # Fractures
    config["simulation"]["fracture_on"] = 1 if "fractures" in config["mesh"] else 0

    # CoolProp AbstractState, reused across all property lookups instead of
    # re-initializing it on every matprops call.
    if config["simulation"]["viscosity_nonconstant"] or config["simulation"]["density_nonconstant"]:
        config["material"]["fluid"]["abstract_state"] = create_abstract_state(fluid_name)
    else:
        config["material"]["fluid"]["abstract_state"] = None

    return config


def load_config(config_name: str, config_dir: Path | None = None, verbose: bool = True) -> dict[str, Any]:
    """Load config file and check for values."""
    if config_dir is None:
        # Fallback to default folder structure
        script_dir = Path(__file__).parent
        config_dir = script_dir.parent.parent / "config"
    else:
        config_dir = Path(config_dir)

    config_path = config_dir / config_name

    with open(config_path) as f:
        config: dict[str, Any] = yaml.safe_load(f)
    config["config_name"] = config_name
    config["config_dir"] = str(config_path.parent)

    if verbose:
        print(f"\n ----- Configuration parameters for {config_name}")

    _apply_defaults(config, DEFAULTS, verbose=verbose)

    _determine_physics_controls(config, verbose)

    if verbose:
        print()

    return config


def _check_region_properties(max_region: int, material: dict[str, Any]) -> None:
    """Check that the correct number of properties are included."""
    missing_props = []
    for mat_prop in material:
        if isinstance(material[mat_prop], list):  # noqa
            if len(material[mat_prop]) < (max_region + 1):
                missing_props.append(mat_prop)

    if len(missing_props) > 0:
        raise IndexError(f"Insufficient properties provided for:\n{chr(10).join(missing_props)}")

    return


def rotation_matrices(
    rot_x: float, rot_y: float, rot_z: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return individual 3D rotation matrices (angles in degrees).

    Always returns full 3x3 matrices. Rotations about in-plane axes (x, y)
    are not valid 2D rotations on their own, so callers working in 2D must
    apply the full 3D rotation to a full 3D tensor and only truncate the
    *result* down to the in-plane (nsd x nsd) block, rather than truncating
    these matrices before multiplying.
    """
    rot_x = np.radians(rot_x)
    rot_y = np.radians(rot_y)
    rot_z = np.radians(rot_z)

    cx, sx = np.cos(rot_x), np.sin(rot_x)
    cy, sy = np.cos(rot_y), np.sin(rot_y)
    cz, sz = np.cos(rot_z), np.sin(rot_z)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    return Rx, Ry, Rz


def _get_region_value(val: Any, r: int) -> Any:
    """Return value for region, handling both scalar and list cases.

    Fallback for homogeneous cases defined as scalars instead of lists.
    """
    return val[r] if isinstance(val, list) else val


def _calculate_derived_properties(mesh: dict[str, Any], material: dict[str, Any]) -> dict[str, Any]:
    """Calculate properties derived from other primary properties."""
    nsd = mesh["nsd"]
    num_regions = max(mesh["regions"]) + 1

    # Initialize derived properties storage
    material["solid"]["storage_mod"] = [None] * num_regions
    material["solid"]["therm_cond_eff"] = [None] * num_regions
    material["solid"]["therm_expansion_eff"] = [None] * num_regions
    material["solid"]["volumetric_heat_capacity_eff"] = [None] * num_regions
    material["solid"]["permeability_eff"] = [None] * num_regions
    material["solid"]["drained_bulk_mod"] = [None] * num_regions

    for r in range(num_regions):
        # ----- material properties
        alpha_s = _get_region_value(material["solid"]["therm_expansion"], r)
        alpha_f = material["fluid"]["therm_expansion"]
        phi = _get_region_value(material["solid"]["porosity"], r)
        biot = _get_region_value(material["solid"]["biot_coeff"], r)
        Kf = material["fluid"]["bulk_mod"]
        Ks = _get_region_value(material["solid"]["bulk_mod"], r)
        lambda_s = _get_region_value(material["solid"]["therm_cond"], r)
        kf = _get_region_value(material["solid"]["permeability"], r)

        # physical properties
        angle_x = _get_region_value(material["solid"]["rotation_x"], r)
        angle_y = _get_region_value(material["solid"]["rotation_y"], r)
        angle_z = _get_region_value(material["solid"]["rotation_z"], r)

        Rx, Ry, Rz = rotation_matrices(angle_x, angle_y, angle_z)
        R = Rx @ Ry @ Rz  # general 3D rotation matrix (always orthogonal)

        # For anisotropic case, rotate the full 3D tensor and only then
        # restrict to the in-plane (nsd x nsd) block.
        if isinstance(kf, list):
            kf.append(0.0) if len(kf) < 3 else kf
            kf = (R @ np.diag(kf) @ R.T)[:nsd, :nsd]

        # Derived properties (computed here, not stored in YAML)
        M = Kf * Ks / (phi * Ks + (biot - phi) * Kf)  # storage modulus

        # effective expansion modulus
        alpha_e = biot * alpha_s + phi * (alpha_f - alpha_s)

        # effective conductivity
        if isinstance(lambda_s, list):
            lambda_s.append(0.0) if len(lambda_s) < 3 else lambda_s
            # anisotropic case
            lambda_e = (
                phi * material["fluid"]["therm_cond"] * np.eye(nsd)
                + (1 - phi) * (R @ np.diag(lambda_s) @ R.T)[:nsd, :nsd]
            )
            material["solid"]["therm_cond"][r] = (R @ np.diag(lambda_s) @ R.T)[:nsd, :nsd]
        else:
            # isotropic case
            lambda_e = phi * material["fluid"]["therm_cond"] + (1 - phi) * lambda_s

        # effective volumetric heat capacity
        cp_s = _get_region_value(material["solid"]["specific_heat"], r)
        rho_s = _get_region_value(material["solid"]["density"], r)
        m_e = (
            phi * material["fluid"]["specific_heat"] * material["fluid"]["density"] + (1 - phi) * cp_s * rho_s
        )

        # store derived properties for future use
        material["solid"]["permeability_eff"][r] = kf  # list of kf_x,y,z
        material["solid"]["storage_mod"][r] = M
        material["solid"]["therm_cond_eff"][r] = lambda_e  # list lambda_x,y,z
        material["solid"]["therm_expansion_eff"][r] = alpha_e
        material["solid"]["volumetric_heat_capacity_eff"][r] = m_e
        material["solid"]["drained_bulk_mod"][r] = (1 - biot) * Ks

    return material


def _get_element_edge_lists(nne: int, nsd: int) -> dict[str, list[int]]:
    # Local node-index pairs for each element edge (distinct from face lists:
    # for 3D solids a face has >2 nodes, an edge always has 2).
    if nne == 8:  # B8
        edges = {
            "edge0": [0, 1],
            "edge1": [1, 2],
            "edge2": [2, 3],
            "edge3": [3, 0],
            "edge4": [4, 5],
            "edge5": [5, 6],
            "edge6": [6, 7],
            "edge7": [7, 4],
            "edge8": [0, 4],
            "edge9": [1, 5],
            "edge10": [2, 6],
            "edge11": [3, 7],
        }
    elif nne == 4 and nsd == 3:  # T4
        edges = {
            "edge0": [0, 1],
            "edge1": [0, 2],
            "edge2": [0, 3],
            "edge3": [1, 2],
            "edge4": [1, 3],
            "edge5": [2, 3],
        }
    elif nsd == 2:  # Q4 / T3 -- edges coincide with the face (boundary-segment) list
        return _get_element_face_lists(nne, nsd)
    else:
        raise NotImplementedError(f"Edge extraction for {nne}-node elements in {nsd} dimensions.")
    return edges


def _build_edge_element_map(mesh: dict[str, Any]) -> dict[tuple[int, int], list[int]]:
    """Map each unique mesh edge (sorted node pair) to the elements that contain it."""
    conn = mesh["conn"]
    nne = mesh["nne"]
    ne = mesh["ne"]
    nsd = mesh["nsd"]
    elem_indx = mesh["elem_indx"]

    edge_to_elems: dict[tuple[int, int], list[int]] = {}
    edge_lists = _get_element_edge_lists(nne[0], nsd)
    counter = 0

    for elem in range(ne):
        if elem > elem_indx[counter]:
            counter += 1
            edge_lists = _get_element_edge_lists(nne[counter], nsd)

        elem_conn = conn[elem]
        for local_edge in edge_lists.values():
            n1, n2 = elem_conn[local_edge[0]], elem_conn[local_edge[1]]
            key = (n1, n2) if n1 < n2 else (n2, n1)
            edge_to_elems.setdefault(key, []).append(elem)

    return edge_to_elems


def _build_mesh_edge_graph(
    mesh: dict[str, Any], edge_to_elems: dict[tuple[int, int], list[int]]
) -> scipy.sparse.csr_matrix:
    """Sparse (nn x nn) graph of unique mesh edges, weighted by edge length."""
    nodes = mesh["nodes"]

    row, col, weight = [], [], []
    for n1, n2 in edge_to_elems:
        d = float(np.linalg.norm(nodes[n1] - nodes[n2]))
        row += [n1, n2]
        col += [n2, n1]
        weight += [d, d]

    return scipy.sparse.csr_matrix((weight, (row, col)), shape=(mesh["nn"], mesh["nn"]))


_LINE_DEVIATION_TOL = 1.0e-3  # 1 mm; assumes mesh coordinates are in meters


def _clip_line_to_domain(
    A: npt.NDArray[np.float64],
    B: npt.NDArray[np.float64],
    mins: npt.NDArray[np.float64],
    maxs: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Clip segment A-B to the mesh's axis-aligned bounding box (slab method)."""
    d = B - A
    t0, t1 = 0.0, 1.0
    for axis in range(len(A)):
        if abs(d[axis]) < 1e-14:
            if A[axis] < mins[axis] or A[axis] > maxs[axis]:
                raise RuntimeError("Requested line boundary condition does not intersect the mesh domain.")
            continue
        t_near = (mins[axis] - A[axis]) / d[axis]
        t_far = (maxs[axis] - A[axis]) / d[axis]
        t_low, t_high = min(t_near, t_far), max(t_near, t_far)
        t0 = max(t0, t_low)
        t1 = min(t1, t_high)
        if t0 > t1:
            raise RuntimeError("Requested line boundary condition does not intersect the mesh domain.")

    return A + t0 * d, A + t1 * d


def _find_lineBC_connectivity(
    mesh: dict[str, Any], line: list[list[float]], edge_to_elems: dict[tuple[int, int], list[int]]
) -> tuple[list[int], list[list[int]]]:
    """Find the line boundary condition connectivity.
    Line boundary conditions find the shortest path between the start and
    end points following pre-existing nodes and edges in the mesh.

    Throws a warning if the deviation is large.
    """
    nodes = mesh["nodes"]
    nsd = mesh["nsd"]

    # Calculate line direction vector
    A = np.asarray(line[0], dtype=float)[:nsd]
    B = np.asarray(line[1], dtype=float)[:nsd]
    length_AB = float(np.linalg.norm(B - A))

    if length_AB == 0.0:
        raise RuntimeError("Line boundary condition start and end points are identical.")

    u = (B - A) / length_AB  # tangent vector

    # Clip to the domain so points prescribed outside the mesh snap to the
    # nearest node at the line/boundary intersection, not to some arbitrary
    # far-off corner node.
    mins = nodes[:, :nsd].min(axis=0)
    maxs = nodes[:, :nsd].max(axis=0)
    A_clipped, B_clipped = _clip_line_to_domain(A, B, mins, maxs)

    start_node = int(np.argmin(np.linalg.norm(nodes[:, :nsd] - A_clipped, axis=1)))
    end_node = int(np.argmin(np.linalg.norm(nodes[:, :nsd] - B_clipped, axis=1)))

    if start_node == end_node:
        raise RuntimeError("Line boundary condition start and end nodes snap to the same mesh node.")

    # Shortest path A -> B weighted by edge length, following existing mesh edges
    graph = _build_mesh_edge_graph(mesh, edge_to_elems)
    dist, predecessors = dijkstra(
        graph, directed=False, indices=np.array([start_node]), return_predecessors=True
    )
    dist, predecessors = dist[0], predecessors[0]

    if np.isinf(dist[end_node]):
        raise RuntimeError(
            "No connected path of mesh edges found between the line boundary condition endpoints."
        )

    path = [end_node]
    node = end_node
    while node != start_node:
        node = int(predecessors[node])
        path.append(node)
    path.reverse()

    line_nodes = path
    line_conn = [[path[i], path[i + 1]] for i in range(len(path) - 1)]

    # Check deviation of the snapped path from the requested line
    rel = nodes[line_nodes, :nsd] - A
    perp = rel - np.outer(rel @ u, u)
    deviation = np.linalg.norm(perp, axis=1)

    if np.any(deviation > _LINE_DEVIATION_TOL):
        warnings.warn(
            f"Exact line not found in mesh, snapping to nearest nodes. "
            f"True line distance = {length_AB}, mesh path distance = {dist[end_node]}",
            stacklevel=2,
        )

    return line_nodes, line_conn


def _resolve_group_geometry(
    mesh: dict[str, Any], group: dict[str, Any], controls: dict[str, Any], need_flux_integral: bool
) -> None:
    """Resolve node indices for one BC group and cache its flux-weight matrix, if needed.

    Shared by "hydraulics" and "thermal" so both reuse the same geometry/weights.
    """
    if "nodes" not in group:
        if "faces" in group:
            nodes: list[int] = []
            conn: list[Any] = []
            elems: list[Any] = []
            regions: list[Any] = []
            for face in group["faces"]:
                nodes.extend(mesh["boundaries"][face])
                conn.extend(mesh["boundaries"][face + "_conn"])
                elems.extend(mesh["boundaries"][face + "_elems"])
                regions.extend(mesh["boundaries"][face + "_regions"])
            group["nodes"], group["_conn"], group["_conn_kind"] = nodes, conn, "face"
            # Exterior faces have exactly one owning element each, so the element
            # at position i in "_elems" touches the face at position i in "_conn".
            group["_elems"], group["_regions"], group["_elem_edge"] = elems, regions, list(range(len(conn)))
        elif "lines" in group:
            group["nodes"] = group["line_nodes"]
            group["_conn"], group["_conn_kind"] = group["line_conn"], "line"
            # Unlike exterior faces, an interior line edge can be touched by more
            # than one element, so "_elem_edge" maps each "_elems"/"_regions" entry
            # back to the index (into "_conn") of the edge it touches.
            group["_elems"], group["_regions"], group["_elem_edge"] = (
                group["line_elems"],
                group["line_regions"],
                group["line_elem_edge"],
            )
        else:
            raise NotImplementedError("Invalid BC geometry specification")

    nodes = group["nodes"]
    if need_flux_integral and nodes and "_conn" in group:
        flux_mats = integrate_boundary_flux_matrices(
            mesh, group["_conn"], nodes, controls, group["_conn_kind"]
        )
        group["flux_weight_matrix"] = flux_mats["A"]


def _preprocess_BCs(mesh: dict[str, Any], config: dict[str, Any], verbose: bool = True) -> dict[str, Any]:  # noqa
    """Preprocessing of boundary conditions.

    1. Determine if any wellbore max pressures on/off exist

    2. During the preprocessing step, turn boundary condition groups
    defined by coordinates into node groups.

    3. Line boundary conditions need to find nodes and connectivity.
    """
    BC_groups = config["boundary_conditions"]
    nodes = mesh["nodes_total"]
    controls = config["simulation"]
    controls["wellbore_check"] = 0  # Default 0
    edge_to_elems = _build_edge_element_map(mesh) if any("lines" in group for group in BC_groups) else {}

    if np.size(nodes, 1) < 3:
        nodes = np.hstack((nodes, np.zeros((mesh["nn_total"], 1))))

    for group in BC_groups:
        BCnodes = []
        # Check if wellbore restrictions exist
        hydraulics = group.get("hydraulics")
        if hydraulics is not None and "constraints" in hydraulics:
            if verbose:
                print("Wellbore group detected:")
            config["simulation"]["wellbore_check"] += 1

            max_pressure = hydraulics["constraints"].get("maximum_pressure")
            min_pressure = hydraulics["constraints"].get("minimum_pressure")

            if max_pressure is None and min_pressure is None:
                warnings.warn(
                    "Wellbore boundary condition specified without maximum or minimum pressure.",
                    stacklevel=2,
                )

            if max_pressure is None:
                max_pressure = 1.0e100
                hydraulics["constraints"]["maximum_pressure"] = max_pressure
                if verbose:
                    print("  Maximum pressure unbounded (1.0e+100).")
            elif verbose:
                print(f"  Maximum pressure: {max_pressure:.2e}")

            if min_pressure is None:
                min_pressure = -1.0e100
                hydraulics["constraints"]["minimum_pressure"] = min_pressure
                if verbose:
                    print("  Minimum pressure unbounded (1.0e+100). \n")
            elif verbose:
                print(f"  Minimum pressure: {min_pressure:.2e} \n")

        # Convert coordinates to node numbers
        if "coords" in group:
            for xi in group["coords"]:
                if np.size(xi, 0) != 3:
                    xi.append(0.0)
                d = np.linalg.norm(nodes - xi, axis=1)
                # In the case of duplicate nodes, take the highest node number.
                # This prioritizes fracture nodes over matrix nodes.
                d_min = np.min(d)
                closest_node = np.max(np.where(d < 1e-14)).astype(int) if d_min < 1e-14 else np.argmin(d)
                BCnodes.append(closest_node)
            group["nodes"] = BCnodes

        if "lines" in group:
            for line in group["lines"]:
                line_nodes, line_conn = _find_lineBC_connectivity(mesh, line, edge_to_elems)

                # Elements touching each edge of this line: on exterior lines there is
                # exactly one element per edge, so `line_elem_edge[i]` == i naturally.
                # Interior lines can have two or more elements per edge, so
                # `line_elem_edge` maps each entry of `line_elems`/`line_regions` back
                # to the index (within the group's combined `line_conn`) of the edge it touches.
                edge_offset = len(group.get("line_conn") or [])
                line_elems: list[int] = []
                line_regions: list[Any] = []
                line_elem_edge: list[int] = []
                for local_edge_indx, edge in enumerate(line_conn):
                    n1, n2 = sorted(edge)
                    elems = edge_to_elems.get((n1, n2), [])
                    line_elems.extend(elems)
                    line_regions.extend(mesh["regions"][e] for e in elems)
                    line_elem_edge.extend([edge_offset + local_edge_indx] * len(elems))

                if group.get("line_nodes") is None:
                    group["line_nodes"] = line_nodes
                else:
                    group["line_nodes"].extend(line_nodes)

                if group.get("line_conn") is None:
                    group["line_conn"] = line_conn
                else:
                    group["line_conn"].extend(line_conn)

                if group.get("line_elems") is None:
                    group["line_elems"] = line_elems
                    group["line_regions"] = line_regions
                    group["line_elem_edge"] = line_elem_edge
                else:
                    group["line_elems"].extend(line_elems)
                    group["line_regions"].extend(line_regions)
                    group["line_elem_edge"].extend(line_elem_edge)

        need_flux_integral = any(
            field in group and group[field]["type"] == "flux" for field in ("hydraulics", "thermal")
        )
        _resolve_group_geometry(mesh, group, controls, need_flux_integral)

    config["boundary_conditions"] = BC_groups
    return config


def preprocessing(
    config_name: str, config_dir: Path | None = None, verbose: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate the config and mesh."""
    config = load_config(config_name, config_dir, verbose)
    config["mesh"]["config_dir"] = config["config_dir"]

    mesh = load_mesh(config["mesh"], verbose)

    config = _preprocess_BCs(mesh, config, verbose)

    _check_region_properties(max(mesh["regions"]), config["material"]["solid"])
    config["material"] = _calculate_derived_properties(mesh, config["material"])

    return config, mesh
