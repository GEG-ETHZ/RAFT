"""Import mesh and create mesh objects."""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

# ----- Default values for config file
DEFAULTS: dict[str, Any] = {
    # ---- mesh ----------------------------------------------------------------
    "mesh": {
        "file": "box3D.msh",
        "dimensions": 3,
    },
    # ---- material ------------------------------------------------------------
    "material": {
        "solid": {
            "bulk_mod": 20.0e9,  # Pa
            "density": 2600.0,  # kg/m^3
            "therm_cond": 3.0,  # W/mK
            "specific_heat": 800.0,  # J/kgK
            "therm_expansion": 12.0e-6,  # 1/K
        },
        "fluid": {
            "bulk_mod": 2.0e9,  # Pa
            "density": 1000,  # kg/m^3
            "therm_cond": 0.6,  # W/mK
            "therm_expansion": 200.0e-6,  # 1/K
            "specific_heat": 4180.0,  # J/kgK
            "viscosity": 1e-3,  # Pa·s
        },
        "porosity": 0.01,  # -
        "biot_coeff": 0.5,  # -
        "permeability": 1.0e-16,  # m^2
        "rotation_x": 0.0,  # rotation of local coordinates around x-axis (degrees)
        "rotation_y": 0.0,  # rotation of local coordinates around y-axis (degrees)
        "rotation_z": 0.0,  # rotation of local coordinates around z-axis (degrees)
    },
    # ---- initial conditions --------------------------------------------------
    "initial_conditions": {
        "geothermal_gradient": 0.0,  # deg C/m
        "T_surf": 20.0,  # deg C
        "pressure_gradient": 0.0,  # Pa/m
        "p_surf": 1.0e5,  # Pa
    },
    # ---- boundary conditions -------------------------------------------------
    "boundary_conditions": {
        "pressure_fixed": [{"nodes": [], "values": []}],
        "temperature_fixed": [{"nodes": [], "values": []}],
        "fluid_flux": [{"nodes": [], "values": []}],
        "temperature_flux": [{"nodes": [], "values": []}],
    },
    # ---- simulation ----------------------------------------------------------
    "simulation": {
        "t_start": 0.0,
        "t_end": 100.0,
        "timestep": 1.0,
        "timestep_min": 1.0,
        "timestep_max": 1.0,
        "output_frequency": 1,
        "output_vtk": 0,
        "rate_order": 1,
        "advection_on": 0,
        "pressurization_on": 0,
        "buoyancy_on": 0,
        "steady_pressure": 0,
        "steady_temperature": 0,
        "rel_error_T": 1.0e-2,
    },
}


def load_Gmsh(meshfile: str, nsd: int) -> tuple[npt.NDArray[np.float64] | None, npt.NDArray[np.int64] | None]:
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

    if verbose:
        print()

    return config


def _extract_exterior(mesh: dict[str, Any]):
    """Find the nodes along the exterior for applying boundary conditions.
    Note that this extraction assumes the reservoir has 6 sides that do not have to be planar.
    """
    conn = mesh["conn"]
    nodes = mesh["nodes"]
    nne = mesh["nne"]
    ne = mesh["ne"]

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
    else:
        raise NotImplementedError(f"Boundary extraction for {nne}-node elements")

    # Loop through all elements, collect a list of faces in global coordinates,
    # then find unique faces.

    # use dictionary to track faces and keep original order for calculating normals.
    # dictionary structure: sorted_face: [count, original face]
    face_tracker: dict[tuple[int, ...], list[int | tuple[int, ...]]] = {}

    for elem in range(ne):
        for face_name, face_local in faces.items():
            original_face = tuple(conn[elem, face_local])  # for normals later
            sorted_face = tuple(sorted(original_face))  # for uniqueness
            if sorted_face in face_tracker:
                # Increment count
                face_tracker[sorted_face][0] += 1  # type: ignore[operator]
            else:
                # Store count and original
                face_tracker[sorted_face] = [1, original_face]

    boundary_faces = [data[1] for data in face_tracker.values() if data[0] == 1]

    # Loop through boundary faces, calculate the normal of each face and sort it based on its closest axis
    x_min: set[int] = set()
    x_max: set[int] = set()
    y_min: set[int] = set()
    y_max: set[int] = set()
    z_min: set[int] = set()
    z_max: set[int] = set()

    for face in boundary_faces:
        # calculate plane normal
        xI = nodes[face, :]
        v1 = [xI[1, 0] - xI[0, 0], xI[1, 1] - xI[0, 1], xI[1, 2] - xI[0, 2]]
        v2 = [xI[2, 0] - xI[0, 0], xI[2, 1] - xI[0, 1], xI[2, 2] - xI[0, 2]]
        n = np.cross(v1, v2)

        # project n onto axis normals
        ex = n @ [1, 0, 0]
        ey = n @ [0, 1, 0]
        ez = n @ [0, 0, 1]

        # Find which component has the largest magnitude
        components = {"x": ex, "y": ey, "z": ez}
        max_component = max(components, key=lambda k: abs(components[k]))

        # Append to appropriate list based on sign
        if max_component == "x":
            (x_max if ex > 0 else x_min).update(face)  # type: ignore[arg-type]
        elif max_component == "y":
            (y_max if ey > 0 else y_min).update(face)  # type: ignore[arg-type]
        else:  # max_component == 'z'
            (z_max if ez > 0 else z_min).update(face)  # type: ignore[arg-type]

    mesh["boundaries"] = {}
    mesh["boundaries"]["x_max"] = x_max
    mesh["boundaries"]["x_min"] = x_min
    mesh["boundaries"]["y_max"] = y_max
    mesh["boundaries"]["y_min"] = y_min
    mesh["boundaries"]["z_max"] = z_max
    mesh["boundaries"]["z_min"] = z_min

    x_all = x_max | x_min
    y_all = y_max | y_min
    z_all = z_max | z_min
    n_all = x_all | y_all | z_all

    mesh["boundaries"]["x_all"] = x_all
    mesh["boundaries"]["y_all"] = y_all
    mesh["boundaries"]["z_all"] = z_all
    mesh["boundaries"]["n_all"] = n_all

    return mesh


def load_mesh(config_mesh: dict[str, Any], verbose: bool):
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

    # Define number of quadrature points
    match (mesh["nsd"], mesh["nne"]):
        case (3, 8):  # B8
            nq, etyp = 2, "B8"
        case (3, 27):  # B27
            nq, etyp = 3, "B27"
        case (2, 4):  # Q4
            nq, etyp = 2, "Q4"
        case (2, 9):  # Q9
            nq, etyp = 3, "Q9"
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


def preprocessing(
    config_name: str, config_dir: Path | None = None, verbose: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate the config and mesh."""
    config = load_config(config_name, config_dir, verbose)
    config["mesh"]["config_dir"] = config["config_dir"]
    mesh = load_mesh(config["mesh"], verbose)
    return config, mesh
