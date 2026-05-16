"""Import mesh and create mesh objects."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import yaml

# ----- Default values for config file
DEFAULTS: dict = {
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
        "rate_order": 1,
        "advection_on": 0,
        "pressurization_on": 0,
        "buoyancy_on": 0,
        "rel_error_T": 1.0e-2,
    },
}


def loadmesh(meshfile: str, nsd: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
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


def _apply_defaults(config: dict, defaults: dict, path: str = "") -> dict:
    """Check for missing values in the config file. Apply and report default values."""
    for key, default_val in defaults.items():
        full_key = f"{path}.{key}" if path else key

        if key not in config or config[key] is None:
            config[key] = default_val
            print(f"  [DEFAULT]  {full_key} = {default_val}")
        elif isinstance(default_val, dict) and isinstance(config[key], dict):
            _apply_defaults(config[key], default_val, full_key)
        else:
            print(f"  [CONFIG]   {full_key} = {config[key]}")

    return config


def load_config(config_name: str) -> dict:
    """Load config file and check for values."""
    script_dir = Path(__file__).parent
    config_path = script_dir.parent.parent / "config" / config_name
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["config_name"] = config_name
    config["config_dir"] = str(config_path.parent)

    print(f"\n ----- Configuration parameters for {config_name}")
    _apply_defaults(config, DEFAULTS)
    print()

    return config


def load_mesh(config_mesh):
    """Load mesh and define mesh structures."""
    config_path = Path(config_mesh["config_dir"]) / config_mesh["file"]
    [nodes, conn] = loadmesh(str(config_path), config_mesh["dimensions"])
    mesh = {
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

    print("----- Mesh statistics")
    print(f"  File            : {config_mesh['file']}")
    print(f"  Element type    : {etyp}")
    print(f"  Dimensions      : {mesh['nsd']}D")
    print(f"  Nodes           : {mesh['nn']}")
    print(f"  Elements        : {mesh['ne']}")
    print(f"  Quadrature pts  : {nq}^{mesh['nsd']} = {nq ** mesh['nsd']}\n")

    return mesh


def preprocessing(config_name: str) -> tuple[dict, dict]:
    """Load and validate the config and mesh.

    config : Configuration dictionary with defaults applied where parameters are missing.
    mesh : Mesh data dictionary
    """
    config = load_config(config_name)
    config["mesh"]["config_dir"] = config["config_dir"]
    mesh = load_mesh(config["mesh"])
    return config, mesh
