"""Import mesh, calculate material properties, apply defaults."""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

from .meshprocessing import load_mesh

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
            "bulk_mod": 2.0e9,  # Pa
            "density": 1000,  # kg/m^3
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
        "solver_type": "linearized",
        "output_frequency": 1,
        "output_vtk": 0,
        "output_viscosity": 0,
        "rate_order": 1,
        "fully_coupled": 0,
        "advection_on": 0,
        "pressurization_on": 0,
        "buoyancy_on": 0,
        "steady_pressure": 0,
        "steady_temperature": 0,
        "axisymmetric_on": 0,
        "rel_error_T": 1.0e-2,
        "iterative_controls": {
            "NR_tol": 1.0e-6,  # Newton-Raphson residual tolerance
            "iter_max": 100,  # Maximum number of iterations
            "aitken_relaxation": 1,  # Turn on/off Aitken's delta^2 relaxation
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
    # Viscosity
    mu = config["material"]["fluid"]["viscosity"]
    if isinstance(mu, float):  # constant
        config["simulation"]["viscosity_nonconstant"] = 0
    elif not isinstance(mu, list) or len(mu) < 1:  # not a list
        raise ValueError(f"Incompatible viscosity format: {mu}")
    elif len(mu) == 1:  # single list -> transfer to float
        config["material"]["fluid"]["viscosity"] = mu[0]
        config["simulation"]["viscosity_nonconstant"] = 0
    else:  # non-constant function
        coeff = np.zeros((4,))
        for i in range(len(mu)):
            coeff[i] = mu[i]
        if verbose:
            print(
                f"""\nViscosity = {coeff[0]:.2e} * exp({coeff[1]:.2e} / T
                + {coeff[2]:.2e} * T + {coeff[3]:.2e} * T^2 )"""
            )
            print("T is assumed in K.")

        config["material"]["fluid"]["viscosity"] = coeff
        config["simulation"]["viscosity_nonconstant"] = 1

    # Axisymmetry
    axisymmetric_on = config["simulation"]["axisymmetric_on"]
    if axisymmetric_on and config["mesh"]["dimensions"] == 3:
        config["simulation"]["axisymmetric_on"] = 0
        if verbose:
            print("Axisymmetry turned off, not valid in 3D analysis.")

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


def _check_region_properties(max_region: int, mat: dict[str, Any]) -> None:
    """Check that the correct number of properties are included."""
    missing_props = []
    for mat_prop in mat:
        if isinstance(mat[mat_prop], list):  # noqa
            if len(mat[mat_prop]) < (max_region + 1):
                missing_props.append(mat_prop)

    if len(missing_props) > 0:
        raise IndexError(f"Insufficient properties provided for:\n{chr(10).join(missing_props)}")

    return


def rotation_matrices(
    rot_x: float, rot_y: float, rot_z: float, nsd: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return individual rotation matrices (angles in degrees)."""
    rot_x = np.radians(rot_x)
    rot_y = np.radians(rot_y)
    rot_z = np.radians(rot_z)

    cx, sx = np.cos(rot_x), np.sin(rot_x)
    cy, sy = np.cos(rot_y), np.sin(rot_y)
    cz, sz = np.cos(rot_z), np.sin(rot_z)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    # extract only initial the required space dimensions
    Rx = Rx[:nsd, :nsd]
    Ry = Ry[:nsd, :nsd]
    Rz = Rz[:nsd, :nsd]

    return Rx, Ry, Rz


def _get_region_value(val: Any, region_idx: int) -> Any:
    """Return value for region, handling both scalar and list cases.

    Fallback for homogeneous cases defined as scalars instead of lists.
    """
    return val[region_idx] if isinstance(val, list) else val


def _calculate_derived_properties(mesh: dict[str, Any], mat: dict[str, Any]) -> dict[str, Any]:
    """Calculate properties derived from other primary properties."""
    nsd = mesh["nsd"]
    num_regions = max(mesh["regions"]) + 1

    # Initialize derived properties storage
    mat["solid"]["storage_mod"] = [None] * num_regions
    mat["solid"]["therm_cond_eff"] = [None] * num_regions
    mat["solid"]["therm_expansion_eff"] = [None] * num_regions
    mat["solid"]["volumetric_heat_capacity_eff"] = [None] * num_regions
    mat["solid"]["permeability_eff"] = [None] * num_regions

    for r in range(num_regions):
        # ----- material properties
        alpha_s = _get_region_value(mat["solid"]["therm_expansion"], r)
        alpha_f = mat["fluid"]["therm_expansion"]
        phi = _get_region_value(mat["solid"]["porosity"], r)
        biot = _get_region_value(mat["solid"]["biot_coeff"], r)
        Kf = mat["fluid"]["bulk_mod"]
        Ks = _get_region_value(mat["solid"]["bulk_mod"], r)
        lambda_s = _get_region_value(mat["solid"]["therm_cond"], r)
        kf = _get_region_value(mat["solid"]["permeability"], r)

        # physical properties
        angle_x = _get_region_value(mat["solid"]["rotation_x"], r)
        angle_y = _get_region_value(mat["solid"]["rotation_y"], r)
        angle_z = _get_region_value(mat["solid"]["rotation_z"], r)

        Rx, Ry, Rz = rotation_matrices(angle_x, angle_y, angle_z, nsd)
        R = Rx @ Ry @ Rz  # general rotation matrix

        # For anisotropic case, perform rotation
        if isinstance(kf, list):
            kf = R @ np.diag(kf[0:nsd]) @ R.T

        # Derived properties (computed here, not stored in YAML)
        M = Kf * Ks / (phi * Ks + (biot - phi) * Kf)  # storage modulus

        # effective expansion modulus
        alpha_e = biot * alpha_s + phi * (alpha_f - alpha_s)

        # effective conductivity
        if isinstance(lambda_s, list):
            # anisotropic case
            lambda_e = (
                phi * mat["fluid"]["therm_cond"] * np.eye(nsd) + (1 - phi) * R @ np.diag(lambda_s) @ R.T
            )
        else:
            # isotropic case
            lambda_e = phi * mat["fluid"]["therm_cond"] + (1 - phi) * lambda_s

        # effective volumetric heat capacity
        cp_s = _get_region_value(mat["solid"]["specific_heat"], r)
        rho_s = _get_region_value(mat["solid"]["density"], r)
        m_e = phi * mat["fluid"]["specific_heat"] * mat["fluid"]["density"] + (1 - phi) * cp_s * rho_s

        # store derived properties for future use
        mat["solid"]["permeability_eff"][r] = kf  # list of kf_x,y,z
        mat["solid"]["storage_mod"][r] = M
        mat["solid"]["therm_cond_eff"][r] = lambda_e  # list lambda_x,y,z
        mat["solid"]["therm_expansion_eff"][r] = alpha_e
        mat["solid"]["volumetric_heat_capacity_eff"][r] = m_e

    return mat


def _convert_coordBCs_to_nodeBCs(mesh: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """During the preprocessing step, turn boundary condition groups
    defined by coordinates into node groups.
    """
    BC_dict = config["boundary_conditions"]
    nodes = mesh["nodes"]
    if np.size(nodes, 1) < 3:
        nodes = np.hstack((nodes, np.zeros((mesh["nn"], 1))))

    for BC_type in BC_dict:
        for group in BC_dict[BC_type]:
            BCnodes = []
            if "coords" in group:
                for xi in group["coords"]:
                    if np.size(xi, 0) != 3:
                        xi.append(0.0)
                    d = np.linalg.norm(nodes - xi, axis=1)
                    closest_node = np.argmin(d).astype(int)
                    BCnodes.append(closest_node)
                group["nodes"] = BCnodes

    config["boundary_conditions"] = BC_dict
    return config


def preprocessing(
    config_name: str, config_dir: Path | None = None, verbose: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate the config and mesh."""
    config = load_config(config_name, config_dir, verbose)
    config["mesh"]["config_dir"] = config["config_dir"]

    mesh = load_mesh(config["mesh"], verbose)

    config = _convert_coordBCs_to_nodeBCs(mesh, config)

    _check_region_properties(max(mesh["regions"]), config["material"]["solid"])
    config["material"] = _calculate_derived_properties(mesh, config["material"])

    return config, mesh
