"""Postprocessing of FEM file."""

import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .writevtk import write_mesh2vtk

# Default output directory: /main_folder/outputs/
# (two levels up from /main_folder/src/postprocessing.py)
_DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs"


def compute_fluid_flux(
    mesh: dict[str, Any], mat: dict[str, Any], p: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Computes fluid flux at each node using nodal averaging.

    Returns nsd x nn array of fluxes at each node.
    """
    from .shapefunctions import getBmatrix

    nn = mesh["nn"]
    nsd = mesh["nsd"]
    ne = mesh["ne"]
    nne = mesh["nne"]
    regions = mesh["regions"]
    conn = mesh["conn"]
    nodes = mesh["nodes"]
    etype = mesh["element_type"]

    # Initialize output arrays
    q = np.zeros((nsd, nn))
    count = np.zeros(nn)

    # Material properties - fluid
    mu = mat["fluid"]["viscosity"]

    # Local coordinates (xi) based on element type
    if etype == "Q4":  # Q4 element
        xi = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    elif etype == "Q9":  # Q9 element
        xi = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    elif etype == "B8":  # B8 element
        xi = np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ]
        )
    elif etype == "T4":
        xi = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    elif etype == "B27":  # B27 element
        xi = np.array(
            [
                [-1, -1, -1],
                [-1, 1, -1],
                [1, 1, -1],
                [1, -1, -1],
                [-1, -1, 1],
                [-1, 1, 1],
                [1, 1, 1],
                [1, -1, 1],
                [-1, 0, -1],
                [0, 1, -1],
                [1, 0, -1],
                [0, -1, -1],
                [-1, 0, 1],
                [0, 1, 1],
                [1, 0, 1],
                [0, -1, 1],
                [-1, -1, 0],
                [-1, 1, 0],
                [1, 1, 0],
                [1, -1, 0],
                [0, 0, -1],
                [0, 0, 1],
                [-1, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, 0],
            ]
        )
    else:
        raise ValueError(f"Unsupported element type: {etype}")

    # Loop over elements
    for e in range(ne):
        enodes = conn[e, :]  # element nodes
        # material properties - element
        kf = mat["solid"]["permeability_eff"][regions[e]]

        xI = nodes[enodes, :nsd]  # element physical coordinates
        pe = p[enodes]  # element nodal pressures

        # Loop over nodes within the element
        for n in range(nne):
            B, _ = getBmatrix(xi[n, :], xI, etype)

            # Fluid flux calculation
            flux_val = -np.dot(kf / mu, (B @ pe))
            q[:, enodes[n]] += flux_val

            count[enodes[n]] += 1

    # Nodal averaging of fluxes
    # Avoid division by zero for unused nodes
    nonzero_count = np.where(count == 0, 1, count)

    for n in range(nsd):
        q[n, :] /= nonzero_count

    return q


def _extract_cartesian_properties(
    material_list: list[Any], indx: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Extracts x, y, and z directional components from a list of region properties.

    Handles both anisotropic 3x3 tensors and isotropic scalars per region,
    mapping them to elements via the provided region index array.
    """
    region_x = []
    region_y = []
    region_z = []

    for item in material_list:
        if isinstance(item, float):  # Isotropic scalar
            region_x.append(item)
            region_y.append(item)
            region_z.append(item)
        else:  # ansisotropic tensor
            region_x.append(item[0, 0])
            region_y.append(item[1, 1])
            region_z.append(item[2, 2])

    # Map the per-region values to individual elements using the index mapping
    return (
        np.array(region_x)[indx],
        np.array(region_y)[indx],
        np.array(region_z)[indx],
    )


def _export_flow_logs(
    n: int,
    primary_variables: dict[str, Any],
    config: dict[str, Any],
    io_path: str,
    t: float,
) -> None:
    """Export flow logs from BC groups.

    Loops over the Neumann and Dirichlet boundary conditions on the flow
    problem and exports pressure, flow rate, and temperature associated with
    each tagged BC group. Only exports tagged groups. Flows and pressures are
    summed over the group and temperatures are taken as a flow weighted mean.
    """
    # ---- File name
    name = "outflow.log"
    filename = os.path.join(io_path, name)  # results folder path

    # ---- Collect tagged groups
    bc_types_to_log = ("pressure_fixed", "fluid_flux")
    tagged_groups: list[dict[str, Any]] = []
    for bc_type in bc_types_to_log:
        for entry in config["boundary_conditions"].get(bc_type, []):
            if "tag" in entry:
                tagged_groups.append(entry)

    # ---- Clear file on first pass
    if n == 0:
        with open(filename, "w") as fid:
            fid.write("time\t")
            fid.write(
                "\t".join(
                    f"{grp['tag']}_pressure\t{grp['tag']}_flow\t{grp['tag']}_T" for grp in tagged_groups
                )
            )
            fid.write("\n")

    # Extract and group flows
    p = primary_variables["p"][:, 1].flatten()
    q = primary_variables["q"].flatten()
    T = primary_variables["T"][:, 1].flatten()

    outflow_values: list[float] = []
    for group in tagged_groups:
        indx = np.asarray(group["nodes"])

        p_group = p[indx]
        q_group = q[indx]
        T_group = T[indx]

        p_avg = np.mean(p_group)
        q_sum = sum(q_group)
        T_avg = np.dot(q_group, T_group) / q_sum if abs(q_sum) > 1e-12 else np.mean(T_group)

        outflow_values.append(p_avg)
        outflow_values.append(q_sum)
        outflow_values.append(T_avg)

    # ----- Extract primary variables
    with open(filename, "a") as fid:
        fid.write(f"{t:.6e}\t")
        fid.write("\t".join(f"{val:.6e}" for val in outflow_values))
        fid.write("\n")

    return


def postprocessing(
    n: int,
    mesh: dict[str, Any],
    config: dict[str, Any],
    material: dict[str, Any],
    primary_variables: dict[str, Any],
    t: float,
    verbose: bool,
) -> dict[str, Any]:
    """Postprocessing of FEM data.

    Calculates fluxes, stresses,and Peclet numbers, then exports the results to a VTK file.
    """
    # --- Path Management
    base_path = config["simulation"].get("output_directory", _DEFAULT_OUTPUT_DIR)
    config_stem = Path(config["config_name"]).stem
    io_path = os.path.join(base_path, config_stem)
    config["output_directory_full"] = io_path

    if not os.path.exists(io_path):
        os.makedirs(io_path)

    # --- Output vtk, frequency is controlled by user user input
    if (n % config["simulation"]["output_frequency"]) == 0 and config["simulation"]["output_vtk"]:
        if verbose:
            print("     Writing to output file...")

        # ---- File name
        name = f"Solution.vtk.{n}"
        filename = os.path.join(io_path, name)

        # ----- Extract primary variables
        p = primary_variables["p"][:, 0]
        T = primary_variables["T"][:, 0]

        # --- Calculate fluxes
        # Assumes compute_fluid_flux is ported from ComputeFluidFlux.m
        flux = compute_fluid_flux(mesh, config["material"], p)

        # --- Sanitize and collect scalar data
        def sanitize(data: npt.NDArray[np.float64], threshold: float = 1e-15) -> npt.NDArray[np.float64]:
            data[np.abs(data) < threshold] = 0
            return data

        # Point Data
        # --- Collect scalar data in a dictionary for export
        scalardata_point = [
            {"name": "T", "data": sanitize(T.copy())},
            {"name": "p", "data": sanitize(p.copy())},
        ]

        if n == 0:  # In the initial condition, also export BC groups
            BCgroups = mesh["boundaries"]
            for name in BCgroups:
                label = "boundary_" + name
                data = np.zeros(mesh["nn"])
                data[np.asarray(list(BCgroups[name]))] = 1
                scalardata_point.append({"name": label, "data": data})

        # --- Vector data
        vectordata_point = [{"name": "flux", "data": sanitize(flux.T)}]

        # Cell Data
        if n == 0:  # In the initial condition, export material properties and regions
            indx = np.array(mesh["regions"])
            # Extract and handle potential isotropic vs anisotropic permeability tensor
            perm_x, perm_y, perm_z = _extract_cartesian_properties(
                material["solid"]["permeability_eff"], indx
            )
            cond_x, cond_y, cond_z = _extract_cartesian_properties(material["solid"]["therm_cond_eff"], indx)

            scalardata_cell = [
                {"name": "region", "data": mesh["regions"].copy()},
                {"name": "porosity", "data": np.atleast_1d(np.asarray(material["solid"]["porosity"]))[indx]},
                {
                    "name": "biot_coefficient",
                    "data": np.atleast_1d(np.asarray(material["solid"]["biot_coeff"]))[indx],
                },
                {
                    "name": "storage_modulus",
                    "data": np.atleast_1d(np.asarray(material["solid"]["storage_mod"]))[indx],
                },
                {
                    "name": "thermal_expansion_effective",
                    "data": np.atleast_1d(np.asarray(material["solid"]["therm_expansion_eff"]))[indx],
                },
                {
                    "name": "volumetric_heatcapacity",
                    "data": np.atleast_1d(np.asarray(material["solid"]["volumetric_heat_capacity_eff"]))[
                        indx
                    ],
                },
                {
                    "name": "rotation_x",
                    "data": np.atleast_1d(np.asarray(material["solid"]["rotation_x"]))[indx],
                },
                {
                    "name": "rotation_y",
                    "data": np.atleast_1d(np.asarray(material["solid"]["rotation_y"]))[indx],
                },
                {
                    "name": "rotation_z",
                    "data": np.atleast_1d(np.asarray(material["solid"]["rotation_z"]))[indx],
                },
                {
                    "name": "effective_permeability_x",
                    "data": perm_x,
                },
                {
                    "name": "effective_permeability_y",
                    "data": perm_y,
                },
                {
                    "name": "effective_permeability_z",
                    "data": perm_z,
                },
                {
                    "name": "effective_conductivity_x",
                    "data": cond_x,
                },
                {
                    "name": "effective_conductivity_y",
                    "data": cond_y,
                },
                {
                    "name": "effective_conductivity_z",
                    "data": cond_z,
                },
            ]
        else:
            scalardata_cell = []

        # ----- Export to vtk
        write_mesh2vtk(
            filename,
            "TANGO",
            mesh["nodes"],
            mesh["conn"],
            scalardata_point,
            vectordata_point,
            scalardata_cell,
        )

    # --- Output text files, frequency is controlled by user user input
    if (n % config["simulation"]["output_frequency"]) == 0:
        _export_flow_logs(n, primary_variables, config, io_path, t)

    return config
