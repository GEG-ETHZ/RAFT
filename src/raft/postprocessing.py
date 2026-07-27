"""Postprocessing of FEM file."""

import bisect
import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .matprops import get_viscosity
from .shapefunctions import getB
from .writevtk import write_mesh2vtk

# Default output directory: /main_folder/outputs/
# (two levels up from /main_folder/src/postprocessing.py)
_DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs"


def _get_nodal_parents_coords(etype: str) -> npt.NDArray[np.float64]:
    """Return element-local nodal coordinates (xi) used to extrapolate flux to the nodes, by element type."""
    if etype == "Q4":  # Q4 element
        xi = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
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
    elif etype == "T3":
        xi = np.array([[0, 0], [1, 0], [0, 1]])
    elif etype == "L2":
        xi = np.array([[-1.0], [1.0]])
    elif etype == "Q9":  # Q9 element
        xi = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
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
    return xi


def compute_fluid_flux(  # noqa
    mesh: dict[str, Any],
    mat: dict[str, Any],
    p: npt.NDArray[np.float64],
    T: npt.NDArray[np.float64],
    isfracture: bool,
) -> npt.NDArray[np.float64]:
    """Computes fluid flux at each node using nodal averaging.

    Returns nsd x nn array of fluxes at each node.
    """
    from .shapefunctions import GNMATRIX_FUNCS

    nn = mesh["nn"]
    nsd = mesh["nsd"]
    nne_list = mesh["nne"]
    conn = mesh["conn"]  # ragged list - each entry is that element's local node list
    nodes = mesh["nodes"]
    etype_list = mesh["element_type"]
    elem_indx = mesh["elem_indx"]
    region_lists = mesh["region_lists"]
    if isfracture:
        num_regions = len(region_lists)
        nsd -= 1  # reduce dofs
        nodes_global = nodes
    else:
        num_regions = mesh["num_regions"]

    # Initialize output arrays
    q = np.zeros((nsd, nn))
    count = np.zeros(nn)

    # Material properties - fluid
    mu = mat["fluid"]["viscosity"]
    mu_nonlinear = False
    if isinstance(mu, np.ndarray):
        mu_nonlinear = True
        mu_coeff = mu

    for r in range(num_regions):
        if isfracture:
            w = mat["fracture"]["aperture"][r]
            kf = w**3 / 12  # longitudinal permeability
            isotropic = True
            u = mesh["fracture_tangent"][r]
            # Project the mesh nodes into local coordinates
            nodes = (nodes_global @ u).reshape(-1, 1)
        else:
            # solid properties - by region
            kf = mat["solid"]["permeability_eff"][r]
            # Check for region isotropy
            isotropic = bool(isinstance(kf, (float, int)))

        # find element type of first element in region
        elem_type_counter = -1  # force an update on the first element

        for e in region_lists[r]:
            new_type_idx = bisect.bisect_left(elem_indx, e)
            if new_type_idx != elem_type_counter:
                # Update for different element type
                etype = etype_list[new_type_idx]
                nne = nne_list[new_type_idx]
                getGN = GNMATRIX_FUNCS[etype]
                xi = _get_nodal_parents_coords(etype)
                GN_q_list = [getGN(xi[n, :]) for n in range(nne)]
                GN_q = np.array(GN_q_list)
                elem_type_counter = new_type_idx

            enodes = conn[e]  # element nodes

            xI = nodes[enodes, :nsd]  # element physical coordinates
            B_q = getB(GN_q, xI, nsd)[0]

            pe = p[enodes]  # element nodal pressures
            if mu_nonlinear:
                Te = T[enodes]
                mu, _ = get_viscosity(Te, mu_coeff)
                mu = mu[:, None]

            gradp_all = B_q @ pe
            flux_e = -(kf / mu) * gradp_all if isotropic else -(gradp_all @ kf.T) / mu

            q[:, enodes] += flux_e.T
            count[enodes] += 1

    # Nodal averaging of fluxes
    # Avoid division by zero for unused nodes
    nonzero_count = np.where(count == 0, 1, count)

    for n in range(nsd):
        q[n, :] /= nonzero_count

    return q


def _extract_cartesian_properties(
    material_list: list[Any], indx: npt.NDArray[np.int64], nsd: int
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
            if nsd == 3:
                region_z.append(item[2, 2])
            else:
                region_z.append(0.0)

    # Map the per-region values to individual elements using the index mapping
    return (
        np.array(region_x)[indx],
        np.array(region_y)[indx],
        np.array(region_z)[indx],
    )


def _export_flow_logs(
    n: int,
    primary_variables: dict[str, Any],
    controls: dict[str, Any],  # noqa
    BC_config: dict[str, Any],
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
        for entry in BC_config.get(bc_type, []):
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


def _sanitize(data: npt.NDArray[np.float64], threshold: float = 1e-15) -> npt.NDArray[np.float64]:
    """Zero out values smaller than threshold to avoid noisy near-zero VTK output."""
    data[np.abs(data) < threshold] = 0
    return data


def _prepare_output_directory(controls: dict[str, Any]) -> str:
    """Resolve this run's output directory, creating it if needed, and record it on config."""
    base_path = controls.get("output_directory", _DEFAULT_OUTPUT_DIR)
    config_stem = Path(controls["config_name"]).stem
    io_path = os.path.join(base_path, config_stem)
    controls["output_directory_full"] = io_path

    if not os.path.exists(io_path):
        os.makedirs(io_path)

    return io_path


def _build_point_scalars(
    controls: dict[str, Any],
    mesh: dict[str, Any],
    mat: dict[str, Any],
    p: npt.NDArray[np.float64],
    T: npt.NDArray[np.float64],
) -> list[dict[str, Any]]:
    """Build the per-node scalar fields (temperature, pressure, optional viscosity) for VTK export."""
    scalardata_point = [
        {"name": "T", "data": _sanitize(T.copy())},
        {"name": "p", "data": _sanitize(p.copy())},
    ]
    if controls["output_viscosity"]:
        if controls["viscosity_nonconstant"]:
            mu, _ = get_viscosity(T, mat["fluid"]["viscosity"])
        else:
            mu = mat["fluid"]["viscosity"] * np.ones(mesh["nn"])
        scalardata_point.append({"name": "viscosity", "data": _sanitize(mu.copy())})

    return scalardata_point


def _build_boundary_group_scalars(mesh: dict[str, Any]) -> list[dict[str, Any]]:
    """Build one indicator scalar field per named BC group, for the initial-condition VTK snapshot.

    Skips any group whose name ends in "_conn", since those are connectivity bookkeeping
    entries rather than physical boundary groups and shouldn't be exported as fields.
    """
    scalardata_point = []
    BCgroups = mesh["boundaries"]
    for name in BCgroups:
        if name.endswith("_conn"):
            continue
        label = "boundary_" + name
        data = np.zeros(mesh["nn"])
        data[np.asarray(list(BCgroups[name]))] = 1
        scalardata_point.append({"name": label, "data": data})

    return scalardata_point


def _build_initial_cell_data(
    mesh: dict[str, Any], material: dict[str, Any], nsd: int
) -> list[dict[str, Any]]:
    """Build per-element (cell) material property fields for the initial-condition VTK snapshot."""
    indx = np.array(mesh["regions"])
    # Extract and handle potential isotropic vs anisotropic permeability tensor
    perm_x, perm_y, perm_z = _extract_cartesian_properties(material["solid"]["permeability_eff"], indx, nsd)
    cond_x, cond_y, cond_z = _extract_cartesian_properties(material["solid"]["therm_cond_eff"], indx, nsd)

    return [
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
            "data": np.atleast_1d(np.asarray(material["solid"]["volumetric_heat_capacity_eff"]))[indx],
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
        {"name": "effective_permeability_x", "data": perm_x},
        {"name": "effective_permeability_y", "data": perm_y},
        {"name": "effective_permeability_z", "data": perm_z},
        {"name": "effective_conductivity_x", "data": cond_x},
        {"name": "effective_conductivity_y", "data": cond_y},
        {"name": "effective_conductivity_z", "data": cond_z},
    ]


def _build_fracture_cell_data(
    mesh_fracture: dict[str, Any], material: dict[str, Any]
) -> list[dict[str, Any]]:
    """Build per-element aperture and heat transfer coefficient fields for the fracture VTK output.

    Each fracture region (indexed the same way as in compute_fluid_flux) has a scalar
    aperture and Nusselt number. The heat transfer coefficient is computed as
    h = nusselt * lambda_f / 2 / aperture, where lambda_f is the fluid thermal conductivity.
    """
    region_lists = mesh_fracture["region_lists"]
    num_regions = len(region_lists)

    apertures = material["fracture"]["aperture"]
    nusselts = material["fracture"]["nusselt"]
    lambda_f = material["fluid"]["therm_cond"]

    nelem = len(mesh_fracture["conn"])
    aperture_cell = np.zeros(nelem)
    h_cell = np.zeros(nelem)

    for r in range(num_regions):
        w = apertures[r]
        nu = nusselts[r]
        h = nu * lambda_f / 2 / w

        elems = region_lists[r]
        aperture_cell[elems] = w
        h_cell[elems] = h

    return [
        {"name": "aperture", "data": _sanitize(aperture_cell)},
        {"name": "heat_transfer_coefficient", "data": _sanitize(h_cell)},
    ]


def _write_matrix_output(
    n: int,
    io_path: str,
    mesh: dict[str, Any],
    controls: dict[str, Any],  # noqa
    material: dict[str, Any],
    primary_variables: dict[str, Any],
    verbose: bool,
) -> None:
    """Compute fluxes and write one solution to VTK for the matrix."""
    if verbose:
        print("     Writing matrix to output file...")

    nn = mesh["nn"]
    nsd = mesh["nsd"]
    filename = os.path.join(io_path, f"Matrix.vtk.{n}")

    # ----- Extract primary variables
    p = primary_variables["p"][0:nn, 0]
    T = primary_variables["T"][0:nn, 0]

    # --- Calculate fluxes
    flux = compute_fluid_flux(mesh, material, p, T, False)

    scalardata_point = _build_point_scalars(controls, mesh, material, p, T)
    if n == 0:  # In the initial condition, also export BC groups
        scalardata_point += _build_boundary_group_scalars(mesh)

    vectordata_point = [{"name": "flux", "data": _sanitize(flux.T)}]

    # In the initial condition, also export material properties and regions
    scalardata_cell = _build_initial_cell_data(mesh, material, nsd) if n == 0 else []

    write_mesh2vtk(
        filename,
        "TANGO",
        mesh["nodes"],
        mesh["conn"],
        scalardata_point,
        vectordata_point,
        scalardata_cell,
    )


def _write_fracture_output(
    n: int,
    io_path: str,
    mesh: dict[str, Any],
    controls: dict[str, Any],  # noqa
    material: dict[str, Any],
    primary_variables: dict[str, Any],
    verbose: bool,
) -> None:
    """Compute fluxes and write one solution to VTK for the fracture."""
    if verbose:
        print("     Writing fractures to output file...")

    nn = mesh["nn"]
    nn_tot = mesh["nn_total"]
    filename = os.path.join(io_path, f"Fracture.vtk.{n}")

    # ----- Extract primary variables
    p = primary_variables["p"][nn:(nn_tot), 0]
    T = primary_variables["T"][nn:(nn_tot), 0]

    # --- Calculate fluxes
    flux = compute_fluid_flux(mesh["fracture_mesh"], material, p, T, True)

    scalardata_point = _build_point_scalars(controls, mesh["fracture_mesh"], material, p, T)

    vectordata_point = [{"name": "flux", "data": _sanitize(flux.T)}]

    scalardata_cell = _build_fracture_cell_data(mesh["fracture_mesh"], material)

    write_mesh2vtk(
        filename,
        "TANGO",
        mesh["fracture_mesh"]["nodes"],
        mesh["fracture_mesh"]["conn"],
        scalardata_point,
        vectordata_point,
        scalardata_cell,
    )


def postprocessing(
    n: int,
    mesh: dict[str, Any],
    controls: dict[str, Any],
    material: dict[str, Any],
    BC_config: dict[str, Any],
    primary_variables: dict[str, Any],
    t: float,
    verbose: bool,
) -> dict[str, Any]:
    """Postprocessing of FEM data.

    Calculates fluxes, stresses,and Peclet numbers, then exports the results to a VTK file.
    """
    io_path = _prepare_output_directory(controls)
    output_now = (n % controls["output_frequency"]) == 0

    if output_now and controls["output_vtk"]:
        _write_matrix_output(n, io_path, mesh, controls, material, primary_variables, verbose)

    if output_now and controls["output_vtk"] and controls["fracture_on"]:
        _write_fracture_output(n, io_path, mesh, controls, material, primary_variables, verbose)

    if output_now:
        _export_flow_logs(n, primary_variables, controls, BC_config, io_path, t)

    return controls
