"""Parse the input boundary conditions and translate them into FEM input."""

from typing import Any

import numpy as np
import numpy.typing as npt

from .matrixassembly import integrate_boundary_flux_matrices


def _createBCpairs(incoming: dict[str, Any]) -> npt.NDArray[np.float64]:
    pairs = list(zip(incoming["nodes"], incoming["BCvalues"]))
    if pairs:
        return np.array(pairs, dtype=np.float64)
    return np.empty((0, 2), dtype=np.float64)


def evaluateBCfunctions(incoming: dict[str, Any], t: float, nodes: npt.NDArray[np.float64]) -> dict[str, Any]:
    """Evaluate functions defining the boundary condition values."""
    match incoming["function"]:
        case "constant":
            incoming["BCvalues"] = [incoming["values"][0]] * len(incoming["nodes"])
        case "linear":
            incoming["BCvalues"] = [t * incoming["values"][0] + incoming["values"][1]] * len(
                incoming["nodes"]
            )
        case "gradient":
            s = incoming["values"][0]
            s = s / np.linalg.norm(s)  # unit vector in gradient direction
            y0 = incoming["values"][1]  # reference gradient value at min point
            slope = incoming["values"][2]  # gradient slope

            indx_nodes = incoming["nodes"]

            x = nodes[indx_nodes, :] @ s
            x_min = min(x)

            incoming["BCvalues"] = y0 + slope * (x - x_min)
        case "ramp":
            y_0 = incoming["values"][0][0]
            y_1 = incoming["values"][0][1]
            t_ramp = incoming["values"][1]
            y = y_1 if t >= t_ramp else y_0 + t / t_ramp * (y_1 - y_0)
            incoming["BCvalues"] = [y] * len(incoming["nodes"])
        case "table":
            tn = incoming["values"][0]
            yn = incoming["values"][1]
            y = np.interp(t, tn, yn) if t >= tn[0] else 0
            incoming["BCvalues"] = [y] * len(incoming["nodes"])
    return incoming


def _gatherBCs(  # noqa: C901
    mesh: dict[str, Any], t: float, incoming: list[dict[str, Any]], bctype: int, controls: dict[str, Any]
) -> npt.NDArray[np.float64]:
    """bctype: 1 - Dirichlet, 2 - Neumann.

    Boundary condition type necessary for flux integrals, not yet implemented.
    """
    BCarray = np.empty((0, 2))
    # for each BC group
    for group in incoming:
        face_flag = False
        line_flag = False
        face_conn = []
        face_nn = 0
        # gather all the corresponding nodes
        if "nodes" in group:
            # directly append the nodes
            # covers "nodes" and "coords" cases, which have been mapped in pre-processing
            BCnodes = group["nodes"]

            # on second passes, check if nodes is an amalgamation of faces
            if "faces" in group:
                face_flag = True
            if "lines" in group:
                line_flag = True
        elif "faces" in group:
            # load the groups of faces determined in the pre-processing
            BCnodes = []
            for face in group["faces"]:
                BCnodes.extend(mesh["boundaries"][face])
                if bctype == 2:
                    face_flag = True
                    face_conn.extend(mesh["boundaries"][face + "_conn"])
                    face_nn += len(mesh["boundaries"][face])
        elif "lines" in group:
            # load the line groups determined in the pre-processing
            BCnodes = []
            line_conn = group["line_conn"]
            BCnodes.extend(group["line_nodes"])
            if bctype == 2:
                line_flag = True
        else:
            raise NotImplementedError("Invalid BC geometry specification")

        # if the group has no specified nodes, skip it
        if not BCnodes:
            continue

        # save collected nodes to group
        group["nodes"] = BCnodes

        # evaluate the value of the nodes
        if "function" not in group:
            group["function"] = "constant"  # assume constant value if no function given

        # get the precribed value of the BC group
        group = evaluateBCfunctions(group, t, mesh["nodes"])
        # Evaluate face flux integrals
        if bctype == 2 and face_flag:
            if "flux_weight_matrix" not in group:
                flux_mats = integrate_boundary_flux_matrices(
                    mesh, face_conn, group["nodes"], controls, "face"
                )
                group["flux_weight_matrix"] = flux_mats["A"]

            group["BCvalues"] = group["flux_weight_matrix"] @ np.asanyarray(group["BCvalues"])
        # Evaluate line flux integrals
        if bctype == 2 and line_flag:
            if "flux_weight_matrix" not in group:
                flux_mats = integrate_boundary_flux_matrices(
                    mesh, line_conn, group["nodes"], controls, "line"
                )
                group["flux_weight_matrix"] = flux_mats["A"]

            group["BCvalues"] = group["flux_weight_matrix"] @ np.asanyarray(group["BCvalues"])

        # combine the group into a single array
        grouparray = _createBCpairs(group)

        # collect all applied BCs
        BCarray = np.vstack((BCarray, grouparray))

    # Ensure node specifications are unique
    if BCarray.size > 0:
        _, idx = np.unique(BCarray[:, 0], return_index=True)
        BCarray = BCarray[idx]

    return BCarray


def _allocate_WB_BCs(bc_config: dict[str, Any]) -> dict[str, Any]:
    """Allocate WB BCs based on current flags."""
    WB_config = bc_config["wellbore"]

    for group in WB_config:
        current_flag = group["curr_BC"]

        if current_flag == "flux":
            group["function"] = group["function_flux"]
            group["values"] = group["values_flux"]
            bc_config["fluid_flux"].append(group)
        elif current_flag == "pressure":
            group["function"] = "constant"
            group["values"] = (
                [group["wellbore"]["maximum_pressure"]]
                if group["pres_constraint"] == "max"
                else [group["wellbore"]["minimum_pressure"]]
            )
            bc_config["pressure_fixed"].append(group)

    return bc_config


def parsebcs(
    bc_config: dict[str, Any],
    mesh: dict[str, Any],
    primary_variables: dict[str, Any],  # noqa
    t: float,
    controls: dict[str, Any],
) -> dict[str, Any]:
    """Parse input boundary conditions and return a dictionary containing the following.

    T_dirichlet:      n x 2 vector containing a list of fixed temperature nodes and values
                        [ nodes, temperatures]
    T_neumann:        n x 2 vector containing a list of fixed temperature flux nodes and values
                        [ nodes, flux]
    p_dirichlet:      n x 2 vector containing a list of fixed pressure nodes and values
                        [ nodes, pressures]
    p_neumann:        n x 2 vector containing a list of fixed fluid flux nodes and values
                        [ nodes, flux]
    """
    WB_on = bool("wellbore" in bc_config)
    # wellbore conditions
    if WB_on:
        # save the original lengths
        original_lengths = {}
        fields = ["pressure_fixed", "temperature_fixed", "fluid_flux", "temperature_flux"]
        for key in fields:
            original_lengths[key] = len(bc_config[key])
        # allocate WB conditions to flux or pressure
        bc_config = _allocate_WB_BCs(bc_config)

    # Parse the boundary condition types
    # Temperature - Dirichlet
    T_dirichlet = _gatherBCs(mesh, t, bc_config["temperature_fixed"], 1, controls)

    # Pressure - Dirichlet
    p_dirichlet = _gatherBCs(mesh, t, bc_config["pressure_fixed"], 1, controls)

    # Temperature - Neumann
    T_neumann = _gatherBCs(mesh, t, bc_config["temperature_flux"], 2, controls)

    # Pressure - Neumann
    p_neumann = _gatherBCs(mesh, t, bc_config["fluid_flux"], 2, controls)

    # Return the config to its original lengths
    if WB_on:
        for key in fields:
            bc_config[key] = bc_config[key][0 : original_lengths[key]]

    return {
        "T_dirichlet": T_dirichlet,
        "T_neumann": T_neumann,
        "p_dirichlet": p_dirichlet,
        "p_neumann": p_neumann,
    }
