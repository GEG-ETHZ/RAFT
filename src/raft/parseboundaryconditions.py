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


def _evaluateBCfunctions(
    incoming: dict[str, Any], t: float, nodes: npt.NDArray[np.float64]
) -> dict[str, Any]:
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
            y_ramp = incoming["values"][0]
            t_ramp = incoming["values"][1]
            y = y_ramp if t >= t_ramp else t / t_ramp * y_ramp
            incoming["BCvalues"] = [y] * len(incoming["nodes"])
        case "table":
            tn = incoming["values"][0]
            yn = incoming["values"][1]
            y = np.interp(t, tn, yn) if t >= tn[0] else 0
            incoming["BCvalues"] = [y] * len(incoming["nodes"])
    return incoming


def _gatherBCs(  # noqa: C901
    mesh: dict[str, Any], t: float, incoming: list[dict[str, Any]], bctype: int
) -> npt.NDArray[np.float64]:
    """bctype: 1 - Dirichlet, 2 - Neumann.

    Boundary condition type necessary for flux integrals, not yet implemented.
    """
    BCarray = np.empty((0, 2))
    face_flag = False
    face_conn = []
    face_nn = 0
    # for each BC group
    for group in incoming:
        # gather all the corresponding nodes
        if "nodes" in group:
            # directly append the nodes
            # covers "nodes" and "coords" cases, which have been mapped in pre-processing
            BCnodes = group["nodes"]

            # on second passes, check that if nodes is an amalgamation of faces
            if "faces" in group:
                face_flag = True
        elif "faces" in group:
            # load the groups of faces determined in the pre-processing
            BCnodes = []
            for face in group["faces"]:
                BCnodes.extend(mesh["boundaries"][face])
                if bctype == 2:
                    face_flag = True
                    face_conn.extend(mesh["boundaries"][face + "_conn"])
                    face_nn += len(mesh["boundaries"][face])
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
        group = _evaluateBCfunctions(group, t, mesh["nodes"])
        if bctype == 2 and face_flag:
            if "flux_weight_matrix" not in group:
                flux_mats = integrate_boundary_flux_matrices(mesh, face_conn, group["nodes"])
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


def parsebcs(
    bc_config: dict[str, Any],
    mesh: dict[str, Any],  # noqa
    primary_variables: dict[str, Any],  # noqa
    t: float,  # noqa
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
    # for initial pass, directly specify the nodes. In the future, this
    # function can be modified to find the corresponding nodes based
    # on specifying the coordinates or to perform edge integration.
    #
    # further sophistication is needed to parse the type of boundary condtion too
    #
    # pressure, temperature, and time (p, T, t) passed in for future use

    # Temperature - Dirichlet
    T_dirichlet = _gatherBCs(mesh, t, bc_config["temperature_fixed"], 1)

    # Pressure - Dirichlet
    p_dirichlet = _gatherBCs(mesh, t, bc_config["pressure_fixed"], 1)

    # Temperature - Neumann
    T_neumann = _gatherBCs(mesh, t, bc_config["temperature_flux"], 2)

    # Pressure - Neumann
    p_neumann = _gatherBCs(mesh, t, bc_config["fluid_flux"], 2)

    return {
        "T_dirichlet": T_dirichlet,
        "T_neumann": T_neumann,
        "p_dirichlet": p_dirichlet,
        "p_neumann": p_neumann,
    }
