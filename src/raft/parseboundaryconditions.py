"""Parse the input boundary conditions and translate them into FEM input."""

from typing import Any

import numpy as np
import numpy.typing as npt


def _createBCpairs(incoming: dict[str, Any]) -> npt.NDArray[np.float64]:
    pairs = list(zip(incoming["BCnodes"], incoming["BCvalues"]))
    if pairs:
        return np.array(pairs, dtype=np.float64)
    return np.empty((0, 2), dtype=np.float64)


def _evaluateBCfunctions(incoming: dict[str, Any], t: float) -> dict[str, Any]:
    match incoming["function"]:
        case "constant":
            incoming["BCvalues"] = [incoming["values"][0]] * len(incoming["BCnodes"])
        case "linear":
            incoming["BCvalues"] = [t * incoming["values"][0] + incoming["values"][1]] * len(
                incoming["BCnodes"]
            )

    return incoming


def _gatherBCs(mesh, t, incoming: list[dict[str, Any]], bctype: int) -> npt.NDArray[np.float64]:  # noqa: C901
    """bctype: 1 - Dirichlet, 2 - Neumann
    Boundary condition type necessary for flux integrals, not yet implemented.
    """
    BCarray = np.empty((0, 2))
    # for each BC group
    for group in incoming:
        # gather all the corresponding nodes
        if "faces" in group:
            # load the groups of faces determined in the pre-processing
            if bctype == 2:
                raise NotImplementedError("Flux face integrals")

            BCnodes = []
            for face in group["faces"]:
                BCnodes.extend(mesh["boundaries"][face])
        elif "coords" in group:
            # find the node closest to the coordinates
            BCnodes = []
            for xi in group["coords"]:
                d = np.linalg.norm(mesh["nodes"] - xi, axis=1)
                closest_node = np.argmin(d).astype(int)
                BCnodes.append(closest_node)
        elif "nodes" in group:
            # directly append the nodes
            BCnodes = group["nodes"]
        else:
            raise NotImplementedError("Invalid BC geometry specification")

        # if the group has no specified nodes, skip it
        if not BCnodes:
            continue

        # save collected nodes to group
        group["BCnodes"] = BCnodes

        # evaluate the value of the nodes
        if "function" not in group:
            group["function"] = "constant"  # assume constant value if no function given

        # get the precribed value of the BC group
        group = _evaluateBCfunctions(group, t)

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
