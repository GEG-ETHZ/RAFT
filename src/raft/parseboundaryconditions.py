"""Parse the input boundary conditions and translate them into FEM input."""

from typing import Any

import numpy as np
import numpy.typing as npt


def _listprescribednodes(incoming: dict[str, Any]) -> npt.NDArray[np.float64]:
    pairs = list(zip(incoming["nodes"], incoming["values"]))
    if pairs:
        return np.array(pairs, dtype=np.float64)
    return np.empty((0, 2), dtype=np.float64)


def _listtimedependentnodes(incoming: dict[str, Any], t: float) -> npt.NDArray[np.float64]:

    match incoming["function"]:
        case "linear":
            incoming["values"] = [t * incoming["param_1"] + incoming["param_2"]] * len(incoming["nodes"])

    return _listprescribednodes(incoming)


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

    # Index 0 - constant values in time
    p_dirichlet = _listprescribednodes(bc_config["pressure_fixed"][0])
    p_neumann = _listprescribednodes(bc_config["fluid_flux"][0])
    T_dirichlet = _listprescribednodes(bc_config["temperature_fixed"][0])
    T_neumann = _listprescribednodes(bc_config["temperature_flux"][0])

    # Indx 1 - time dependent boundary conditions
    if len(bc_config["pressure_fixed"]) > 1:
        p_dirichlet_t = _listtimedependentnodes(bc_config["pressure_fixed"][1], t)
        p_dirichlet = np.vstack((p_dirichlet, p_dirichlet_t))
    if len(bc_config["fluid_flux"]) > 1:
        p_neumann_t = _listtimedependentnodes(bc_config["fluid_flux"][1], t)
        p_neumann = np.vstack((p_neumann, p_neumann_t))
    if len(bc_config["temperature_fixed"]) > 1:
        T_dirichlet_t = _listtimedependentnodes(bc_config["temperature_fixed"][1], t)
        T_dirichlet = np.vstack((T_dirichlet, T_dirichlet_t))
    if len(bc_config["temperature_flux"]) > 1:
        T_neumann_t = _listtimedependentnodes(bc_config["temperature_flux"][1], t)
        T_neumann = np.vstack((T_neumann, T_neumann_t))

    return {
        "T_dirichlet": T_dirichlet,
        "T_neumann": T_neumann,
        "p_dirichlet": p_dirichlet,
        "p_neumann": p_neumann,
    }
