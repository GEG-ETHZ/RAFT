"""Parse the input boundary conditions and translate them into FEM input."""

import numpy as np
import numpy.typing as npt


def _listprescribednodes(incoming: dict) -> npt.NDArray[np.float64]:
    pairs = list(zip(incoming["nodes"], incoming["values"]))
    if pairs:
        return np.array(pairs, dtype=np.float64)
    return np.empty((0, 2), dtype=np.float64)


def parsebcs(
    bc_config: dict,
    mesh: dict,  # noqa
    p: npt.NDArray[np.float64],  # noqa
    T: npt.NDArray[np.float64],  # noqa
    t: float,  # noqa
) -> dict:
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
    # pressure, temperature, and time (p, T, t) passed in for future use

    # can I remove the [0]? How to format the input yaml to take it out level out?
    p_dirichlet = _listprescribednodes(bc_config["pressure_fixed"][0])
    p_neumann = _listprescribednodes(bc_config["fluid_flux"][0])
    T_dirichlet = _listprescribednodes(bc_config["temperature_fixed"][0])
    T_neumann = _listprescribednodes(bc_config["temperature_flux"][0])

    return {
        "T_dirichlet": T_dirichlet,
        "T_neumann": T_neumann,
        "p_dirichlet": p_dirichlet,
        "p_neumann": p_neumann,
    }
