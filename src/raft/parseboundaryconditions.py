"""Parse the input boundary conditions and translate them into FEM input.

Each group in `config["boundary_conditions"]` describes one geometry feature and
carries up to two field specs, "hydraulics" and "thermal".
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from .matrixassembly import integrate_boundary_flux_matrices


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


def _resolve_group_geometry(
    mesh: dict[str, Any], group: dict[str, Any], controls: dict[str, Any], need_flux_integral: bool
) -> list[int]:
    """Resolve node indices for one BC group, caching its flux-weight matrix once.

    Shared by "hydraulics" and "thermal" so both reuse the same geometry/weights.
    """
    if "nodes" not in group:
        if "faces" in group:
            nodes: list[int] = []
            conn: list[Any] = []
            for face in group["faces"]:
                nodes.extend(mesh["boundaries"][face])
                conn.extend(mesh["boundaries"][face + "_conn"])
            group["nodes"], group["_conn"], group["_conn_kind"] = nodes, conn, "face"
        elif "lines" in group:
            group["nodes"] = group["line_nodes"]
            group["_conn"], group["_conn_kind"] = group["line_conn"], "line"
        else:
            raise NotImplementedError("Invalid BC geometry specification")

    nodes = group["nodes"]
    if need_flux_integral and nodes:  # noqa: SIM102
        if "_conn" in group and "flux_weight_matrix" not in group:
            flux_mats = integrate_boundary_flux_matrices(
                mesh, group["_conn"], nodes, controls, group["_conn_kind"]
            )
            group["flux_weight_matrix"] = flux_mats["A"]

    return nodes


def _apply_field(
    mesh: dict[str, Any], t: float, group: dict[str, Any], field: str, nodes: list[int]
) -> tuple[npt.NDArray[np.float64], bool] | None:
    """Evaluate one field spec ("hydraulics" or "thermal") on a group into [node, value] pairs."""
    spec = group.get(field)
    if spec is None or not nodes:
        return None

    spec["nodes"] = nodes
    if "function" not in spec:
        spec["function"] = "constant"  # assume constant value if no function given
    spec = evaluateBCfunctions(spec, t, mesh["nodes"])
    values = np.asarray(spec["BCvalues"], dtype=np.float64)

    if spec["type"] == "flux" and "flux_weight_matrix" in group:
        values = group["flux_weight_matrix"] @ values

    is_dirichlet = spec["type"] in ("pressure", "temperature")
    pairs = np.column_stack((np.asarray(nodes, dtype=np.float64), values))
    return pairs, is_dirichlet


def _stack_unique(rows: list[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    """Stack collected [node, value] row blocks, keeping one entry per node."""
    if not rows:
        return np.empty((0, 2))
    arr = np.vstack(rows)
    _, idx = np.unique(arr[:, 0], return_index=True)
    return arr[idx]


def parsebcs(
    bc_config: list[dict[str, Any]],
    mesh: dict[str, Any],
    primary_variables: dict[str, Any],  # noqa
    t: float,
    controls: dict[str, Any],
) -> dict[str, Any]:
    """Parse input boundary condition groups and return a dictionary containing the following.

    T_dirichlet:      n x 2 vector containing a list of fixed temperature nodes and values
                        [ nodes, temperatures]
    T_neumann:        n x 2 vector containing a list of fixed temperature flux nodes and values
                        [ nodes, flux]
    p_dirichlet:      n x 2 vector containing a list of fixed pressure nodes and values
                        [ nodes, pressures]
    p_neumann:        n x 2 vector containing a list of fixed fluid flux nodes and values
                        [ nodes, flux]
    """
    p_dirichlet_rows: list[npt.NDArray[np.float64]] = []
    p_neumann_rows: list[npt.NDArray[np.float64]] = []
    T_dirichlet_rows: list[npt.NDArray[np.float64]] = []
    T_neumann_rows: list[npt.NDArray[np.float64]] = []

    for group in bc_config:
        need_flux_integral = False
        for field in ("hydraulics", "thermal"):
            if field in group and group[field]["type"] == "flux":
                need_flux_integral = True

        nodes = _resolve_group_geometry(mesh, group, controls, need_flux_integral)
        if not nodes:
            continue

        hydraulics = _apply_field(mesh, t, group, "hydraulics", nodes)
        if hydraulics is not None:
            pairs, is_dirichlet = hydraulics
            (p_dirichlet_rows if is_dirichlet else p_neumann_rows).append(pairs)

        thermal = _apply_field(mesh, t, group, "thermal", nodes)
        if thermal is not None:
            pairs, is_dirichlet = thermal
            (T_dirichlet_rows if is_dirichlet else T_neumann_rows).append(pairs)

    return {
        "T_dirichlet": _stack_unique(T_dirichlet_rows),
        "T_neumann": _stack_unique(T_neumann_rows),
        "p_dirichlet": _stack_unique(p_dirichlet_rows),
        "p_neumann": _stack_unique(p_neumann_rows),
    }
