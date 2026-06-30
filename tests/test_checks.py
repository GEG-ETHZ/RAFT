"""Check FEM test cases."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from raft.preprocessing import load_config


def check_patch_test(
    config_name: str,
    config_dir: Path,
    results: dict[str, Any],
    mesh: dict[str, Any],
    dimension: int,
    test_name: str,
) -> bool:
    """Check patch test. Result should be an exact linear solution."""
    # Set tolerance
    tol = 1.0e-9

    # Load config
    config = load_config(config_name, config_dir, verbose=False)

    # Extract nodes along lines
    nodes = mesh["nodes"]
    dims = [0, 1, 2]

    if dimension in dims:  # Cases aligned with cartesian axis
        dims.remove(dimension)
        line_1 = np.nonzero(
            np.logical_and(
                nodes[:, dims[0]] == min(nodes[:, dims[0]]), nodes[:, dims[1]] == min(nodes[:, dims[1]])
            )
        )[0]
        line_2 = np.nonzero(
            np.logical_and(
                nodes[:, dims[0]] == max(nodes[:, dims[0]]), nodes[:, dims[1]] == min(nodes[:, dims[1]])
            )
        )[0]
        line_3 = np.nonzero(
            np.logical_and(
                nodes[:, dims[0]] == min(nodes[:, dims[0]]), nodes[:, dims[1]] == max(nodes[:, dims[1]])
            )
        )[0]
        line_4 = np.nonzero(
            np.logical_and(
                nodes[:, dims[0]] == max(nodes[:, dims[0]]), nodes[:, dims[1]] == max(nodes[:, dims[1]])
            )
        )[0]
    else:
        # Cases not aligned with cartesian axis. Only works if only 1 element
        # specified across the other axes.
        n = 8
        indx = list(range(n, n + n + 1))
        line_1 = np.array([0] + indx + [4])
        line_2 = np.array([3] + [i + n + 1 for i in indx] + [7])
        line_3 = np.array([1] + [i + 2 * (n + 1) for i in indx] + [5])
        line_4 = np.array([2] + [i + 3 * (n + 1) for i in indx] + [6])

    L = np.sqrt(np.sum((nodes[4, :] - nodes[0, :]) ** 2))

    d_1 = np.sqrt(np.sum((nodes[line_1, :] - nodes[line_1[0], :]) ** 2, 1))
    d_2 = np.sqrt(np.sum((nodes[line_2, :] - nodes[line_2[0], :]) ** 2, 1))
    d_3 = np.sqrt(np.sum((nodes[line_3, :] - nodes[line_3[0], :]) ** 2, 1))
    d_4 = np.sqrt(np.sum((nodes[line_4, :] - nodes[line_4[0], :]) ** 2, 1))

    # check temperature

    T0 = config["boundary_conditions"]["temperature_fixed"][0]["values"][0]
    T1 = config["boundary_conditions"]["temperature_fixed"][1]["values"][0]

    T = results["T"][:, 0]

    plt.figure()
    plt.plot([0, L], [T0, T1])
    plt.plot(d_1, T[line_1], "ko")
    plt.plot(d_2, T[line_2], "bo")
    plt.plot(d_3, T[line_3], "k^")
    plt.plot(d_4, T[line_4], "b^")
    plt.xlabel("d (m)")
    plt.ylabel("T")
    plt.title(test_name)

    # define exact solution - linear
    def _lin_exact(y0: float, y1: float, x: float, X: float) -> float:
        y = y0 + (y1 - y0) * x / X
        return y

    # calculate error
    err_ex = sum((_lin_exact(T0, T1, d_1, L) - T[line_1]) ** 2)
    err_ex += sum((_lin_exact(T0, T1, d_2, L) - T[line_2]) ** 2)
    err_ex += sum((_lin_exact(T0, T1, d_3, L) - T[line_3]) ** 2)
    err_ex += sum((_lin_exact(T0, T1, d_4, L) - T[line_4]) ** 2)

    # check pressure
    p0 = config["boundary_conditions"]["pressure_fixed"][0]["values"][0]
    p1 = config["boundary_conditions"]["pressure_fixed"][1]["values"][0]

    p = results["p"][:, 0]

    plt.figure()
    plt.plot([0, L], [p0, p1])
    plt.plot(d_1, p[line_1], "ko")
    plt.plot(d_2, p[line_2], "bo")
    plt.plot(d_3, p[line_3], "k^")
    plt.plot(d_4, p[line_4], "b^")
    plt.xlabel("d (m)")
    plt.ylabel("p")
    plt.title(test_name)

    err_ex += sum((_lin_exact(p0, p1, d_1, L) - p[line_1]) ** 2)
    err_ex += sum((_lin_exact(p0, p1, d_2, L) - p[line_2]) ** 2)
    err_ex += sum((_lin_exact(p0, p1, d_3, L) - p[line_3]) ** 2)
    err_ex += sum((_lin_exact(p0, p1, d_4, L) - p[line_4]) ** 2)

    test_pass = bool(err_ex < tol)

    return test_pass


def check_advection(
    config_name: str, config_dir: Path, results: dict[str, Any], mesh: dict[str, Any], test_name: str
) -> bool:
    """Check advection against analytical solution."""
    # Set tolerance
    tol = 1.0e-4

    # Load config
    config = load_config(config_name, config_dir, verbose=False)

    # Extract nodes along lines
    nodes = mesh["nodes"]
    line_1 = np.where((nodes[:, 0] == 0) & (nodes[:, 1] == 0))[0]

    L = np.sqrt(np.sum((nodes[4, :] - nodes[0, :]) ** 2))

    d_1 = np.sqrt(np.sum((nodes[line_1, :] - nodes[0, :]) ** 2, 1))

    # Define analytical solution
    mat = config["material"]
    # flux
    p0 = config["boundary_conditions"]["pressure_fixed"][1]["values"][0]
    p1 = config["boundary_conditions"]["pressure_fixed"][0]["values"][0]
    kf = mat["solid"]["permeability"]
    mu = mat["fluid"]["viscosity"]
    q = kf / mu * (p1 - p0) / L

    # conductivity
    phi = mat["solid"]["porosity"]
    lambda_e = phi * mat["fluid"]["therm_cond"] + (1 - phi) * mat["solid"]["therm_cond"]
    cp = mat["fluid"]["specific_heat"]
    rho_f = mat["fluid"]["density"]

    # peclet number per length
    Pe = L * (q * cp * rho_f) / lambda_e

    # analytical solution
    x = np.linspace(0, L, 100) / L
    T_exact = (np.exp(Pe * x) - 1) / (np.exp(Pe) - 1)

    T0 = config["boundary_conditions"]["temperature_fixed"][0]["values"][0]
    T1 = config["boundary_conditions"]["temperature_fixed"][1]["values"][0]

    # extract T and normalize
    T = (results["T"][:, 0] - T0) / (T1 - T0)

    plt.figure()
    plt.plot(x, T_exact, "r-")
    plt.plot(d_1 / L, T[line_1], "ko")
    plt.ylabel("T")
    plt.title(test_name)
    plt.text(0.05, 0.9, f"Pe = {Pe * L:.2f}")

    # calculate error
    T_exact = (np.exp(Pe * d_1 / L) - 1) / (np.exp(Pe) - 1)
    err_ex = sum((T_exact - T[line_1]) ** 2)
    # The error here is calculated pased on the nodal solutions, which with SUPG
    # stabilization should be close to exact at the nodes. The full error
    # includes the linear interpolation between the nodes, which is much higher.

    test_pass = bool(err_ex < tol)

    return test_pass


def check_buoyancy(
    config_name: str, config_dir: Path, results: dict[str, Any], mesh: dict[str, Any], test_name: str
) -> bool:
    """Check buoyancy against analytical solution."""
    # Set tolerance
    tol = 1.0e-6

    # Load config
    config = load_config(config_name, config_dir, verbose=False)

    # Extract nodes along lines
    nodes = mesh["nodes"]
    line_1 = np.where((nodes[:, 0] == 0) & (nodes[:, 1] == 0))[0]

    L = np.sqrt(np.sum((nodes[4, :] - nodes[0, :]) ** 2))

    d_1 = np.sqrt(np.sum((nodes[line_1, :] - nodes[0, :]) ** 2, 1))

    # Define analytical solution
    mat = config["material"]

    g = -9.81
    rho_f = mat["fluid"]["density"]
    alpha_f = mat["fluid"]["therm_expansion"]
    T0 = config["boundary_conditions"]["temperature_fixed"][0]["values"][0]
    T1 = config["boundary_conditions"]["temperature_fixed"][1]["values"][0]
    dT = T0 - T1

    beta = rho_f * alpha_f * g * dT * L / 2

    # analytical solution
    x = np.linspace(0, L, 100) / L
    p_exact = beta * (x**2 - x)

    # extract T and normalize
    p = results["p"][:, 0]

    plt.figure()
    plt.plot(x, p_exact, "r-")
    plt.plot(d_1 / L, p[line_1], "ko")
    plt.ylabel("p")
    plt.title(test_name)

    # calculate error
    x = d_1 / L
    p_exact = beta * (x**2 - x)
    err_ex = sum((p_exact - p[line_1]) ** 2)
    # The error here is calculated pased on the nodal solutions.

    test_pass = bool(err_ex < tol)

    return test_pass


def check_pressurization(
    config_name: str, config_dir: Path, results: dict[str, Any], mesh: dict[str, Any], test_name: str
) -> bool:
    """Check pressurization against analytical solution."""
    # Set tolerance
    tol = 1.0e-6

    # Load config
    config = load_config(config_name, config_dir, verbose=False)

    # Extract nodes along lines
    nodes = mesh["nodes"]
    line_1 = np.where((nodes[:, 0] == 0) & (nodes[:, 1] == 0))[0]

    L = np.sqrt(np.sum((nodes[4, :] - nodes[0, :]) ** 2))

    d_1 = np.sqrt(np.sum((nodes[line_1, :] - nodes[0, :]) ** 2, 1))

    # Define analytical solution
    mat = config["material"]

    # ----- material properties
    kf = mat["solid"]["permeability"]
    mu = mat["fluid"]["viscosity"]
    alpha_s = mat["solid"]["therm_expansion"]
    alpha_f = mat["fluid"]["therm_expansion"]
    phi = mat["solid"]["porosity"]
    biot = mat["solid"]["biot_coeff"]

    # effective expansion modulus
    alpha_e = biot * alpha_s + phi * (alpha_f - alpha_s)

    dT = -config["boundary_conditions"]["temperature_fixed"][1]["values"][0]

    beta = alpha_e * dT * mu * L**2 / 6 / kf

    # analytical solution
    x = np.linspace(0, 1, 100)
    p_exact = beta * (x**3 - x)

    # extract T and normalize
    p = results["p"][:, 0]

    plt.figure()
    plt.plot(1 - x, p_exact, "r-")
    plt.plot(d_1 / L, p[line_1], "ko")
    plt.ylabel("p")
    plt.title(test_name)

    # calculate error
    x = 1 - (d_1 / L)
    p_exact = beta * (x**3 - x)
    err_ex = sum((p_exact - p[line_1]) ** 2)
    # The error here is calculated pased on the nodal solutions.

    test_pass = bool(err_ex < tol)

    return test_pass


def check_regions(config_dir: Path, mesh: dict[str, Any]) -> bool:
    """Check regions against known list."""
    # saved file with correct regions allocated to mesh.
    region_path = config_dir / "regions_test_check.txt"

    with open(region_path) as file:
        element_regions = [int(line.strip()) for line in file]

    test_regions = np.asarray(mesh["regions"])

    test_pass = bool(np.all(element_regions == test_regions))

    return test_pass


def check_heterogeneous(
    config_name: str,
    config_dir: Path,
    results: dict[str, Any],
    mesh: dict[str, Any],
    test_name: str,
) -> bool:
    """Check patch test. Result should be an exact linear solution."""
    # Set tolerance
    tol = 1.0e-4

    # Load config
    config = load_config(config_name, config_dir, verbose=False)

    # Extract nodes along lines
    nodes = mesh["nodes"]

    n = 13
    indx = list(range(8, 8 + n + 1))
    line_1 = np.array([0] + indx + [4])

    d_1 = np.sqrt(np.sum((nodes[line_1, :] - nodes[line_1[0], :]) ** 2, 1))

    # Define regions
    H = 21
    regions = np.zeros(len(line_1)).astype(int)
    regions[d_1 < H] = 1

    # check temperature

    T0 = config["boundary_conditions"]["temperature_fixed"][0]["values"][0]
    q = config["boundary_conditions"]["temperature_flux"][0]["values"][0]
    A = max(mesh["nodes"][:, 0]) * max(mesh["nodes"][:, 1])
    q = -q / A * 4

    k = config["material"]["solid"]["therm_cond"]

    T_ex = T0 * np.ones(len(d_1))
    for i in range(len(d_1) - 1):
        T_ex[i + 1] = T_ex[i] - q / k[regions[i + 1]] * (d_1[i + 1] - d_1[i])

    T = results["T"][:, 0]

    plt.figure()
    plt.plot(d_1, T_ex, "r-")
    plt.plot(d_1, T[line_1], "ko")
    plt.xlabel("d (m)")
    plt.ylabel("T")
    plt.title(test_name)

    # calculate error
    err_ex = sum((T_ex - T[line_1]) ** 2)

    # check pressure
    p0 = config["boundary_conditions"]["pressure_fixed"][0]["values"][0]
    q = config["boundary_conditions"]["fluid_flux"][0]["values"][0]
    q = -q / A * 4

    k = config["material"]["solid"]["permeability"]
    mu = config["material"]["fluid"]["viscosity"]

    p_ex = p0 * np.ones(len(d_1))
    for i in range(len(d_1) - 1):
        p_ex[i + 1] = p_ex[i] - mu * q / k[regions[i + 1]] * (d_1[i + 1] - d_1[i])

    p = results["p"][:, 0]

    plt.figure()
    plt.plot(d_1, p_ex, "b-")
    plt.plot(d_1, p[line_1], "ko")
    plt.xlabel("d (m)")
    plt.ylabel("p")
    plt.title(test_name)

    err_ex += sum((p_ex - p[line_1]) ** 2)

    test_pass = bool(err_ex < tol)

    return test_pass
