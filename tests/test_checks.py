"""Check FEM test cases."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from raft.preprocessing import load_config


def check_patch_test(
    config_name: str,
    config_dir: Path,
    results: dict[str, Any],
    mesh: dict[str, Any],
    dimension: int,
    test_name: str,
    plot_on: bool,
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

    if plot_on:
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

    if plot_on:
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


def check_patch_test_flux(
    config_name: str,
    config_dir: Path,
    results: dict[str, Any],
    mesh: dict[str, Any],
    dimension: int,
    test_name: str,
    plot_on: bool,
) -> bool:
    """Check patch test. Result should be an exact linear solution."""
    # Set tolerance
    tol = 1.0e-9

    # Load config
    config = load_config(config_name, config_dir, verbose=False)

    # Extract nodes along lines
    nodes = mesh["nodes"]
    nsd = mesh["nsd"]
    dims = [0, 1, 2] if nsd == 3 else [0, 1]

    dims.remove(dimension)
    if nsd == 3:
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
        line_1 = line_1[np.argsort(nodes[line_1, dimension])]
        line_2 = line_2[np.argsort(nodes[line_2, dimension])]
    elif nsd == 2:
        line_1 = np.nonzero(nodes[:, dims[0]] == min(nodes[:, dims[0]]))[0]
        line_2 = np.nonzero(nodes[:, dims[0]] == max(nodes[:, dims[0]]))[0]
        line_1 = line_1[np.argsort(nodes[line_1, dimension])]
        line_2 = line_2[np.argsort(nodes[line_2, dimension])]

    L = np.sqrt(np.sum((nodes[line_1[-1], :] - nodes[line_1[0], :]) ** 2))

    d_1 = np.sqrt(np.sum((nodes[line_1, :] - nodes[line_1[0], :]) ** 2, 1))
    d_2 = np.sqrt(np.sum((nodes[line_2, :] - nodes[line_2[0], :]) ** 2, 1))

    # check temperature

    T0 = config["boundary_conditions"]["temperature_fixed"][0]["values"][0]
    q = config["boundary_conditions"]["temperature_flux"][0]["values"][0]
    k = config["material"]["solid"]["therm_cond"]

    # define exact solution - linear
    def _lin_exact(y0: float, m: float, x: float) -> float:
        y = y0 + m * x
        return y

    m = q / k

    T1 = _lin_exact(T0, m, L)

    T = results["T"][:, 0]

    if plot_on:
        plt.figure()
        plt.plot([0, L], [T0, T1])
        plt.plot(d_1, T[line_1], "ko")
        plt.plot(d_2, T[line_2], "bo")
        plt.xlabel("d (m)")
        plt.ylabel("T")
        plt.title(test_name)

    # calculate error
    mean_val = 0.5 * (T0 + T1)
    err_ex = sum((_lin_exact(T0, m, d_1) - T[line_1]) ** 2) / mean_val
    err_ex += sum((_lin_exact(T0, m, d_2) - T[line_2]) ** 2) / mean_val

    # check pressure
    p1 = config["boundary_conditions"]["pressure_fixed"][0]["values"][0]
    q = config["boundary_conditions"]["fluid_flux"][0]["values"][0]
    k = config["material"]["solid"]["permeability"]
    mu = config["material"]["fluid"]["viscosity"]

    # define exact solution - linear
    # pressure is defined in the opposite direction
    def _lin_exact_p(y0: float, m: float, x: float, L: float) -> float:
        y = y0 + m * (L - x)
        return y

    m = mu * q / k
    p0 = _lin_exact_p(p1, m, 0, L)

    p = results["p"][:, 0]

    if plot_on:
        plt.figure()
        plt.plot([0, L], [p0, p1])
        plt.plot(d_1, p[line_1], "ko")
        plt.plot(d_2, p[line_2], "bo")
        plt.xlabel("d (m)")
        plt.ylabel("p")
        plt.title(test_name)

    mean_val = 0.5 * (p0 + p1)
    err_ex = sum((_lin_exact_p(p1, m, d_1, L) - p[line_1]) ** 2) / mean_val
    err_ex += sum((_lin_exact_p(p1, m, d_2, L) - p[line_2]) ** 2) / mean_val

    test_pass = bool(err_ex < tol)

    return test_pass


def check_advection(
    config_name: str,
    config_dir: Path,
    results: dict[str, Any],
    mesh: dict[str, Any],
    test_name: str,
    plot_on: bool,
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

    if plot_on:
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
    config_name: str,
    config_dir: Path,
    results: dict[str, Any],
    mesh: dict[str, Any],
    test_name: str,
    plot_on: bool,
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

    if plot_on:
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
    config_name: str,
    config_dir: Path,
    results: dict[str, Any],
    mesh: dict[str, Any],
    test_name: str,
    plot_on: bool,
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

    if plot_on:
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
    plot_on: bool,
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

    if plot_on:
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

    if plot_on:
        plt.figure()
        plt.plot(d_1, p_ex, "b-")
        plt.plot(d_1, p[line_1], "ko")
        plt.xlabel("d (m)")
        plt.ylabel("p")
        plt.title(test_name)

    err_ex += sum((p_ex - p[line_1]) ** 2)

    test_pass = bool(err_ex < tol)

    return test_pass


def check_NL_viscosity(
    config_name: str,
    config_dir: Path,
    results: dict[str, Any],
    mesh: dict[str, Any],
    test_name: str,
    plot_on: bool,
) -> bool:
    """Check pressure against semi-analytical solution.

    By setting flux on a quasi-1D problem, the solution mandates constant flux
    throughout the domain. This decouples pressure and temperature, allowing us to
    solve for the analytical temperature distribution. Then, we use the analytical
    temperature distribution to find the viscosity distribution and numerically
    integrate to find the semi-analytical pressure distribution.
    """
    from raft.matprops import get_viscosity

    # Set tolerance
    tol = 1.0e-3

    # Load config
    config = load_config(config_name, config_dir, verbose=False)

    # Extract nodes along lines
    nodes = mesh["nodes"]
    line_1 = np.where((nodes[:, 0] == 0) & (nodes[:, 1] == 0))[0]

    L = np.max(nodes[:, 2])

    d_1 = np.sqrt(np.sum((nodes[line_1, :] - nodes[0, :]) ** 2, 1))

    # Define analytical solution
    mat = config["material"]
    # flux
    p0 = config["boundary_conditions"]["pressure_fixed"][0]["values"][0]
    q = config["boundary_conditions"]["fluid_flux"][0]["values"][0]

    # temperature
    T0 = config["boundary_conditions"]["temperature_fixed"][0]["values"][0]
    T1 = config["boundary_conditions"]["temperature_fixed"][1]["values"][0]

    # conductivity
    phi = mat["solid"]["porosity"]
    lambda_e = phi * mat["fluid"]["therm_cond"] + (1 - phi) * mat["solid"]["therm_cond"]
    cp = mat["fluid"]["specific_heat"]
    rho_f = mat["fluid"]["density"]

    # peclet number per length
    Pe = L * abs(q * cp * rho_f) / lambda_e

    # analytical solution
    x = np.linspace(0, L, 100) / L
    T_exact = (np.exp(Pe * x) - 1) / (np.exp(Pe) - 1)
    T_exact = T0 + (T1 - T0) * T_exact

    # Viscosity
    visc_coeff = mat["fluid"]["viscosity"]
    visc_exact = visc_coeff[0] * np.exp(
        visc_coeff[1] / T_exact + visc_coeff[2] * T_exact + visc_coeff[3] * T_exact**2
    )

    # pressure
    k = mat["solid"]["permeability"]
    p_exact = np.empty(100)
    p_exact[0] = p0
    dx = x[1] * L
    for i in range(1, 100):
        p_exact[i] = p_exact[i - 1] + dx * q / k * 0.5 * (visc_exact[i - 1] + visc_exact[i])

    # extract results
    p = results["p"][:, 0]
    T = results["T"][:, 0]
    visc_approx, _ = get_viscosity(T[line_1], visc_coeff)

    if plot_on:
        plt.figure()
        plt.plot(x, T_exact, "r-")
        plt.plot(d_1 / L, T[line_1], "ko")
        plt.ylabel("T")
        plt.title(test_name)

        plt.figure()
        plt.plot(x, visc_exact, "r--")
        plt.plot(d_1 / L, visc_approx, "ko")
        plt.ylabel("viscosity")
        plt.title(test_name)

        plt.figure()
        plt.plot(x, p_exact, "b-")
        plt.plot(d_1 / L, p[line_1], "ko")
        plt.ylabel("p")
        plt.title(test_name)

    # calculate error
    p_exact_atnodes = np.interp(d_1, x * L, p_exact)
    numerator = sum((p_exact_atnodes - p[line_1]) ** 2)
    denominator = sum((p_exact_atnodes) ** 2)
    # The error here is calculated based on the nodal solutions.
    err_ex = np.sqrt(numerator / denominator)

    test_pass = bool(err_ex < tol)

    return test_pass


def check_axisymmetry(
    config_name: str,
    config_dir: Path,
    results: dict[str, Any],
    mesh: dict[str, Any],
    test_name: str,
    plot_on: bool,
) -> bool:
    """Flux on the inside of an annulus. Linear approximation of logarithm
    will not be exact, use high tolerance."""
    # Set tolerance
    tol = 1.0e-2

    # Load config
    config = load_config(config_name, config_dir, verbose=False)

    # Extract nodes along lines
    nodes = mesh["nodes"]

    line_1 = np.nonzero(nodes[:, 1] == min(nodes[:, 1]))[0]
    line_2 = np.nonzero(nodes[:, 1] == max(nodes[:, 1]))[0]
    line_1 = line_1[np.argsort(nodes[line_1, 0])]
    line_2 = line_2[np.argsort(nodes[line_2, 0])]

    Ri = min(nodes[:, 0])
    Ro = max(nodes[:, 0])

    d_1 = np.sqrt(np.sum((nodes[line_1, :] - nodes[line_1[0], :]) ** 2, 1)) + Ri
    d_2 = np.sqrt(np.sum((nodes[line_2, :] - nodes[line_2[0], :]) ** 2, 1)) + Ri

    # check temperature

    T0 = config["boundary_conditions"]["temperature_fixed"][0]["values"][0]
    q = config["boundary_conditions"]["temperature_flux"][0]["values"][0]
    k = config["material"]["solid"]["therm_cond"]

    # define exact solution
    def _y_exact(
        r: npt.NDArray[np.float64], y0: float, q: float, k: float, Ri: float, Ro: float
    ) -> npt.NDArray[np.float64]:
        y = y0 - q / k * Ri * np.log(r / Ro)
        return y

    r = np.linspace(Ri, Ro, 50)
    T_exact = _y_exact(r, T0, q, k, Ri, Ro)

    T = results["T"][:, 0]

    if plot_on:
        plt.figure()
        plt.plot(r, T_exact)
        plt.plot(d_1, T[line_1], "ko")
        plt.plot(d_2, T[line_2], "bo")
        plt.xlabel("r (m)")
        plt.ylabel("T")
        plt.title(test_name)

    # calculate error
    mean_val = np.mean(T_exact)
    err_ex = sum((_y_exact(d_1, T0, q, k, Ri, Ro) - T[line_1]) ** 2) / mean_val**2
    err_ex += sum((_y_exact(d_2, T0, q, k, Ri, Ro) - T[line_2]) ** 2) / mean_val**2

    # check pressure
    p0 = config["boundary_conditions"]["pressure_fixed"][0]["values"][0]
    q = config["boundary_conditions"]["fluid_flux"][0]["values"][0]
    k = config["material"]["solid"]["permeability"] / config["material"]["fluid"]["viscosity"]

    # define exact solution
    p_exact = _y_exact(r, p0, q, k, Ri, Ro)

    p = results["p"][:, 0]

    if plot_on:
        plt.figure()
        plt.plot(r, p_exact)
        plt.plot(d_1, p[line_1], "ko")
        plt.plot(d_2, p[line_2], "bo")
        plt.xlabel("d (m)")
        plt.ylabel("p")
        plt.title(test_name)

    denom = p_exact[0]
    err_ex += sum((_y_exact(d_1, p0, q, k, Ri, Ro) - p[line_1]) ** 2) / denom**2
    err_ex += sum((_y_exact(d_2, p0, q, k, Ri, Ro) - p[line_2]) ** 2) / denom**2

    test_pass = bool(err_ex < tol)

    return test_pass


def check_fracture_single(config_name: str, config_dir: Path, mesh: dict[str, Any]) -> bool:
    """Check mesh-cutting correctness for a single, fully-interior 2D fracture.

    Verifies that the fracture mesh produced by `_create_fracture_mesh` is geometrically
    and topologically consistent with the fracture specification: fracture nodes lie on
    the prescribed line and within its tip-to-tip extent, every fracture element has a
    matrix element identified on each side, and the matrix mesh partition (element areas)
    is conserved by the cut (no gaps or overlaps introduced).
    """
    config = load_config(config_name, config_dir, verbose=False)
    fracture_spec = config["mesh"]["fractures"][0]

    center = np.asarray(fracture_spec["values"][0][:2])
    half_length = fracture_spec["values"][1][0]
    theta = np.radians(fracture_spec["values"][2][0])
    u = np.array([np.cos(theta), np.sin(theta)])

    # Node-snapping tolerance mirrors the one used internally in meshprocessing.
    d_max = np.amax(mesh["nodes"], 0)
    tol = (d_max.prod()) ** 0.5 * 0.005

    frac_nodes = mesh["fracture_mesh"]["nodes"]
    d = frac_nodes - center
    s = d @ u  # tangential coordinate along the fracture
    perp = d - np.outer(s, u)  # component perpendicular to the fracture line
    perp_dist = np.linalg.norm(perp, axis=1)

    on_line = bool(np.all(perp_dist < tol))
    within_extent = bool(np.all(s <= half_length + tol) and np.all(s >= -half_length - tol))
    reaches_tips = bool(
        np.isclose(s.max(), half_length, atol=tol) and np.isclose(s.min(), -half_length, atol=tol)
    )

    # Every fracture element should find a matrix element on both sides (fully interior fracture).
    matrix_conn = mesh["fracture_mesh"]["matrix_conn"]
    ne = mesh["ne"]
    both_sides_found = all(len(pair) == 2 for pair in matrix_conn)
    valid_matrix_indices = all(0 <= idx < ne for pair in matrix_conn for idx in pair)
    correct_conn_count = len(matrix_conn) == mesh["fracture_mesh"]["ne"]

    # Cutting the mesh should not change the total matrix area (no gaps/overlaps).
    def _poly_area(coords: npt.NDArray[np.float64]) -> float:
        x, y = coords[:, 0], coords[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    nodes = mesh["nodes"]
    domain_area = (mesh["nodes"][:, 0].max() - mesh["nodes"][:, 0].min()) * (
        mesh["nodes"][:, 1].max() - mesh["nodes"][:, 1].min()
    )
    cut_area = sum(_poly_area(nodes[np.asarray(elem)]) for elem in mesh["conn"])
    area_conserved = bool(np.isclose(cut_area, domain_area, rtol=1e-8))

    test_pass = (
        on_line
        and within_extent
        and reaches_tips
        and both_sides_found
        and valid_matrix_indices
        and correct_conn_count
        and area_conserved
    )

    return test_pass


def check_fracture_perpendicular_patch(
    config_name: str,
    config_dir: Path,
    results: dict[str, Any],
    mesh: dict[str, Any],
    test_name: str,
    plot_on: bool,
) -> bool:
    """Check patch test for a fracture oriented perpendicular to the pressure/temperature gradient.

    When the fracture sits exactly on an equipotential (its tangent is perpendicular to the
    imposed gradient), there is no along-fracture flow: leak-off to the matrix is pointwise
    zero (fracture pressure/temperature equal the local matrix value everywhere), and the
    natural zero-flux condition at the fracture tips is trivially satisfied. The exact
    solution is therefore the same linear field as the fracture-free patch test, evaluated
    at every node including the fracture's own DOFs. This does not hold for any other
    fracture orientation, where longitudinal flow along the fracture perturbs the field
    away from linear.
    """
    # Set tolerance (relative to the imposed BC range)
    tol = 1.0e-3

    # Load config
    config = load_config(config_name, config_dir, verbose=False)

    p0 = config["boundary_conditions"]["pressure_fixed"][0]["values"][0]
    p1 = config["boundary_conditions"]["pressure_fixed"][1]["values"][0]
    T0 = config["boundary_conditions"]["temperature_fixed"][0]["values"][0]
    T1 = config["boundary_conditions"]["temperature_fixed"][1]["values"][0]

    # Gradient direction is x; extents taken from the (uncut) matrix nodes.
    nodes = mesh["nodes"]
    xmin, xmax = nodes[:, 0].min(), nodes[:, 0].max()

    nodes_total = mesh["nodes_total"]
    x = nodes_total[:, 0]
    p_exact = p0 + (p1 - p0) * (x - xmin) / (xmax - xmin)
    T_exact = T0 + (T1 - T0) * (x - xmin) / (xmax - xmin)

    p = results["p"][:, 0]
    T = results["T"][:, 0]

    if plot_on:
        plt.figure()
        plt.plot(x, p_exact, "ro", label="exact")
        plt.plot(x, p, "k.", label="RAFT")
        plt.xlabel("x (m)")
        plt.ylabel("p")
        plt.title(test_name)
        plt.legend()

        plt.figure()
        plt.plot(x, T_exact, "ro", label="exact")
        plt.plot(x, T, "k.", label="RAFT")
        plt.xlabel("x (m)")
        plt.ylabel("T")
        plt.title(test_name)
        plt.legend()

    # Relative error, including the fracture's own nodes.
    err_p = np.max(np.abs(p - p_exact)) / (p0 - p1)
    err_T = np.max(np.abs(T - T_exact)) / (T0 - T1)

    test_pass = bool(err_p < tol and err_T < tol)

    return test_pass
