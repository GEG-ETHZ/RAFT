# Copyright 2026 The RAFT Community
#
# This file is part of RAFT.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check FEM test cases."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from raft.preprocessing import load_config


def _bc_value(config: dict[str, Any], field: str, bc_type: str, index: int) -> float:
    """Return values[0] of the nth BC group whose `field` spec is of type `bc_type`."""
    groups = config["boundary_conditions"]
    matches = [
        group[field]["values"] for group in groups if field in group and group[field]["type"] == bc_type
    ]
    return float(matches[index][0])


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
    tol = 1.0e-7

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

    T0 = _bc_value(config, "thermal", "temperature", 0)
    T1 = _bc_value(config, "thermal", "temperature", 1)

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
    p0 = _bc_value(config, "hydraulics", "pressure", 0)
    p1 = _bc_value(config, "hydraulics", "pressure", 1)

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

    T0 = _bc_value(config, "thermal", "temperature", 0)
    q = _bc_value(config, "thermal", "flux", 0)
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
    p1 = _bc_value(config, "hydraulics", "pressure", 0)
    q = _bc_value(config, "hydraulics", "flux", 0)
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
    tol = 1.0e-3

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
    p0 = _bc_value(config, "hydraulics", "pressure", 1)
    p1 = _bc_value(config, "hydraulics", "pressure", 0)
    kf = mat["solid"]["permeability"]
    mu = mat["fluid"]["viscosity"]
    q = kf / mu * (p1 - p0) / L

    # conductivity
    phi = mat["solid"]["porosity"]
    lambda_e = phi * mat["fluid"]["therm_cond"] + (1 - phi) * mat["solid"]["therm_cond"]
    cp = mat["fluid"]["specific_heat"]
    rho_f = mat["fluid"]["density"]

    # peclet number
    Pe = L * (q * cp * rho_f) / lambda_e

    # analytical solution
    x = np.linspace(0, L, 100) / L
    T_exact = (np.exp(Pe * x) - 1) / (np.exp(Pe) - 1)

    T0 = _bc_value(config, "thermal", "temperature", 0)
    T1 = _bc_value(config, "thermal", "temperature", 1)

    # extract T and normalize
    T = (results["T"][:, 0] - T0) / (T1 - T0)

    if plot_on:
        plt.figure()
        plt.plot(x, T_exact, "r-", label="exact")
        plt.plot(d_1 / L, T[line_1], "ko", label="RAFT")
        plt.xlabel("z/L")
        plt.ylabel(r"$(T - T_0)/(T_L - T_0)$")
        plt.title(test_name)
        plt.text(0.05, 0.9, f"Pe = {Pe:.2f}")
        plt.legend(loc="lower right")

    # calculate error
    T_exact = (np.exp(Pe * d_1 / L) - 1) / (np.exp(Pe) - 1)
    err_ex = sum((T_exact - T[line_1]) ** 2)
    # The error here is calculated pased on the nodal solutions, which with SUPG
    # stabilization should be close to exact at the nodes. The full error
    # includes the linear interpolation between the nodes, which is much higher.

    test_pass = bool(err_ex < tol)

    return test_pass


def check_thermoelastic_cooling(
    config_name: str,
    config_dir: Path,
    analysis: dict[str, Any],
    results: dict[str, Any],
    _test_name: str,
    plot_on: bool,
    plot_path: Path | None = None,
) -> bool:
    """Check thermoelastic cooling against the cold-injection analytical solution."""
    config = load_config(config_name, config_dir, verbose=False)
    material = config["material"]
    solid = material["solid"]
    fluid = material["fluid"]

    nodes = analysis["points"]
    cells = analysis["cells"]
    x = np.array([np.mean(nodes[cell, 0]) for cell in cells])
    L = float(np.max(nodes[:, 0]) - np.min(nodes[:, 0]))

    p0 = float(config["initial_conditions"]["p_ini"])
    T0 = float(config["initial_conditions"]["T_ini"])
    mass_flux = _bc_value(config, "hydraulics", "flux", 0)
    transverse_length = float(np.max(nodes[:, 1]) - np.min(nodes[:, 1]))
    q = mass_flux / transverse_length
    pout = _bc_value(config, "hydraulics", "pressure", 0)
    Tin = _bc_value(config, "thermal", "temperature", 0)
    Tout = _bc_value(config, "thermal", "temperature", 1)

    bulk_modulus = float(solid["bulk_mod"])
    biot_coefficient = float(solid["biot_coeff"])
    drained_bulk_modulus = (1.0 - biot_coefficient) * bulk_modulus
    thermal_expansion = float(solid["therm_expansion"])
    b_bb = float(solid["b_BB"])
    stress_init = float(solid["stress_init"])
    reference_permeability = float(np.asarray(solid["permeability"], dtype=float).reshape(-1)[0])
    viscosity = float(fluid["viscosity"])

    y0 = 1.0 + b_bb * stress_init
    temperature = Tin + (Tout - Tin) * x / L
    C1 = b_bb * drained_bulk_modulus * thermal_expansion * (Tout - Tin) / L
    C2 = b_bb * biot_coefficient * q * viscosity / (reference_permeability * y0**2)
    yL = (
        y0
        + b_bb * drained_bulk_modulus * thermal_expansion * (Tout - Tin)
        - b_bb * biot_coefficient * (pout - p0)
    )
    y = np.sqrt(C1 / C2) * np.tan(np.arctan(np.sqrt(C2 / C1) * yL) + np.sqrt(C1 * C2) * (x - L))
    pressure = p0 + (y0 + b_bb * drained_bulk_modulus * thermal_expansion * (temperature - T0) - y) / (
        b_bb * biot_coefficient
    )
    stress = (
        stress_init
        + drained_bulk_modulus * thermal_expansion * (temperature - T0)
        - biot_coefficient * (pressure - p0)
    )
    aperture = np.sqrt(12.0 * reference_permeability) * y0 / y
    permeability = aperture**2 / 12.0

    actual_stress = analysis["scalars"]["effective_stress"]
    actual_aperture = analysis["scalars"]["aperture"]
    actual_permeability = analysis["scalars"]["permeability"]
    actual_pressure = np.array([np.mean(results["scalars"]["p"][cell]) for cell in cells])
    actual_temperature = np.array([np.mean(results["scalars"]["T"][cell]) for cell in cells])

    comparison_data = [
        ("Temperature (K)", temperature, actual_temperature),
        ("Pressure (Pa)", pressure, actual_pressure),
        ("Effective stress (Pa)", stress, actual_stress),
        ("Aperture (m)", aperture, actual_aperture),
        ("Permeability (m^2)", permeability, actual_permeability),
    ]
    relative_errors = [
        float(np.sum((expected - actual) ** 2) / np.sum(expected**2))
        for _, expected, actual in comparison_data
    ]
    relative_error = sum(relative_errors)

    if plot_on:
        plot_order = np.argsort(x)
        sorted_x = x[plot_order]
        unique_x, unique_indices = np.unique(sorted_x, return_index=True)
        figure, axes = plt.subplots(3, 2, figsize=(12, 12))
        figure.subplots_adjust(hspace=0.5, wspace=0.3)
        for axis, (label, expected, actual), error in zip(
            axes.ravel()[: len(comparison_data)], comparison_data, relative_errors, strict=True
        ):
            sorted_expected = expected[plot_order]
            axis.plot(unique_x, sorted_expected[unique_indices], "r-", label="Analytical")
            axis.plot(x, actual, "ko", markersize=3, linestyle="none", label="RAFT")
            axis.set_xlabel("x (m)", fontsize=14)
            axis.set_ylabel(label, fontsize=14)
            axis.set_title(label, fontsize=15)
            axis.text(
                0.5,
                0.08,
                f"relative error: {error:.3e}",
                transform=axis.transAxes,
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )
            axis.tick_params(axis="both", labelsize=12)
            axis.grid(True, alpha=0.3)
            if axis is axes.flat[0]:
                axis.legend(fontsize=12)
        axes.flat[-1].axis("off")
        if plot_path is None:
            plot_path = config_dir / "thermoelastic_cooling_comparison.png"
        figure.savefig(plot_path, dpi=150)
        plt.close(figure)

    return bool(relative_error < 1.0e-5)


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
    tol = 1.0e-5

    # Load config
    config = load_config(config_name, config_dir, verbose=False)

    # Extract nodes along lines
    nodes = mesh["nodes"]
    line_1 = np.where((nodes[:, 0] == 0) & (nodes[:, 1] == 0))[0]

    L = np.sqrt(np.sum((nodes[4, :] - nodes[0, :]) ** 2))

    d_1 = np.sqrt(np.sum((nodes[line_1, :] - nodes[0, :]) ** 2, 1))

    # Define analytical solution
    mat = config["material"]

    g = 9.81
    T0 = _bc_value(config, "thermal", "temperature", 0)
    T1 = _bc_value(config, "thermal", "temperature", 1)
    dT = T1 - T0
    # density and expansion referenced to T0 (x = 0)
    T_ref = mat["fluid"]["reference_temperature"]
    shift = 1 - mat["fluid"]["therm_expansion"] * (T0 - T_ref)
    rho0 = mat["fluid"]["density"] * shift
    alpha_f = mat["fluid"]["therm_expansion"] / shift
    p0 = _bc_value(config, "hydraulics", "pressure", 0)
    p1 = _bc_value(config, "hydraulics", "pressure", 1)

    # analytical solution
    x = np.linspace(0, L, 100) / L
    A = alpha_f * dT
    B = rho0 * g * L

    def _p_exact(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        C = (p1 - p0) / B - (A / 2 - 1)
        p: npt.NDArray[np.float64] = p0 + B * (C * np.log(1 - A * x) / np.log(1 - A) + (A * x / 2 - 1) * x)
        return p

    p_exact = _p_exact(x)

    # extract T and normalize
    p = results["p"][:, 0]

    if plot_on:
        plt.figure()
        plt.plot(x, p_exact, "r-", label="exact")
        plt.plot(d_1 / L, p[line_1], "ko", label="RAFT")
        plt.xlabel("z/L")
        plt.ylabel("p (Pa)")
        plt.title(test_name)
        plt.legend()

        # mu = mat["fluid"]["viscosity"]
        # k = mat["solid"]["permeability"]
        # q_exact = -k/mu * (2*beta * x - beta * L + p1/L)*x
        # indx = np.argsort(d_1)
        # dx = d_1[2]*np.ones(np.shape(d_1))
        # #dx[0] = 0.5*dx[0]
        # #dx[-1] = 0.5*dx[-1]
        # p_num = p[line_1][indx]
        # q_num = -k/mu * np.gradient(p_num) / dx
        # plt.figure()
        # plt.plot(x, q_exact, "r--")
        # plt.plot(d_1/L, q_num, "ko")
        # plt.ylabel("q")

    # calculate error
    x = d_1 / L
    p_exact = _p_exact(x)
    err_ex = sum((p_exact - p[line_1]) ** 2) / sum(p_exact**2)
    # The error here is calculated based on the nodal solutions.

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
    """Check heterogeneous layers. Result should be an exact piecewise-linear solution."""
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

    T0 = _bc_value(config, "thermal", "temperature", 0)
    q = _bc_value(config, "thermal", "flux", 0)
    A = max(mesh["nodes"][:, 0]) * max(mesh["nodes"][:, 1])
    q = -q / A * 4

    k = config["material"]["solid"]["therm_cond"]

    T_ex = T0 * np.ones(len(d_1))
    for i in range(len(d_1) - 1):
        T_ex[i + 1] = T_ex[i] - q / k[regions[i + 1]] * (d_1[i + 1] - d_1[i])

    T = results["T"][:, 0]

    if plot_on:
        plt.figure()
        plt.plot(d_1, T_ex, "r-", label="exact")
        plt.plot(d_1, T[line_1], "ko", label="RAFT")
        plt.xlabel("z (m)")
        plt.ylabel("T")
        plt.title(test_name)
        plt.legend()

    # calculate error
    err_ex = sum((T_ex - T[line_1]) ** 2 / T_ex**2)

    # check pressure
    p0 = _bc_value(config, "hydraulics", "pressure", 0)
    q = _bc_value(config, "hydraulics", "flux", 0)
    q = -q / A * 4

    k = config["material"]["solid"]["permeability"]
    mu = config["material"]["fluid"]["viscosity"]

    p_ex = p0 * np.ones(len(d_1))
    for i in range(len(d_1) - 1):
        p_ex[i + 1] = p_ex[i] - mu * q / k[regions[i + 1]] * (d_1[i + 1] - d_1[i])

    p = results["p"][:, 0]

    if plot_on:
        plt.figure()
        plt.plot(d_1, p_ex, "b-", label="exact")
        plt.plot(d_1, p[line_1], "ko", label="RAFT")
        plt.xlabel("z (m)")
        plt.ylabel("p (Pa)")
        plt.title(test_name)
        plt.legend()

    err_ex += sum((p_ex - p[line_1]) ** 2 / p_ex**2)

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
    p0 = _bc_value(config, "hydraulics", "pressure", 0)
    q = _bc_value(config, "hydraulics", "flux", 0)

    # temperature
    T0 = _bc_value(config, "thermal", "temperature", 0)
    T1 = _bc_value(config, "thermal", "temperature", 1)

    # conductivity
    phi = mat["solid"]["porosity"]
    lambda_e = phi * mat["fluid"]["therm_cond"] + (1 - phi) * mat["solid"]["therm_cond"]
    cp = mat["fluid"]["specific_heat"]
    rho_f = mat["fluid"]["density"]

    q = q / rho_f

    # peclet number per length
    Pe = L * abs(q * cp * rho_f) / lambda_e

    # analytical solution
    x = np.linspace(0, L, 100) / L
    T_exact = (np.exp(Pe * x) - 1) / (np.exp(Pe) - 1)
    T_exact = T0 + (T1 - T0) * T_exact

    # Viscosity - real-fluid (CoolProp) viscosity. Self-consistent with the local pressure
    # via fixed-point refinement: p_exact depends on visc_exact(p), and vice versa.
    k = mat["solid"]["permeability"]
    dx = x[1] * L
    p_exact = np.full(100, config["initial_conditions"]["p_ini"])
    for _ in range(3):
        visc_exact = get_viscosity(T_exact, p_exact, mat["fluid"]).value
        p_exact[0] = p0
        for i in range(1, 100):
            p_exact[i] = p_exact[i - 1] + dx * q / k * 0.5 * (visc_exact[i - 1] + visc_exact[i])

    # extract results
    p = results["p"][:, 0]
    T = results["T"][:, 0]
    visc_approx = get_viscosity(T[line_1], p[line_1], mat["fluid"]).value

    if plot_on:
        plt.figure()
        plt.plot(x, T_exact, "r-", label="exact")
        plt.plot(d_1 / L, T[line_1], "ko", label="RAFT")
        plt.xlabel("z/L")
        plt.ylabel("T (K)")
        plt.title(test_name)
        plt.legend()

        plt.figure()
        plt.plot(x, visc_exact, "r--", label="exact")
        plt.plot(d_1 / L, visc_approx, "ko", label="RAFT")
        plt.xlabel("z/L")
        plt.ylabel(r"$\mu$ (Pa$\cdot$s)")
        plt.title(test_name)
        plt.legend()

        plt.figure()
        plt.plot(x, p_exact, "b-", label="semi-analytical")
        plt.plot(d_1 / L, p[line_1], "ko", label="RAFT")
        plt.xlabel("z/L")
        plt.ylabel("p (Pa)")
        plt.title(test_name)
        plt.legend()

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

    T0 = _bc_value(config, "thermal", "temperature", 0)
    q = _bc_value(config, "thermal", "flux", 0)
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
        plt.plot(r, T_exact, "r-", label="exact")
        plt.plot(d_1, T[line_1], "ko", label="RAFT, lower edge")
        plt.plot(d_2, T[line_2], "bo", mfc="none", label="RAFT, upper edge")
        plt.xlabel("r (m)")
        plt.ylabel("T")
        plt.title(test_name)
        plt.legend()

    # calculate error
    mean_val = np.mean(T_exact)
    err_ex = sum((_y_exact(d_1, T0, q, k, Ri, Ro) - T[line_1]) ** 2) / mean_val**2
    err_ex += sum((_y_exact(d_2, T0, q, k, Ri, Ro) - T[line_2]) ** 2) / mean_val**2

    # check pressure
    p0 = _bc_value(config, "hydraulics", "pressure", 0)
    q = _bc_value(config, "hydraulics", "flux", 0)
    k = config["material"]["solid"]["permeability"] / config["material"]["fluid"]["viscosity"]

    # define exact solution
    p_exact = _y_exact(r, p0, q, k, Ri, Ro)

    p = results["p"][:, 0]

    if plot_on:
        plt.figure()
        plt.plot(r, p_exact, "b-", label="exact")
        plt.plot(d_1, p[line_1], "ko", label="RAFT, lower edge")
        plt.plot(d_2, p[line_2], "bo", mfc="none", label="RAFT, upper edge")
        plt.xlabel("r (m)")
        plt.ylabel("p (Pa)")
        plt.title(test_name)
        plt.legend()

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

    p0 = _bc_value(config, "hydraulics", "pressure", 0)
    p1 = _bc_value(config, "hydraulics", "pressure", 1)
    T0 = _bc_value(config, "thermal", "temperature", 0)
    T1 = _bc_value(config, "thermal", "temperature", 1)

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


def check_fracture_advection_bounded(
    config_name: str,
    config_dir: Path,
    results: dict[str, Any],
    mesh: dict[str, Any],
    test_name: str,
    plot_on: bool,
) -> bool:
    """Check maximum-principle boundedness for an angled (non-perpendicular) fracture with
    advective leak-off turned on.

    With advection_on and no internal sources, steady p/T have no interior extrema: every
    node - including the fracture's own leak-off nodes and both sides of the matrix - must
    stay within the range spanned by the imposed boundary values. A sign error in the
    fracture-matrix coupling's advective tangent (`_integrate_fracture_coupling_tangent`
    in matrixassembly.py) or a wrong upwind branch tends to show up as overshoot beyond this
    envelope rather than a subtly wrong interior profile, so this is a cheap discriminating
    check even without an analytical reference solution.
    """
    tol_rel = 1.0e-6

    config = load_config(config_name, config_dir, verbose=False)

    p0 = _bc_value(config, "hydraulics", "pressure", 0)
    p1 = _bc_value(config, "hydraulics", "pressure", 1)
    T0 = _bc_value(config, "thermal", "temperature", 0)
    T1 = _bc_value(config, "thermal", "temperature", 1)

    p_lo, p_hi = min(p0, p1), max(p0, p1)
    T_lo, T_hi = min(T0, T1), max(T0, T1)

    p = results["p"][:, 0]
    T = results["T"][:, 0]

    if plot_on:
        nodes_total = mesh["nodes_total"]
        plt.figure()
        plt.scatter(nodes_total[:, 0], nodes_total[:, 1], c=T)
        plt.colorbar(label="T")
        plt.title(test_name)

    p_bounded = bool(
        np.all(p >= p_lo - tol_rel * (p_hi - p_lo)) and np.all(p <= p_hi + tol_rel * (p_hi - p_lo))
    )
    T_bounded = bool(
        np.all(T_lo - tol_rel * (T_hi - T_lo) <= T) and np.all(T_hi + tol_rel * (T_hi - T_lo) >= T)
    )

    return p_bounded and T_bounded


def check_enthalpy_bndry_bounded(
    config_name: str,
    config_dir: Path,
    results: dict[str, Any],
    mesh: dict[str, Any],
    test_name: str,
    plot_on: bool,
) -> bool:
    """Check the outlet's temperature against the exact solution for an outlet with no
    thermal BC of its own.

    The outlet's temperature is free, advected out through
    `integrate_enthlpy_bndry_residual`/`integrate_enthlpy_bndry_tangent` in matrixassembly.py
    rather than fixed by a Dirichlet condition, and (for a pressure-type outlet) also picks up
    the `fully_coupled` Kpp/KpT cross-term there. With a single Dirichlet inlet value, no
    sources, and steady, divergence-free flow, the uniform field T = T_in exactly satisfies
    the advection-diffusion equation everywhere (both the diffusive and advective terms vanish
    identically for a spatially constant field) and the natural BCs elsewhere - so T should
    equal T_in to within discretization error, not just fall inside some broad envelope.
    Missing or sign-flipped advective outflow would instead let heat build up at the outlet,
    pushing T away from T_in.
    """
    tol_rel = 1.0e-3

    config = load_config(config_name, config_dir, verbose=False)

    p_in = _bc_value(config, "hydraulics", "pressure", 0)
    T_in = _bc_value(config, "thermal", "temperature", 0)
    p_ini = float(config["initial_conditions"]["p_ini"])
    T_ini = float(config["initial_conditions"]["T_ini"])

    p = results["p"][:, 0]
    T = results["T"][:, 0]

    if plot_on:
        nodes_total = mesh["nodes_total"]
        plt.figure()
        plt.scatter(nodes_total[:, 0], nodes_total[:, 1], c=T)
        plt.colorbar(label="T")
        plt.title(test_name)

    T_scale = abs(T_in - T_ini)
    T_bounded = bool(np.all(np.abs(T - T_in) <= tol_rel * T_scale))

    # p isn't pinned at the outlet in the flux-type variant, so there's no simple a-priori
    # bound to check it against; just guard against the field blowing up or going non-physical.
    p_scale = max(abs(p_in), abs(p_ini))
    p_sane = bool(np.all(np.isfinite(p)) and np.all(p > 0) and np.all(np.abs(p) < 10 * p_scale))

    return T_bounded and p_sane


def check_well_convection(
    config_name: str,
    config_dir: Path,
    results: dict[str, Any],
    mesh: dict[str, Any],
    test_name: str,
    plot_on: bool,
) -> bool:
    """Check well convection against analytical solution."""
    # Set tolerance
    tol = 1.0e-5

    # Load config
    config = load_config(config_name, config_dir, verbose=False)

    # Extract nodes along lines
    nodes = mesh["nodes_total"]
    n1 = mesh["nn"]
    n2 = mesh["nn_total"]
    line_1 = np.arange(n1, n2)  # well nodes

    L = np.sqrt(np.sum((nodes[4, :] - nodes[0, :]) ** 2))

    d_1 = np.sqrt(np.sum((nodes[line_1, :] - nodes[line_1[0], :]) ** 2, 1))

    # Define analytical solution
    mat = config["material"]
    # BCs
    Tm = _bc_value(config, "thermal", "temperature", 0)
    T0 = _bc_value(config, "thermal", "temperature", 1)
    q = _bc_value(config, "hydraulics", "flux", 0)
    # properties
    mu = mat["fluid"]["viscosity"]
    cp = mat["fluid"]["specific_heat"]
    rho = mat["fluid"]["density"]
    D = mat["wells"]["diameter"][0]
    R = D * 0.5
    lambda_f = mat["fluid"]["therm_cond"]

    Nu = 4.01

    hc = Nu * lambda_f / D

    # analytical solution from constant heating in the rock with pressure work
    x = np.linspace(0, L, 100)
    h1 = hc * np.pi * D
    friction_heating = 8 * mu * q**2 / (np.pi * h1 * rho**2 * R**4)
    T_exact = T0 + (Tm - T0 + friction_heating) * (1 - np.exp(-h1 / q / cp * x))

    # extract T and normalize
    T = results["T"][:, 0]

    if plot_on:
        plt.figure()
        plt.plot(x / L, T_exact, "r-", label="exact")
        plt.plot(d_1 / L, T[line_1], "ko", label="RAFT")
        plt.xlabel("s/L")
        plt.ylabel("T (K)")
        plt.title(test_name)
        plt.legend()

    # calculate error
    x = d_1
    T_exact = T0 + (Tm - T0 + friction_heating) * (1 - np.exp(-h1 / q / cp * x))

    err_ex = sum((T_exact - T[line_1]) ** 2) / sum(T_exact**2)

    test_pass = bool(err_ex < tol)

    return test_pass
