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
from scipy.optimize import brentq

from geg_raft.preprocessing import load_config, preprocessing
from geg_raft.solidmechanics import insitu_stress, nodal_stress
from geg_raft.subsurface import advance, initialize_simulation


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
    from geg_raft.matprops import get_viscosity

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
    the `monolithic` Kpp/KpT cross-term there. With a single Dirichlet inlet value, no
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


def _elastic_moduli(config: dict[str, Any]) -> tuple[float, float]:
    """Drained bulk and shear moduli from the config's solid properties."""
    solid = config["material"]["solid"]
    K_d = (1.0 - solid["biot_coeff"]) * solid["bulk_mod"]
    nu = solid["poisson_ratio"]
    return K_d, 3.0 * K_d * (1.0 - 2.0 * nu) / (2.0 * (1.0 + nu))


def check_patch_test_elastic(
    config_name: str,
    config_dir: Path,
    results: dict[str, Any],
    mesh: dict[str, Any],
    test_name: str,
    plot_on: bool,
) -> bool:
    """Check elastic patch test. Displacement should be exactly linear and stress constant."""
    # Set tolerance
    tol = 1.0e-9

    # Load config
    config = load_config(config_name, config_dir, verbose=False)
    nodes = mesh["nodes"]
    nsd = mesh["nsd"]

    # exact field from the gradient BCs: u_i = c + m (s . X - min s . X)
    u = results["u"][:, 0].reshape(-1, nsd)
    u_exact = np.zeros_like(u)
    grad_u = np.zeros((nsd, nsd))
    for group in config["boundary_conditions"]:
        spec = group["mechanics"]
        i = "xyz".index(spec["components"][0])
        direction, c, slope = spec["values"]
        s = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
        u_exact[:, i] = c + slope * (nodes @ s - (nodes @ s).min())
        grad_u[i] = slope * s

    if plot_on:
        plt.figure()
        for i in range(nsd):
            plt.plot(u_exact[:, i], u[:, i], "o", label="xyz"[i])
        plt.xlabel("u exact (m)")
        plt.ylabel("u (m)")
        plt.title(test_name)
        plt.legend()

    # stress: sigma_0 + lambda tr(eps) I + 2 G eps, in Voigt order
    K_d, G = _elastic_moduli(config)
    eps = 0.5 * (grad_u + grad_u.T)
    sig = (K_d - 2.0 * G / 3.0) * np.trace(eps) * np.eye(nsd) + 2.0 * G * eps
    if nsd == 2:
        voigt = [sig[0, 0], sig[1, 1], sig[0, 1]]
    else:
        voigt = [sig[0, 0], sig[1, 1], sig[2, 2], sig[0, 2], sig[1, 2], sig[0, 1]]
    sigma_0 = insitu_stress(config["initial_conditions"]["stress"], nodes, nsd, False)
    sigma_exact = sigma_0 + np.array(voigt)

    # nodal stress; flow and heat off, so p and T stay at their reference values
    cfg, _ = preprocessing(config_name, config_dir, verbose=False)
    material = cfg["material"]
    material["p_0"] = results["p"][:, 0]
    material["T_0"] = results["T"][:, 0]
    material["sigma_0"] = sigma_0
    sigma = nodal_stress(mesh, material, cfg["simulation"], results)

    # calculate error
    err_u = np.sum((u - u_exact) ** 2) / np.sum(u_exact**2)
    err_sigma = np.sum((sigma - sigma_exact) ** 2) / np.sum(sigma_exact**2)

    test_pass = bool(err_u < tol**2 and err_sigma < tol**2)

    return test_pass


def check_lame_cylinder(
    config_name: str,
    config_dir: Path,
    results: dict[str, Any],
    mesh: dict[str, Any],
    test_name: str,
    plot_on: bool,
) -> bool:
    """Axisymmetric thick-walled cylinder under internal pressure in plane strain (Lame).
    Linear elements do not capture 1/r exactly, use high tolerance."""
    # Set tolerance
    tol = 5.0e-3

    # Load config
    config = load_config(config_name, config_dir, verbose=False)

    p_in = _bc_value(config, "mechanics", "traction", 0)
    K_d, G = _elastic_moduli(config)
    nu = config["material"]["solid"]["poisson_ratio"]
    E = 2.0 * G * (1.0 + nu)

    r = mesh["nodes"][:, 0]
    a, b = r.min(), r.max()

    # define exact solution
    def _u_exact(r: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        u_r: npt.NDArray[np.float64] = (
            (1.0 + nu) * a**2 * p_in / (E * (b**2 - a**2)) * ((1.0 - 2.0 * nu) * r + b**2 / r)
        )
        return u_r

    u = results["u"][:, 0].reshape(-1, 2)

    if plot_on:
        r_plot = np.linspace(a, b, 200)
        plt.figure()
        plt.plot(r_plot, _u_exact(r_plot), "r-", label="exact")
        line_1 = np.nonzero(mesh["nodes"][:, 1] == min(mesh["nodes"][:, 1]))[0]
        line_2 = np.nonzero(mesh["nodes"][:, 1] == max(mesh["nodes"][:, 1]))[0]
        plt.plot(r[line_1], u[line_1, 0], "ko", label="RAFT, lower edge")
        plt.plot(r[line_2], u[line_2, 0], "bo", mfc="none", label="RAFT, upper edge")
        plt.xlabel("r (m)")
        plt.ylabel("u_r (m)")
        plt.title(test_name)
        plt.legend()

    # calculate error
    err_ex = np.linalg.norm(u[:, 0] - _u_exact(r)) / np.linalg.norm(_u_exact(r))

    test_pass = bool(err_ex < tol and np.max(np.abs(u[:, 1])) < 1e-12)

    return test_pass


# Output times of the consolidation checks
TERZAGHI_TIMES = [0.5, 2.0, 8.0, 20.0]
MANDEL_TIMES = [0.05, 0.5, 2.0, 8.0, 20.0]


def run_at_times(
    config_name: str, config_dir: Path, times: list[float]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run a config, returning the primary variables at each of `times` and the mesh."""
    config, mesh = preprocessing(config_name, config_dir, verbose=False)
    config["simulation"].update(config_name=config["config_name"], config_dir=config["config_dir"])
    setup, state = initialize_simulation(config, mesh, config["simulation"]["t_start"], False)
    results = []
    for t in times:
        state, _ = advance(setup, state, t, False)
        results.append({key: state.primary_variables[key].copy() for key in ("u", "p")})
    return results, mesh


def _poroelastic_constants(config: dict[str, Any]) -> dict[str, float]:
    """Biot constants of a homogeneous config."""
    solid = config["material"]["solid"]
    fluid = config["material"]["fluid"]
    K_d, G = _elastic_moduli(config)
    biot = solid["biot_coeff"]
    nu = solid["poisson_ratio"]
    M = 1.0 / (solid["porosity"] / fluid["bulk_mod"] + (biot - solid["porosity"]) / solid["bulk_mod"])
    K_v = K_d + 4.0 * G / 3.0  # constrained modulus
    K_u = K_d + biot**2 * M  # undrained bulk modulus
    return {
        "biot": biot,
        "G": G,
        "nu": nu,
        "nu_u": (3.0 * K_u - 2.0 * G) / (2.0 * (3.0 * K_u + G)),
        "K_v": K_v,
        "B": biot * M / K_u,  # Skempton coefficient
        "S": 1.0 / M + biot**2 / K_v,  # uniaxial storage
        "c": solid["permeability"] / fluid["viscosity"] / (1.0 / M + biot**2 / K_v),  # consolidation
    }


def check_terzaghi(
    config_name: str,
    config_dir: Path,
    results: list[dict[str, Any]],
    mesh: dict[str, Any],
    times: list[float],
    dimension: int,
    test_name: str,
    plot_on: bool,
) -> bool:
    """One-dimensional consolidation under a step load: excess pressure and displacement profiles."""
    # Set tolerance
    tol = 2.0e-2

    # Load config
    config = load_config(config_name, config_dir, verbose=False)
    constants = _poroelastic_constants(config)
    biot, K_v, c = constants["biot"], constants["K_v"], constants["c"]

    nodes = mesh["nodes"]
    nsd = mesh["nsd"]
    y = nodes[:, dimension]
    H = y.max()
    P = -_bc_value(config, "mechanics", "traction", 0)
    p_bc = _bc_value(config, "hydraulics", "pressure", 0)
    p_undrained = biot / (K_v * constants["S"]) * P

    # define exact solution
    m = 2.0 * np.arange(200) + 1.0

    def _p_exact(y: npt.NDArray[np.float64], t: float) -> npt.NDArray[np.float64]:
        modes = 4.0 / (m * np.pi) * np.exp(-(m**2) * np.pi**2 * c * t / (4.0 * H**2))
        p_e: npt.NDArray[np.float64] = p_undrained * (np.sin(np.outer(H - y, m) * np.pi / (2.0 * H)) @ modes)
        return p_e

    def _u_exact(y: npt.NDArray[np.float64], t: float) -> npt.NDArray[np.float64]:
        # uniaxial strain (-P + biot p) / K_v, integrated from the fixed base
        modes = 4.0 / (m * np.pi) * np.exp(-(m**2) * np.pi**2 * c * t / (4.0 * H**2)) * 2.0 * H / (m * np.pi)
        p_integral = p_undrained * (np.cos(np.outer(H - y, m) * np.pi / (2.0 * H)) @ modes)
        u_e: npt.NDArray[np.float64] = (-P * y + biot * p_integral) / K_v
        return u_e

    if plot_on:
        fig_p, ax_p = plt.subplots()
        fig_u, ax_u = plt.subplots()

    err_ex = 0.0
    y_plot = np.linspace(0.0, H, 200)
    order = np.argsort(y)
    for t, result in zip(times, results, strict=True):
        p = result["p"][:, 0] - p_bc
        u = result["u"][:, 0].reshape(-1, nsd)[:, dimension]

        # calculate error
        err_p = np.linalg.norm(p - _p_exact(y, t)) / np.linalg.norm(_p_exact(y, t))
        err_u = np.linalg.norm(u - _u_exact(y, t)) / np.linalg.norm(_u_exact(y, t))
        err_ex = max(err_ex, float(err_p), float(err_u))

        if plot_on:
            label = f"t = {t:g} s"
            line = ax_p.plot(_p_exact(y_plot, t) / P, y_plot / H, "-", label=label)[0]
            ax_p.plot(p[order] / P, y[order] / H, "o", color=line.get_color(), mfc="none", ms=3)
            line = ax_u.plot(_u_exact(y_plot, t) * K_v / (P * H), y_plot / H, "-", label=label)[0]
            ax_u.plot(u[order] * K_v / (P * H), y[order] / H, "o", color=line.get_color(), mfc="none", ms=3)

    if plot_on:
        ax_p.set_xlabel("(p - p_b) / P")
        ax_p.set_ylabel("y / H")
        ax_p.set_title(test_name)
        ax_p.legend()
        ax_u.set_xlabel("u K_v / (P H)")
        ax_u.set_ylabel("y / H")
        ax_u.set_title(test_name)
        ax_u.legend()

    test_pass = bool(err_ex < tol)

    return test_pass


def check_mandel(
    config_name: str,
    config_dir: Path,
    results: list[dict[str, Any]],
    mesh: dict[str, Any],
    times: list[float],
    test_name: str,
    plot_on: bool,
) -> bool:
    """Mandel's problem: pressure across the strip, with the Mandel-Cryer rise at the centre."""
    # Set tolerance
    tol = 3.0e-2

    # Load config
    config = load_config(config_name, config_dir, verbose=False)
    constants = _poroelastic_constants(config)
    nu, nu_u, c = constants["nu"], constants["nu_u"], constants["c"]

    x = mesh["nodes"][:, 0]
    a = x.max()
    p_bc = _bc_value(config, "hydraulics", "pressure", 0)
    q = 1.0e6  # mean plate load behind the displacement table in the config
    p_undrained = constants["B"] * (1.0 + nu_u) * q / 3.0

    # roots of tan(alpha) = (1 - nu) / (nu_u - nu) alpha
    C = (1.0 - nu) / (nu_u - nu)

    def _root_equation(alpha: float) -> float:
        return float(np.tan(alpha) - C * alpha)

    alpha = np.array(
        [brentq(_root_equation, 1.0e-9, np.pi / 2.0 - 1.0e-12)]
        + [brentq(_root_equation, n * np.pi + 1.0e-9, (n + 0.5) * np.pi - 1.0e-12) for n in range(1, 300)]
    )
    A = np.sin(alpha) / (alpha - np.sin(alpha) * np.cos(alpha))

    # define exact solution
    def _p_exact(x: npt.NDArray[np.float64], t: float) -> npt.NDArray[np.float64]:
        modes = A * np.exp(-(alpha**2) * c * t / a**2)
        p_e: npt.NDArray[np.float64] = (
            2.0 * p_undrained * ((np.cos(np.outer(x, alpha) / a) - np.cos(alpha)) @ modes)
        )
        return p_e

    if plot_on:
        plt.figure()

    err_ex = 0.0
    x_plot = np.linspace(0.0, a, 200)
    centre = np.nonzero(x == 0.0)[0]
    p_centre = []
    monotone = True
    base = np.nonzero(mesh["nodes"][:, 1] == 0.0)[0]
    base = base[np.argsort(x[base])]
    for t, result in zip(times, results, strict=True):
        p = result["p"][:, 0] - p_bc

        # calculate error
        err_ex = max(err_ex, float(np.linalg.norm(p - _p_exact(x, t)) / np.linalg.norm(_p_exact(x, t))))
        p_centre.append(np.mean(p[centre]) / p_undrained)

        # no oscillation: the pressure falls steadily from its peak to the drained edge
        peak = int(np.argmax(p[base]))
        monotone = monotone and bool(np.all(np.diff(p[base][peak:]) <= 0.0))

        if plot_on:
            line = plt.plot(x_plot / a, _p_exact(x_plot, t) / p_undrained, "-", label=f"t = {t:g} s")[0]
            plt.plot(x / a, p / p_undrained, "o", color=line.get_color(), mfc="none", ms=3)

    if plot_on:
        plt.xlabel("x / a")
        plt.ylabel("(p - p_b) / p_undrained")
        plt.title(test_name)
        plt.legend()

    # Mandel-Cryer effect: the centre pressure first rises above its undrained value
    test_pass = bool(err_ex < tol and max(p_centre) > 1.0 and monotone)

    return test_pass
