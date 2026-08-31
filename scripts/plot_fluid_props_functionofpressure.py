"""Plot fluid properties and their T/P derivatives as functions of pressure."""

# ruff: noqa: N816

import CoolProp.CoolProp as CP  # noqa: N817
import matplotlib.pyplot as plt
import numpy as np

# Select fluid: 'Water' or 'CO2'
fluid = "Water"  # Change to 'CO2' for Carbon Dioxide

# Fixed temperatures in °C to evaluate across pressure sweeps
temperatures_C = [50, 150, 300] if fluid == "Water" else [0, 50, 100]

# Pressure range in MPa
P_min_MPa = 1.0
P_max_MPa = 100.0
pressures_MPa = np.linspace(P_min_MPa, P_max_MPa, 300)

# Initialize AbstractState object
AS = CP.AbstractState("HEOS", fluid)

# Create a 3x3 plot grid (Values | dT Derivatives | dP Derivatives)
fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True)

dT_K = 0.1  # 0.1 K perturbation for central difference wrt temperature

for T_C in temperatures_C:
    T_K = T_C + 273.15

    density = []
    cp = []
    viscosity = []

    d_density_dT_exact = []
    d_density_dP_exact = []
    d_cp_dT_list = []
    d_visc_dT_list = []

    # Calculate properties via AbstractState along pressure sweep
    for P_MPa in pressures_MPa:
        P_Pa = P_MPa * 1e6  # Convert MPa to Pa

        # 1. Nominal State
        AS.update(CP.PT_INPUTS, P_Pa, T_K)

        rho = AS.rhomass()  # kg/m³
        c_p = AS.cpmass() / 1000.0  # J/kg·K to kJ/kg·K
        visc = AS.viscosity() * 1e3  # Pa·s to mPa·s

        # Exact analytical derivatives for density
        dD_dT = AS.first_partial_deriv(CP.iDmass, CP.iT, CP.iP)  # kg/m³·K
        dD_dP = AS.first_partial_deriv(CP.iDmass, CP.iP, CP.iT) * 1e6  # kg/m³·MPa

        # 2. Perturbed States for (∂Cp/∂T)_P and (∂μ/∂T)_P via central difference
        AS.update(CP.PT_INPUTS, P_Pa, T_K + dT_K)
        cp_p = AS.cpmass() / 1000.0
        visc_p = AS.viscosity() * 1e3

        AS.update(CP.PT_INPUTS, P_Pa, T_K - dT_K)
        cp_m = AS.cpmass() / 1000.0
        visc_m = AS.viscosity() * 1e3

        d_cp_dT = (cp_p - cp_m) / (2 * dT_K)  # (kJ/kg·K)/K
        d_visc_dT = (visc_p - visc_m) / (2 * dT_K)  # (mPa·s)/K

        density.append(rho)
        cp.append(c_p)
        viscosity.append(visc)
        d_density_dT_exact.append(dD_dT)
        d_density_dP_exact.append(dD_dP)
        d_cp_dT_list.append(d_cp_dT)
        d_visc_dT_list.append(d_visc_dT)

    density_arr = np.array(density)
    cp_arr = np.array(cp)
    viscosity_arr = np.array(viscosity)

    # Temperature derivatives (Col 2)
    d_density_dT = np.array(d_density_dT_exact)
    d_cp_dT_arr = np.array(d_cp_dT_list)
    d_visc_dT_arr = np.array(d_visc_dT_list)

    # Pressure derivatives (Col 3) - evaluated via gradient along x-axis (pressure)
    d_density_dP = np.array(d_density_dP_exact)
    d_cp_dP = np.gradient(cp_arr, pressures_MPa)
    d_visc_dP = np.gradient(viscosity_arr, pressures_MPa)

    label = f"{T_C:g} °C"

    # Column 1: Values
    axes[0, 0].plot(pressures_MPa, density_arr, label=label)
    axes[1, 0].plot(pressures_MPa, cp_arr, label=label)
    axes[2, 0].plot(pressures_MPa, viscosity_arr, label=label)

    # Column 2: Temperature Derivatives (∂ / ∂T)
    axes[0, 1].plot(pressures_MPa, d_density_dT, label=label)
    axes[1, 1].plot(pressures_MPa, d_cp_dT_arr, label=label)
    axes[2, 1].plot(pressures_MPa, d_visc_dT_arr, label=label)

    # Column 3: Pressure Derivatives (∂ / ∂P)
    axes[0, 2].plot(pressures_MPa, d_density_dP, label=label)
    axes[1, 2].plot(pressures_MPa, d_cp_dP, label=label)
    axes[2, 2].plot(pressures_MPa, d_visc_dP, label=label)

# Axis Labels
axes[0, 0].set_ylabel("Density (kg/m³)")
axes[1, 0].set_ylabel("Specific Heat $C_p$ (kJ/kg·K)")
axes[2, 0].set_ylabel("Dynamic Viscosity (mPa·s)")

axes[0, 1].set_ylabel(r"$\partial \rho / \partial T$ (kg/m³·K)")
axes[1, 1].set_ylabel(r"$\partial C_p / \partial T$ (kJ/kg·K²)")
axes[2, 1].set_ylabel(r"$\partial \mu / \partial T$ (mPa·s/K)")

axes[0, 2].set_ylabel(r"$\partial \rho / \partial P$ (kg/m³·MPa)")
axes[1, 2].set_ylabel(r"$\partial C_p / \partial P$ (kJ/kg·K·MPa)")
axes[2, 2].set_ylabel(r"$\partial \mu / \partial P$ (mPa·s/MPa)")

for col in range(3):
    axes[2, col].set_xlabel("Pressure (MPa)")

# Titles
titles = [
    ["Density", r"$\partial\rho/\partial T$", r"$\partial\rho/\partial P$"],
    ["Specific Heat Capacity", r"$\partial C_p/\partial T$", r"$\partial C_p/\partial P$"],
    ["Dynamic Viscosity", r"$\partial\mu/\partial T$", r"$\partial\mu/\partial P$"],
]

for row in range(3):
    for col in range(3):
        axes[row, col].grid(True)
        axes[row, col].set_title(titles[row][col], fontsize=10, fontweight="bold")
        axes[row, col].legend(title="Temperature", fontsize="small")

plt.suptitle(f"Thermophysical Properties & Partial Derivatives ({fluid})", fontsize=14)
plt.tight_layout()
plt.show()
