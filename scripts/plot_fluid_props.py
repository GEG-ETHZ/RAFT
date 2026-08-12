"""Plot fluid properties and derivatives."""

import CoolProp.CoolProp as CP  # noqa
import matplotlib.pyplot as plt
import numpy as np

# Select fluid: 'Water' or 'CO2'
fluid = "CO2"  # Change to 'CO2' for Carbon Dioxide

# Pressures in MPa
pressures_MPa = [10, 20, 30]  # noqa

# Initialize AbstractState object
AS = CP.AbstractState("HEOS", fluid)
P_crit = AS.p_critical()  # Critical pressure in Pa

# Create a 3x3 plot grid (Values | dT Derivatives | dP Derivatives)
fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True)

for P_MPa in pressures_MPa:
    P_Pa = P_MPa * 1e6  # Convert MPa to Pa

    # Handle phase boundary (Subcritical vs Supercritical)
    if P_Pa < P_crit:
        # Subcritical: limit temperature up to the saturation line with margin for dP perturbation
        AS.update(CP.PQ_INPUTS, P_Pa, 0)
        T_max_C = AS.T() - 273.15 - 0.5
        T_min_C = 5.0 if fluid == "Water" else -30.0
    else:
        # Supercritical: evaluate across the critical transition region
        T_max_C = 400.0 if fluid == "Water" else 150.0
        T_min_C = 5.0 if fluid == "Water" else 0.0

    temperatures_C = np.linspace(T_min_C, T_max_C, 300)  # noqa
    temperatures_K = temperatures_C + 273.15  # noqa

    density = []
    cp = []
    viscosity = []

    d_density_dT_exact = []  # noqa
    d_density_dP_exact = []  # noqa
    d_cp_dP_list = []  # noqa
    d_visc_dP_list = []  # noqa

    dP_Pa = 1e1  # 1 kPa perturbation for central difference wrt pressure # noqa

    # Calculate properties via AbstractState
    for T_K in temperatures_K:
        # 1. Nominal State
        AS.update(CP.PT_INPUTS, P_Pa, T_K)

        rho = AS.rhomass()  # kg/m³
        c_p = AS.cpmass() / 1000.0  # J/kg·K to kJ/kg·K
        visc = AS.viscosity() * 1e3  # Pa·s to mPa·s

        # Exact analytical derivatives for density
        dD_dT = AS.first_partial_deriv(CP.iDmass, CP.iT, CP.iP)  # kg/m³·K # noqa
        dD_dP = AS.first_partial_deriv(CP.iDmass, CP.iP, CP.iT) * 1e6  # kg/m³·Pa to kg/m³·MPa # noqa

        # 2. Perturbed States for (∂Cp/∂P)_T and (∂μ/∂P)_T via central difference
        AS.update(CP.PT_INPUTS, P_Pa + dP_Pa, T_K)
        cp_p = AS.cpmass() / 1000.0
        visc_p = AS.viscosity() * 1e3

        AS.update(CP.PT_INPUTS, P_Pa - dP_Pa, T_K)
        cp_m = AS.cpmass() / 1000.0
        visc_m = AS.viscosity() * 1e3

        d_cp_dP = (cp_p - cp_m) / (2 * dP_Pa / 1e6)  # (kJ/kg·K)/MPa # noqa
        d_visc_dP = (visc_p - visc_m) / (2 * dP_Pa / 1e6)  # (mPa·s)/MPa # noqa

        density.append(rho)
        cp.append(c_p)
        viscosity.append(visc)
        d_density_dT_exact.append(dD_dT)
        d_density_dP_exact.append(dD_dP)
        d_cp_dP_list.append(d_cp_dP)
        d_visc_dP_list.append(d_visc_dP)

    density_arr = np.array(density)
    cp_arr = np.array(cp)
    viscosity_arr = np.array(viscosity)

    # Temperature derivatives (Col 2)
    d_density_dT = np.array(d_density_dT_exact)  # noqa
    d_cp_dT = np.gradient(cp_arr, temperatures_C)  # noqa
    d_visc_dT = np.gradient(viscosity_arr, temperatures_C)  # noqa

    # Pressure derivatives (Col 3)
    d_density_dP = np.array(d_density_dP_exact)  # noqa
    d_cp_dP_arr = np.array(d_cp_dP_list)  # noqa
    d_visc_dP_arr = np.array(d_visc_dP_list)  # noqa

    label = f"{P_MPa:g} MPa"

    # Column 1: Values
    axes[0, 0].plot(temperatures_C, density_arr, label=label)
    axes[1, 0].plot(temperatures_C, cp_arr, label=label)
    axes[2, 0].plot(temperatures_C, viscosity_arr, label=label)

    # Column 2: Temperature Derivatives (∂ / ∂T)
    axes[0, 1].plot(temperatures_C, d_density_dT, label=label)
    axes[1, 1].plot(temperatures_C, d_cp_dT, label=label)
    axes[2, 1].plot(temperatures_C, d_visc_dT, label=label)

    # Column 3: Pressure Derivatives (∂ / ∂P)
    axes[0, 2].plot(temperatures_C, d_density_dP, label=label)
    axes[1, 2].plot(temperatures_C, d_cp_dP_arr, label=label)
    axes[2, 2].plot(temperatures_C, d_visc_dP_arr, label=label)

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
    axes[2, col].set_xlabel("Temperature (°C)")

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
        axes[row, col].legend(title="Pressure", fontsize="small")

plt.suptitle(f"Thermophysical Properties & Partial Derivatives ({fluid})", fontsize=14)
plt.tight_layout()
plt.show()
