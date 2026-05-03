"""Subsurface Simulator."""

import yaml


def load_config(path):
    """Load config file."""
    with open(path) as f:
        config = yaml.safe_load(f)

    # Derived properties (computed here, not stored in YAML)
    mat = config["material"]
    phi = mat["porosity"]
    biot = mat["biot_coeff"]
    Kf = mat["fluid"]["bulk_mod"]
    Ks = mat["solid"]["bulk_mod"]

    # storage modulus
    M = Kf * Ks / (phi * Ks + (biot - phi) * Kf)
    # effective expansion modulus
    a_e = biot * mat["solid"]["therm_expansion"] + phi * (
        mat["fluid"]["therm_expansion"] - mat["solid"]["therm_expansion"]
    )
    # effective conductivity
    lambda_e = phi * mat["fluid"]["therm_cond"] + (1 - phi) * mat["solid"]["therm_cond"]
    # effective heat capacity
    cp_e = (
        phi * mat["fluid"]["specific_heat"] * mat["fluid"]["density"]
        + (1 - phi) * mat["solid"]["specific_heat"] * mat["solid"]["density"]
    )

    config["material"]["storage_mod"] = M
    config["material"]["therm_cond_eff"] = lambda_e
    config["material"]["therm_expansion_eff"] = a_e
    config["material"]["volumetric_heat_capacity_eff"] = cp_e

    return config


def main():
    """Main simulation run."""
    ## Config file
    config_name = "config_1.yaml"
    config = load_config(config_name)  # noqa: F841
    # ... rest of simulation

    ## Material Properties

    ## Boundary Conditions

    ## Initial Conditions

    ## Matrix Assembly

    ## Force Vector Assembly

    ## Matrix Inversion

    ## Solve


if __name__ == "__main__":
    main()
