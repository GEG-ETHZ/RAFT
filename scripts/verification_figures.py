# SPDX-FileCopyrightText: 2026 ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Generate the verification.md figures from the analytical test checks.

Run with `just verification-figures`.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # for tests.test_checks

import tests.test_checks as test_checks  # noqa: E402

from geg_raft.subsurface import main  # noqa: E402

CASE_DIR = ROOT / "tests" / "test_configs"
OUT_DIR = ROOT / "figures" / "verification"

# (figure prefix, config, check function, test name, suffix per figure)
CASES = [
    ("heterogeneous", "config_test_heterogeneous.yaml", test_checks.check_heterogeneous,
     "Heterogeneous Test", ["T", "p"]),
    ("axisymmetry", "config_test_axisymmetry.yaml", test_checks.check_axisymmetry,
     "Axisymmetric", ["T", "p"]),
    ("advection", "config_test_advection.yaml", test_checks.check_advection,
     "Advection Test", ["T"]),
    ("nl_viscosity", "config_test_NL_viscosity.yaml", test_checks.check_NL_viscosity,
     "Non-linear Viscosity", ["T", "mu", "p"]),
    ("buoyancy", "config_test_buoyancy.yaml", test_checks.check_buoyancy,
     "Buoyancy Test", ["p"]),
    ("well_convection", "config_test_wellconvection.yaml", test_checks.check_well_convection,
     "Wellbore heat conduction", ["T"]),
]  # fmt: skip


def main_figures() -> None:
    """Run each case with plotting on and save its figures."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for prefix, config, check, test_name, suffixes in CASES:
        plt.close("all")
        results, mesh = main(config_name=config, config_dir=CASE_DIR, verbosity="quiet")
        check(config, CASE_DIR, results, mesh, test_name, True)
        for num, suffix in zip(plt.get_fignums(), suffixes, strict=True):
            fig = plt.figure(num)
            fig.set_size_inches(5.0, 3.75)
            fig.tight_layout()
            fig.savefig(OUT_DIR / f"{prefix}_{suffix}.png", dpi=150)


if __name__ == "__main__":
    main_figures()
