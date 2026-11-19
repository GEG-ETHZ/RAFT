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

"""Generate the verification.md figures from the analytical test checks.

Run with `just verification-figures`.
"""

import sys
import tempfile
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
    ("lame_cylinder", "config_test_lame_cylinder.yaml", test_checks.check_lame_cylinder,
     "Thick-Walled Cylinder (Lame)", ["u"]),
]  # fmt: skip


def _save_open_figures(prefix: str, suffixes: list[str]) -> None:
    """Save the open figures, one per suffix."""
    for num, suffix in zip(plt.get_fignums(), suffixes, strict=True):
        fig = plt.figure(num)
        fig.set_size_inches(5.0, 3.75)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{prefix}_{suffix}.png", dpi=150)


def _consolidation_figures() -> None:
    """Consolidation cases, compared at several times."""
    plt.close("all")
    config = "config_test_terzaghi.yaml"
    results, mesh = test_checks.run_at_times(config, CASE_DIR, test_checks.TERZAGHI_TIMES)
    test_checks.check_terzaghi(
        config, CASE_DIR, results, mesh, test_checks.TERZAGHI_TIMES, 1, "Terzaghi Consolidation", True
    )
    _save_open_figures("terzaghi", ["p", "u"])

    plt.close("all")
    config = "config_test_mandel.yaml"
    results, mesh = test_checks.run_at_times(config, CASE_DIR, test_checks.MANDEL_TIMES)
    test_checks.check_mandel(
        config, CASE_DIR, results, mesh, test_checks.MANDEL_TIMES, "Mandel's Problem", True
    )
    _save_open_figures("mandel", ["p"])

    plt.close("all")
    config = "config_test_thermoporoelastic.yaml"
    times = test_checks.THERMAL_CONSOLIDATION_TIMES
    results, mesh = test_checks.run_at_times(config, CASE_DIR, times)
    test_checks.check_thermal_consolidation(
        config, CASE_DIR, results, mesh, times, "Thermo-poroelastic Consolidation", True
    )
    _save_open_figures("thermoporoelastic", ["T", "p", "u"])

    plt.close("all")
    config = "config_test_elastic_wave.yaml"
    times = test_checks.ELASTIC_WAVE_TIMES
    results, mesh = test_checks.run_at_times(config, CASE_DIR, times)
    test_checks.check_elastic_wave(config, CASE_DIR, results, mesh, times, "Elastic Wave", True)
    _save_open_figures("elastic", ["wave"])


def _fracture_contact_figures() -> None:
    """Fracture face loading and contact after one time step."""
    for prefix, config, check, test_name in (
        ("sneddon", "config_test_sneddon.yaml", test_checks.check_sneddon, "Sneddon Pressurized Fracture"),
        (
            "fracture_closure",
            "config_test_fracture_closure.yaml",
            test_checks.check_fracture_closure,
            "Fracture Closure",
        ),
    ):
        plt.close("all")
        results, mesh = main(config_name=config, config_dir=CASE_DIR, verbosity="quiet")
        check(config, CASE_DIR, results, mesh, test_name, True)
        _save_open_figures(prefix, ["opening" if prefix == "sneddon" else "contact"])


def _cohesive_figures() -> None:
    """Cohesive opening and shear traction paths through loading, unloading, and separation."""
    plt.close("all")
    config = test_checks.COHESIVE_CONFIG
    results, mesh = test_checks.run_at_times(config, CASE_DIR, test_checks.COHESIVE_TIMES)
    test_checks.check_cohesive_opening(config, CASE_DIR, results, mesh, "Cohesive Opening", True)
    _save_open_figures("cohesive", ["opening"])

    plt.close("all")
    with tempfile.TemporaryDirectory() as tmp:
        shear_dir = Path(tmp)
        config = test_checks.write_cohesive_shear_config(CASE_DIR, shear_dir)
        slip, force = test_checks.run_cohesive_shear(config, shear_dir, test_checks.COHESIVE_TIMES)
        test_checks.check_cohesive_shear(config, shear_dir, slip, force, "Cohesive Shear", True)
    _save_open_figures("cohesive", ["shear"])


def main_figures() -> None:
    """Run each case with plotting on and save its figures."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for prefix, config, check, test_name, suffixes in CASES:
        plt.close("all")
        results, mesh = main(config_name=config, config_dir=CASE_DIR, verbosity="quiet")
        check(config, CASE_DIR, results, mesh, test_name, True)
        _save_open_figures(prefix, suffixes)
    _consolidation_figures()
    _fracture_contact_figures()
    _cohesive_figures()


if __name__ == "__main__":
    main_figures()
