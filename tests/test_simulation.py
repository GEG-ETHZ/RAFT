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

"""Test subsurface simulator for TANGO."""

import sys
from pathlib import Path

import pytest

import tests.test_checks as test_checks
from geg_raft.preprocessing import preprocessing
from geg_raft.subsurface import main


def _test_data_dir() -> Path:
    """Provides the absolute path to the test data directory."""
    return Path(__file__).parent / "test_configs"


def test_well_conduction() -> None:
    """Test well heat exchange against analytical solution"""
    case_dir = _test_data_dir()
    test_name = "Wellbore heat conduction"
    config_filename = "config_test_wellconvection.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    plot_on = False
    test_pass = test_checks.check_well_convection(
        config_filename, case_dir, primary_variables, mesh, test_name, plot_on
    )
    assert test_pass


def test_fracture_advection_bounded() -> None:
    """Check p/T stay within the boundary-value envelope for a fracture angled into the
    pressure/temperature gradient, with advective leak-off turned on."""
    case_dir = _test_data_dir()
    test_name = "Fracture Advection - Boundedness"
    config_filename = "config_test_fracture_advection_bounded.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    plot_on = False
    test_pass = test_checks.check_fracture_advection_bounded(
        config_filename, case_dir, primary_variables, mesh, test_name, plot_on
    )
    assert test_pass


def test_enthalpy_bndry_pressure() -> None:
    """Check the advective enthalpy boundary vector/tangent at a pressure-type outlet with
    no thermal BC of its own (also exercises the monolithic Kpp/KpT cross-term)."""
    case_dir = _test_data_dir()
    test_name = "Enthalpy Boundary Flux - Pressure Outlet"
    config_filename = "config_test_enthalpy_bndry_pressure.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    plot_on = False
    test_pass = test_checks.check_enthalpy_bndry_bounded(
        config_filename, case_dir, primary_variables, mesh, test_name, plot_on
    )
    assert test_pass


def test_enthalpy_bndry_flux() -> None:
    """Same as test_enthalpy_bndry_pressure, but with a flux-type outlet, which should not
    pick up the monolithic Kpp/KpT cross-term (a flux-type node's q is a prescribed value,
    never a function of p/T)."""
    case_dir = _test_data_dir()
    test_name = "Enthalpy Boundary Flux - Flux Outlet"
    config_filename = "config_test_enthalpy_bndry_flux.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    plot_on = False
    test_pass = test_checks.check_enthalpy_bndry_bounded(
        config_filename, case_dir, primary_variables, mesh, test_name, plot_on
    )
    assert test_pass


def test_patch_anisotropy() -> None:
    """Check exact linear approximation in anisotropy."""
    case_dir = _test_data_dir()
    test_name = "Patch Test - Anisotropy"
    config_filename = "config_test_anisotropic.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    dimension = 4  # not aligned with cartesian grid, use d =/= 0,1,2
    plot_on = False
    test_pass = test_checks.check_patch_test(
        config_filename, case_dir, primary_variables, mesh, dimension, test_name, plot_on
    )
    assert test_pass


def test_patch_anisotropy_conductivity() -> None:
    """Check exact linear approximation with anisotropic thermal conductivity."""
    case_dir = _test_data_dir()
    test_name = "Patch Test - Anisotropic Conductivity"
    config_filename = "config_test_anisotropic_cond.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    dimension = 4  # not aligned with cartesian grid, use d =/= 0,1,2
    plot_on = False
    test_pass = test_checks.check_patch_test(
        config_filename, case_dir, primary_variables, mesh, dimension, test_name, plot_on
    )
    assert test_pass


def test_buoyancy() -> None:
    """Test buoyancy against analytical solutions."""
    case_dir = _test_data_dir()
    test_name = "Buoyancy Test"
    config_filename = "config_test_buoyancy.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    plot_on = False
    test_pass = test_checks.check_buoyancy(
        config_filename, case_dir, primary_variables, mesh, test_name, plot_on
    )
    assert test_pass


def test_advection() -> None:
    """Test advection against analytical solutions."""
    case_dir = _test_data_dir()
    test_name = "Advection Test"
    config_filename = "config_test_advection.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    plot_on = False
    test_pass = test_checks.check_advection(
        config_filename, case_dir, primary_variables, mesh, test_name, plot_on
    )
    assert test_pass


def test_heterogeneous() -> None:
    """Verify solution of heterogeneous properties against analytical solution."""
    case_dir = _test_data_dir()
    test_name = "Heterogeneous Test"
    config_filename = "config_test_heterogeneous.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    plot_on = False
    test_pass = test_checks.check_heterogeneous(
        config_filename, case_dir, primary_variables, mesh, test_name, plot_on
    )
    assert test_pass


def test_nonlinear_viscosity() -> None:
    """Check exactness of non-linear viscosity"""
    case_dir = _test_data_dir()
    test_name = "Non-linear Viscosity"
    config_filename = "config_test_NL_viscosity.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="full")

    plot_on = False
    test_pass = test_checks.check_NL_viscosity(
        config_filename, case_dir, primary_variables, mesh, test_name, plot_on
    )
    assert test_pass


def test_axisymmetry() -> None:
    """Check axisymmetry against exact solution."""
    case_dir = _test_data_dir()
    test_name = "Axisymmetric"
    config_filename = "config_test_axisymmetry.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    plot_on = False
    test_pass = test_checks.check_axisymmetry(
        config_filename, case_dir, primary_variables, mesh, test_name, plot_on
    )
    assert test_pass


def test_patch_x_flux_2D() -> None:
    """Check exact linear approximation in X in 2D with Q4 elements."""
    case_dir = _test_data_dir()
    test_name = "Patch Test - X - flux - 2D Q4"
    config_filename = "config_test_patchX2D.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    dimension = 0
    plot_on = False
    test_pass = test_checks.check_patch_test_flux(
        config_filename, case_dir, primary_variables, mesh, dimension, test_name, plot_on
    )
    assert test_pass


def test_patch_y_flux_2D() -> None:
    """Check exact linear approximation in Y in 2D with T3 elements."""
    case_dir = _test_data_dir()
    test_name = "Patch Test - Y - flux - 2D T3"
    config_filename = "config_test_patchY2D.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    dimension = 1
    plot_on = False
    test_pass = test_checks.check_patch_test_flux(
        config_filename, case_dir, primary_variables, mesh, dimension, test_name, plot_on
    )
    assert test_pass


def test_patch_z_flux() -> None:
    """Check exact linear approximation in Z."""
    case_dir = _test_data_dir()
    test_name = "Patch Test - Z - flux"
    config_filename = "config_test_patchZ_flux.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="full")

    dimension = 2
    plot_on = False
    test_pass = test_checks.check_patch_test_flux(
        config_filename, case_dir, primary_variables, mesh, dimension, test_name, plot_on
    )
    assert test_pass


def test_patch_y_flux() -> None:
    """Check exact linear approximation in Y."""
    case_dir = _test_data_dir()
    test_name = "Patch Test - Y - flux"
    config_filename = "config_test_patchY_flux.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    dimension = 1
    plot_on = False
    test_pass = test_checks.check_patch_test_flux(
        config_filename, case_dir, primary_variables, mesh, dimension, test_name, plot_on
    )
    assert test_pass


def test_patch_x_flux() -> None:
    """Check exact linear approximation in X."""
    case_dir = _test_data_dir()
    test_name = "Patch Test - X - flux"
    config_filename = "config_test_patchX_flux.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    dimension = 0
    plot_on = False
    test_pass = test_checks.check_patch_test_flux(
        config_filename, case_dir, primary_variables, mesh, dimension, test_name, plot_on
    )
    assert test_pass


def test_patch_z_T4() -> None:
    """Check exact linear approximation in Z."""
    case_dir = _test_data_dir()
    test_name = "Patch Test - Z - T4"
    config_filename = "config_test_patchZ_T4.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    dimension = 2
    plot_on = False
    test_pass = test_checks.check_patch_test(
        config_filename, case_dir, primary_variables, mesh, dimension, test_name, plot_on
    )
    assert test_pass


def test_patch_y_T4() -> None:
    """Check exact linear approximation in Y."""
    case_dir = _test_data_dir()
    test_name = "Patch Test - Y - T4"
    config_filename = "config_test_patchY_T4.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    dimension = 1
    plot_on = False
    test_pass = test_checks.check_patch_test(
        config_filename, case_dir, primary_variables, mesh, dimension, test_name, plot_on
    )
    assert test_pass


def test_patch_x_T4() -> None:
    """Check exact linear approximation in X."""
    case_dir = _test_data_dir()
    test_name = "Patch Test - X - T4"
    config_filename = "config_test_patchX_T4.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    dimension = 0
    plot_on = False
    test_pass = test_checks.check_patch_test(
        config_filename, case_dir, primary_variables, mesh, dimension, test_name, plot_on
    )
    assert test_pass


def test_regions() -> None:
    """Verify assignment of regions against a known list."""
    case_dir = _test_data_dir()
    # test_name = "Regions Test"
    config_filename = "config_test_regions.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    test_pass = test_checks.check_regions(case_dir, mesh)
    assert test_pass


def test_patch_y() -> None:
    """Check exact linear approximation in Y."""
    case_dir = _test_data_dir()
    test_name = "Patch Test - Y"
    config_filename = "config_test_patchY.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    dimension = 1
    plot_on = False
    test_pass = test_checks.check_patch_test(
        config_filename, case_dir, primary_variables, mesh, dimension, test_name, plot_on
    )
    assert test_pass


def test_patch_x() -> None:
    """Check exact linear approximation in X."""
    case_dir = _test_data_dir()
    test_name = "Patch Test - X"
    config_filename = "config_test_patchX.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    dimension = 0
    plot_on = False
    test_pass = test_checks.check_patch_test(
        config_filename, case_dir, primary_variables, mesh, dimension, test_name, plot_on
    )
    assert test_pass


def test_patch_z() -> None:
    """Check exact linear approximation in Z."""
    case_dir = _test_data_dir()
    test_name = "Patch Test - Z"
    config_filename = "config_test_patchZ.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    dimension = 2
    plot_on = False
    test_pass = test_checks.check_patch_test(
        config_filename, case_dir, primary_variables, mesh, dimension, test_name, plot_on
    )
    assert test_pass


def test_line_bc_patch_flux_2D() -> None:
    """Check exact linear approximation when x_min/x_max are specified as line BCs instead of faces."""
    case_dir = _test_data_dir()
    test_name = "Patch Test - Line BC - flux - 2D Q4"
    config_filename = "config_test_line_bc.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    dimension = 0
    plot_on = False
    test_pass = test_checks.check_patch_test_flux(
        config_filename, case_dir, primary_variables, mesh, dimension, test_name, plot_on
    )
    assert test_pass


def test_fracture_single() -> None:
    """Verify mesh-cutting correctness for a single, fully-interior 2D fracture."""
    case_dir = _test_data_dir()
    config_filename = "config_test_fracture_single.yaml"
    _, mesh = preprocessing(config_filename, case_dir, verbose=False)

    test_pass = test_checks.check_fracture_single(config_filename, case_dir, mesh)
    assert test_pass


def test_fracture_perpendicular_patch() -> None:
    """Patch test: a fracture perpendicular to the gradient should recover the exact linear solution."""
    case_dir = _test_data_dir()
    test_name = "Fracture Patch Test - Perpendicular"
    config_filename = "config_test_fracture_perpendicular.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    plot_on = False
    test_pass = test_checks.check_fracture_perpendicular_patch(
        config_filename, case_dir, primary_variables, mesh, test_name, plot_on
    )
    assert test_pass


def test_patch_elastic_2D_T3() -> None:
    """Check exact linear displacement and constant stress in 2D with T3 elements."""
    case_dir = _test_data_dir()
    test_name = "Elastic Patch Test - 2D T3"
    config_filename = "config_test_patch_elastic_2D_T3.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    plot_on = False
    test_pass = test_checks.check_patch_test_elastic(
        config_filename, case_dir, primary_variables, mesh, test_name, plot_on
    )
    assert test_pass


def test_patch_elastic_2D_Q4() -> None:
    """Check exact linear displacement and constant stress in 2D with Q4 elements."""
    case_dir = _test_data_dir()
    test_name = "Elastic Patch Test - 2D Q4"
    config_filename = "config_test_patch_elastic_2D_Q4.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    plot_on = False
    test_pass = test_checks.check_patch_test_elastic(
        config_filename, case_dir, primary_variables, mesh, test_name, plot_on
    )
    assert test_pass


def test_patch_elastic_T4() -> None:
    """Check exact linear displacement and constant stress with T4 elements."""
    case_dir = _test_data_dir()
    test_name = "Elastic Patch Test - T4"
    config_filename = "config_test_patch_elastic_T4.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    plot_on = False
    test_pass = test_checks.check_patch_test_elastic(
        config_filename, case_dir, primary_variables, mesh, test_name, plot_on
    )
    assert test_pass


def test_patch_elastic() -> None:
    """Check exact linear displacement and constant stress with B8 elements."""
    case_dir = _test_data_dir()
    test_name = "Elastic Patch Test - B8"
    config_filename = "config_test_patch_elastic_B8.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    plot_on = False
    test_pass = test_checks.check_patch_test_elastic(
        config_filename, case_dir, primary_variables, mesh, test_name, plot_on
    )
    assert test_pass


def test_lame_cylinder() -> None:
    """Check the axisymmetric thick-walled cylinder against the Lame solution."""
    case_dir = _test_data_dir()
    test_name = "Thick-Walled Cylinder (Lame)"
    config_filename = "config_test_lame_cylinder.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    plot_on = False
    test_pass = test_checks.check_lame_cylinder(
        config_filename, case_dir, primary_variables, mesh, test_name, plot_on
    )
    assert test_pass


def test_terzaghi() -> None:
    """Check one-dimensional consolidation in 2D with Q4 elements, split solver."""
    case_dir = _test_data_dir()
    test_name = "Terzaghi Consolidation"
    config_filename = "config_test_terzaghi.yaml"
    times = test_checks.TERZAGHI_TIMES
    results, mesh = test_checks.run_at_times(config_filename, case_dir, times)

    dimension = 1
    plot_on = False
    test_pass = test_checks.check_terzaghi(
        config_filename, case_dir, results, mesh, times, dimension, test_name, plot_on
    )
    assert test_pass


def test_terzaghi_3D() -> None:
    """Check one-dimensional consolidation of a 3D column with B8 elements."""
    case_dir = _test_data_dir()
    test_name = "Terzaghi Consolidation - 3D"
    config_filename = "config_test_terzaghi_3D.yaml"
    times = test_checks.TERZAGHI_TIMES
    results, mesh = test_checks.run_at_times(config_filename, case_dir, times)

    dimension = 2
    plot_on = False
    test_pass = test_checks.check_terzaghi(
        config_filename, case_dir, results, mesh, times, dimension, test_name, plot_on
    )
    assert test_pass


def test_mandel() -> None:
    """Check Mandel's problem with the monolithic solver: Mandel-Cryer effect and no oscillation."""
    case_dir = _test_data_dir()
    test_name = "Mandel's Problem"
    config_filename = "config_test_mandel.yaml"
    times = test_checks.MANDEL_TIMES
    results, mesh = test_checks.run_at_times(config_filename, case_dir, times)

    plot_on = False
    test_pass = test_checks.check_mandel(config_filename, case_dir, results, mesh, times, test_name, plot_on)
    assert test_pass


@pytest.mark.parametrize(
    ("config_filename", "times"),
    [
        ("config_test_thermoelastic.yaml", test_checks.THERMAL_CONSOLIDATION_TIMES),
        ("config_test_thermoporoelastic.yaml", test_checks.THERMAL_CONSOLIDATION_TIMES),
        # the extrapolated temperature lags the early thermal pressurization, so start later
        ("config_test_thermoporoelastic_split.yaml", test_checks.THERMAL_CONSOLIDATION_SPLIT_TIMES),
    ],
)
def test_thermal_consolidation(config_filename: str, times: list[float]) -> None:
    """Check thermal stress and pressurization in a column, without and with flow, under both solvers."""
    case_dir = _test_data_dir()
    test_name = "Thermo-poroelastic Consolidation"
    results, mesh = test_checks.run_at_times(config_filename, case_dir, times)

    plot_on = False
    test_pass = test_checks.check_thermal_consolidation(
        config_filename, case_dir, results, mesh, times, test_name, plot_on
    )
    assert test_pass


def test_sneddon() -> None:
    """Check the opening of a pressurized fracture under in-situ compression against Sneddon."""
    case_dir = _test_data_dir()
    test_name = "Sneddon Pressurized Fracture"
    config_filename = "config_test_sneddon.yaml"
    results, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    plot_on = False
    test_pass = test_checks.check_sneddon(config_filename, case_dir, results, mesh, test_name, plot_on)
    assert test_pass


def test_fracture_closure() -> None:
    """Check penalty contact on two crossing fractures closed by the in-situ stress, after a time step."""
    case_dir = _test_data_dir()
    test_name = "Fracture Closure"
    config_filename = "config_test_fracture_closure.yaml"
    results, mesh = main(config_name=config_filename, config_dir=case_dir, verbosity="quiet")

    plot_on = False
    test_pass = test_checks.check_fracture_closure(
        config_filename, case_dir, results, mesh, test_name, plot_on
    )
    assert test_pass


def test_cohesive_opening(tmp_path: Path) -> None:
    """Check a cemented fracture opened, unloaded, and separated, and the same fracture debonded."""
    case_dir = _test_data_dir()
    test_name = "Cohesive Opening"
    config_filename = test_checks.COHESIVE_CONFIG
    results, mesh = test_checks.run_at_times(config_filename, case_dir, test_checks.COHESIVE_TIMES)

    plot_on = False
    test_pass = test_checks.check_cohesive_opening(
        config_filename, case_dir, results, mesh, test_name, plot_on
    )
    assert test_pass

    # initial damage 1: the same opening without traction
    config_filename = test_checks.write_cohesive_debonded_config(case_dir, tmp_path)
    results, mesh = test_checks.run_at_times(config_filename, tmp_path, [2.5, 5.0])
    assert test_checks.check_cohesive_debonded(config_filename, tmp_path, results, mesh)


def test_cohesive_shear(tmp_path: Path) -> None:
    """Check a cemented fracture slid, unloaded, and debonded by a rigid upper block."""
    test_name = "Cohesive Shear"
    config_filename = test_checks.write_cohesive_shear_config(_test_data_dir(), tmp_path)
    slip, force = test_checks.run_cohesive_shear(config_filename, tmp_path, test_checks.COHESIVE_TIMES)

    plot_on = False
    test_pass = test_checks.check_cohesive_shear(config_filename, tmp_path, slip, force, test_name, plot_on)
    assert test_pass


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
