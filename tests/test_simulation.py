"""Test subsurface simulator for TANGO."""

import sys
from pathlib import Path

import pytest

import tests.test_checks as test_checks
from raft.subsurface import main


def _test_data_dir() -> Path:
    """Provides the absolute path to the test data directory."""
    return Path(__file__).parent / "test_configs"


def test_pressurization():
    """Test pressurization against analytical solutions."""
    case_dir = _test_data_dir()
    test_name = "Pressurization Test"
    config_filename = "config_test_pressurization.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbose=False)

    test_pass = test_checks.check_pressurization(
        config_filename, case_dir, primary_variables, mesh, test_name
    )
    assert test_pass


def test_buoyancy():
    """Test buoyancy against analytical solutions."""
    case_dir = _test_data_dir()
    test_name = "Buoyancy Test"
    config_filename = "config_test_buoyancy.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbose=False)

    test_pass = test_checks.check_buoyancy(config_filename, case_dir, primary_variables, mesh, test_name)
    assert test_pass


def test_advection():
    """Test advection against analytical solutions."""
    case_dir = _test_data_dir()
    test_name = "Advection Test"
    config_filename = "config_test_advection.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbose=False)

    test_pass = test_checks.check_advection(config_filename, case_dir, primary_variables, mesh, test_name)
    assert test_pass


def test_patch_y():
    """Check exact linear approximation in Y."""
    case_dir = _test_data_dir()
    test_name = "Patch Test - Y"
    config_filename = "config_test_patchY.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbose=False)

    test_pass = test_checks.check_patch_test(config_filename, case_dir, primary_variables, mesh, test_name)
    assert test_pass


def test_patch_x():
    """Check exact linear approximation in X."""
    case_dir = _test_data_dir()
    test_name = "Patch Test - X"
    config_filename = "config_test_patchX.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbose=False)

    test_pass = test_checks.check_patch_test(config_filename, case_dir, primary_variables, mesh, test_name)
    assert test_pass


def test_patch_z():
    """Check exact linear approximation in Z."""
    case_dir = _test_data_dir()
    test_name = "Patch Test - Z"
    config_filename = "config_test_patchZ.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbose=False)

    test_pass = test_checks.check_patch_test(config_filename, case_dir, primary_variables, mesh, test_name)
    assert test_pass


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
