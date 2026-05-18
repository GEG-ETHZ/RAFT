"""Test subsurface simulator for TANGO."""

import sys
from pathlib import Path

import test_checks
from subsurface import main


def _test_data_dir() -> Path:
    """Provides the absolute path to the test data directory."""
    return Path(__file__).parent / "test_configs"


def _test_pass_fail(test_name, test_result):
    if test_result:
        print("\nTest: " + test_name + " - PASS")
    else:
        print("\nTest: " + test_name + " - FAIL")
        sys.exit()


def _run_tests():
    case_dir = _test_data_dir()
    # Test 5: Check advection against analytical solutions
    test_name = "Buoyancy Test"
    config_filename = "config_test_buoyancy.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbose=False)

    test_pass = test_checks.check_buoyancy(config_filename, case_dir, primary_variables, mesh, test_name)
    _test_pass_fail(test_name, test_pass)

    # Test 4: Check advection against analytical solutions
    test_name = "Advection Test"
    config_filename = "config_test_advection.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbose=False)

    test_pass = test_checks.check_advection(config_filename, case_dir, primary_variables, mesh, test_name)
    _test_pass_fail(test_name, test_pass)

    # Test 3: Check exact linear approximation in Y
    test_name = "Patch Test - Y"
    config_filename = "config_test_patchY.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbose=False)

    test_pass = test_checks.check_patch_test(config_filename, case_dir, primary_variables, mesh, test_name)
    _test_pass_fail(test_name, test_pass)

    # Test 2: Check exact linear approximation in X
    test_name = "Patch Test - X"
    config_filename = "config_test_patchX.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbose=False)

    test_pass = test_checks.check_patch_test(config_filename, case_dir, primary_variables, mesh, test_name)
    _test_pass_fail(test_name, test_pass)

    # Test 1: Check exact linear approximation in Z
    test_name = "Patch Test - Z"
    config_filename = "config_test_patchZ.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir, verbose=False)

    test_pass = test_checks.check_patch_test(config_filename, case_dir, primary_variables, mesh, test_name)
    _test_pass_fail(test_name, test_pass)


if __name__ == "__main__":
    _run_tests()
