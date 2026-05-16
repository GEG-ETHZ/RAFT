"""Test subsurface simulator for TANGO."""

from pathlib import Path

from subsurface import main


def test_data_dir() -> Path:
    """Provides the absolute path to the test data directory."""
    return Path(__file__).parent / "test_configs"


def _run_tests():
    case_dir = test_data_dir()

    # Test 1: Check exact linear approximation in Z43
    config_filename = "config_test1.yaml"
    primary_variables, mesh = main(config_name=config_filename, config_dir=case_dir)

    print(primary_variables["p"])
    print(primary_variables["T"])


if __name__ == "__main__":
    _run_tests()
