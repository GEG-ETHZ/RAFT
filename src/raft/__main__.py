"""Entry point for running the simulator as a module.

Run a simulation:
    uv run -m raft config.yaml

Run the test suite:
    uv run -m raft --test
"""

import argparse
from pathlib import Path


def main():
    """Run the TANGO geothermal simulator."""
    parser = argparse.ArgumentParser(
        prog="raft",
        description="RAFT - Reservoir Analysis for TANGO",
    )
    parser.add_argument(
        "config",
        nargs="?",
        help="Name of the config file (e.g. config_1.yaml)",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Directory containing the config file (default: <repo_root>/config/)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress simulation output",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the test suite instead of a simulation",
    )

    args = parser.parse_args()

    if args.test:
        import sys

        import pytest

        sys.exit(pytest.main(["tests"]))
    elif args.config:
        from raft.subsurface import main as run_sim

        run_sim(
            config_name=args.config,
            config_dir=args.config_dir,
            verbose=not args.quiet,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
