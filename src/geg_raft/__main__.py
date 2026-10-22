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

"""Entry point for running the simulator as a module.

Run a simulation:
    uv run -m geg_raft some-config-name

Run the test suite:
    uv run -m geg_raft --test
"""

import argparse
from pathlib import Path


def main() -> None:
    """Run the TANGO geothermal simulator."""
    parser = argparse.ArgumentParser(
        prog="geg_raft",
        description="RAFT - Reservoir Analysis for TANGO",
    )
    parser.add_argument(
        "config",
        nargs="?",
        help="Name of the config file in CONFIG_DIR, with or without .yaml (e.g. config_example)",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Directory containing the config file (default: <repo_root>/config/)",
    )
    parser.add_argument(
        "--verbosity",
        choices=["quiet", "lite", "full"],
        default="full",
        help="Console output: none, one line per timestep attempt, or full detail (default: full)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress simulation output (same as --verbosity quiet)",
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
        from geg_raft.subsurface import main as run_sim

        run_sim(
            config_name=args.config,
            config_dir=args.config_dir,
            verbosity="quiet" if args.quiet else args.verbosity,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
