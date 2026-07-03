"""Script to run the simulation or test suite within the Spyder IDE."""

# from raft.subsurface import main

# dofs, mesh = main("config_example.yaml")

import sys
from pathlib import Path

import pytest

# Add project root and src directory to sys.path so raft can be imported
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir / "src"))
sys.path.append(str(root_dir))
# Run the test suite
pytest.main(["-v", str(root_dir / "tests")])
