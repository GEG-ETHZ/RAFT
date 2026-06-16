# RAFT — Reservoir Analysis for TANGO

A finite element method (FEM) simulator for coupled subsurface fluid flow and
heat transport in geothermal reservoirs, specifically developed for native
coupling with TANGO. RAFT solves pressure and temperature fields over time
using a one-pass linearized extrapolation time integration scheme . Pressure
and temperature are decoupled by lagging temperature and then extrapolating
it to the current timestep.

 Physical processes include:
 - Fluid flow
     - Darcy flow
     - Pressurization
     - Buoyancy
 - Heat flow
     - Diffusion
     - Advection (Streamline Upwind Pretrov Galerkin (SUPG) stabilization)

## Running the simulation
Run a simulation:
    uv run -m raft **some-config-file.yaml**

replace the config file with a specific config file name, such as:
    uv run -m raft config_example.yaml

Run the test suite:
    uv run pytest

## Prerequisites
- Python version specified in `.python-version`
- uv (if not installed - see below)

### Installation
1. [install uv](https://docs.astral.sh/uv/getting-started/installation/) <!--NOTE: link to Wiki for uv-->
1. run `uv sync` to install dependencies. Re-run whenever dependencies in `pyproject.toml` change
1. execute a script via: `uv run some_script.py`

## Project Structure
This section should be deleted when creating a specific project from this template.
```text
└── tango-reservoir-simulator
    ├── README.md
    ├── pyproject.toml
    ├── .gitignore
    ├── .python-version
    ├── config/
    ├── outputs/
    ├── src/
    │   └── raft/
    │       ├── __init__.py
    │       ├── __main__.py
    │       ├── subsurface.py
    │       ├── preprocessing.py
    │       ├── matrixassembly.py
    │       ├── shapefunctions.py
    │       ├── parseboundaryconditions.py
    │       ├── postprocessing.py
    │       └── writevtk.py
    ├── tests/
    │   ├── __init__.py
    │   ├── test_simulation.py
    │   ├── test_checks.py
    │   ├── test_configs/
    │   └── test_demo.py
    └── uv.lock
```

**Tip:** To regenerate this tree, run:
`tree -I '__pycache__|.venv|*.egg-info|.git'`
(If tree isn't installed run: `brew install tree`)


## Developers
### Branching Strategy
Explained [here](./CONTRIBUTING.md#branching-strategy).

### Running Checks
- **Tests:** `uv run pytest`

- **Linting:** `uv run ruff check .`
- **Formatting:** `uv run ruff format .` (use `--check` to verify without modifying)
- **Type checking:** `uv run mypy .`
