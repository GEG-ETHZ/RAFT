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
    uv run -m raft config.yaml

Run the test suite:
    uv run -m raft --test

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
    ├── config/                         # YAML configuration files
    ├── outputs/                        # VTK output files (auto-created)
    ├── src/
    │   └── raft/
    │       ├── __init__.py
    │       ├── __main__.py             # Entry point for `python -m` or Spyder
    │       ├── subsurface.py           # Main simulation loop
    │       ├── preprocessing.py        # Config and mesh loading
    │       ├── matrixassembly.py       # FEM matrix integration
    │       ├── shapefunctions.py       # Shape functions and quadrature
    │       ├── parseboundaryconditions.py  # BC parsing
    │       ├── postprocessing.py       # Flux computation and VTK export
    │       ├── writevtk.py             # VTK file writer
    │       ├── test_simulation.py      # Integration test runner
    │       └── test_checks.py          # Analytical solution checks

    └── uv.lock
```

**Tip:** To regenerate this tree, run:
`tree -I '__pycache__|.venv|*.egg-info|.git'`
(If tree isn't installed run: `brew install tree`)


## Developers
### Branching Strategy
Explained [here](./CONTRIBUTING.md#branching-strategy).

### Running Checks
- **Tests:** `uv run run-tests`

- **Linting:** `uv run ruff check .`
- **Formatting:** `uv run ruff format .` (use `--check` to verify without modifying)
- **Type checking:** `uv run mypy src/`
