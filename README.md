# TANGO Reservoir Simulator

THM simulator for native coupling with TANGO


## Prerequisites
- Python version specified in `.python-version`
- uv (if not installed - see below)

### Installation
1. [install uv](https://docs.astral.sh/uv/getting-started/installation/) <!--NOTE: link to Wiki for uv-->
1. run `uv sync` to install dependencies. Re-run whenever dependencies in `pyproject.toml` change
1. execute a script via: `uv run some_script.py`

## Repository Setup
**Note**: this section can be deleted after initial setup.

If you just generated this project from the template, you need to connect it to a remote repository:
1. **Create the remote repository:** Create a new project in
   [GitLab](https://gitlab.ethz.ch/geg) (unless it is already known that the
   project will be released publicly via [GitHub](https://github.com/GEG-ETHZ/)).
   - *Note: If you run into permissions issues creating the repository, please
      contact the Software Team (e.g., Chad Baker, Anton Kulik).*
1. **Link your local repository:** Follow the [Syncing with Multiple
   Repos](https://gitlab.ethz.ch/geg/computing-guide/-/wikis/git#syncing-with-multi
   ple-repos) guide to set up your git remotes.
   - *Example:* `git remote add origin git@gitlab.ethz.ch:geg/your-project.git`
   - *Then push your initial commit:* `git push -u --no-verify origin main`
      -- you will need to use `--no-verify` for the first push because `main` is
      protected.  After that, **DO NOT** use `--no-verify`.
1. **Set up a GitLab Runner:** If you have not previously done so,
   [set up a GitLab runner](https://gitlab.ethz.ch/geg/computing-guide/-/wikis/git#setting-up-a-gitlab-runner-on-user-laptop)

## Project Structure
This section should be deleted when creating a specific project from this template.
```text
└── tango-reservoir-simulator
    ├── README.md
    ├── pyproject.toml
    ├── .gitignore
    ├── .python-version
    ├── data/        # Raw and processed data files (ignored by git by default)
    ├── outputs/     # Generated figures, reports, and models
    ├── scripts/     # Standalone Python scripts for data processing and analysis
    ├── src/         # Installable Python package source code and project-specific utilities
    │   └── tango-reservoir-simulator
    │       └── __init__.py
    ├── tests/       # Unit and integration tests
    │   ├── __init__.py
    │   └── test_demo.py
    └── uv.lock
```

> **Note on Repository Type:** You generated a `standard` repository.
> This layout contains both a standardized data analysis environment (`data`, `outputs`, `scripts`, `tests`) and an installable Python library in `src/`.
> If you find you do not need the data science folders, you may safely delete them.

**Tip:** To regenerate this tree, run:
`tree -I '__pycache__|.venv|*.egg-info|.git'`
(If tree isn't installed run: `brew install tree`)


## Developers
### Branching Strategy
Explained [here](./CONTRIBUTING.md#branching-strategy).

### Running Checks
- **Tests:** `uv run pytest -v`
- **Linting:** `uv run ruff check .`
- **Formatting:** `uv run ruff format .` (use `--check` to verify without modifying)
- **Type checking:** `uv run mypy src/`
