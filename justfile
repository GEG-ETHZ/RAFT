set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# One-time setup after cloning: install deps, approve .envrc (macOS/Linux), and install git hooks
setup: _setup-direnv
    uv sync
    uv run prek install --hook-type pre-commit --hook-type pre-push --hook-type post-checkout --overwrite

[unix]
_setup-direnv:
    #!/usr/bin/env sh
    if ! command -v direnv >/dev/null 2>&1; then
        echo ""
        echo "Note: direnv not found (optional — install it for automatic environment activation on cd)."
        echo "  macOS:  brew install direnv"
        echo "  Linux:  sudo apt install direnv"
        echo "  Then add the shell hook: https://direnv.net/docs/hook.html"
        echo ""
    else
        grep -q 'direnv hook' ~/.zshrc 2>/dev/null || grep -q 'direnv hook' ~/.bashrc 2>/dev/null || {
            echo ""
            echo "Note: direnv is installed but not hooked into your shell."
            echo "Add one of the following (as appropriate) to your shell rc file and restart your shell:"
            echo "  ~/.zshrc:   eval \"\$(direnv hook zsh)\""
            echo "  ~/.bashrc:  eval \"\$(direnv hook bash)\""
            echo "  See: https://direnv.net/docs/hook.html"
            echo ""
        }
        direnv allow . || {
            echo ""
            echo "Warning: 'direnv allow .' failed — run it manually after checking your .envrc."
            echo ""
        }
    fi

[windows]
_setup-direnv:
    @echo "Note: direnv is not available on Windows — skipping."

# Format Python code with ruff
fmt:
    uv run ruff format .

# Lint Python code with ruff
lint:
    uv run ruff check .

# Type-check with mypy
typecheck:
    uv run mypy .

# Run test suite
test:
    uv run pytest -v

# Run all quality checks in CI order: format → lint → typecheck → test
check: fmt lint typecheck test

# The gitleaks hook only checks each commit's staged changes; run this before a repository goes public.
# prek builds the gitleaks version pinned in .pre-commit-config.yaml, so nothing else needs installing.
# Scan the whole git history (every branch and tag) for secrets
leaks:
    uv run prek run --hook-stage manual gitleaks-history

# Generate config/<mesh>.geo and .msh from a RAFT config, e.g. `just geo config_example_2D -o 2D_generated`
geo config *args:
    uv run --with gmsh scripts/raft_to_geo.py {{config}} --mesh {{args}}

# Regenerate the verification.md comparison figures into figures/verification
verification-figures:
    uv run python scripts/verification_figures.py

# Build HTML documentation from README.md, theory.md and verification.md into docs/_build/html
docs:
    uv run --group docs sphinx-build -b html docs/source docs/_build/html
