# Changelog

All notable changes to RAFT are documented here.
RAFT does not yet follow [Semantic Versioning](https://semver.org/) — everything is
pre-release. Starting with `1.0.0`, the project will follow SemVer strictly.

## [In Development]

Prepares RAFT for its public, open-source release (#18).

### Added

- **Citation metadata** — `CITATION.cff`, so GitHub shows a "Cite this
  repository" button; the README has Citation and License sections.
- **GitHub Actions** — `tests.yml` runs formatting, lint, mypy and the tests
  on Linux, macOS and Windows, and on newer Python versions;
  `release.yml` publishes a version tag to TestPyPI, then to PyPI after a
  maintainer approves, using Trusted Publishing.
- **REUSE compliance** — `REUSE.toml` and `LICENSES/`, with `reuse` and
  `gitleaks` prek hooks; `just leaks` scans the whole history for secrets.

### Changed

- **Copyright notice** — ETH Zurich is the copyright holder, in `NOTICE`, the
  README and the docs; every Python file starts with the two SPDX lines
  instead of the long license header.
- **Package metadata** — maintainer, classifiers, project URLs and license
  files in `pyproject.toml`; the sdist holds only the source, tests and
  release files.

---

## [0.2.1] — 2026-10-23

Patch release: renames the package to `geg-raft` for publishing on PyPI and
adds solver diagnostics and well profiles to the coupling API.

### Added

- **Coupling diagnostics** — `RaftModel.solve_stats` reports the internal
  timesteps, cut attempts and wall time of the last `solve`.
- **Well profiles** — `RaftModel.well_profiles()` returns pressure,
  temperature and flow along each well, the same values written to
  `Well.vtk`.
- **Coupling example plot** — `scripts/coupling_example.py` plots the well
  temperature and pressure at each accepted step to
  `outputs/coupling_example_wells.png`; its default coupling step is now 60
  days.

### Changed
- Boundary condition groups are named with `label:` instead of `tag:`. Rename it in existing configs; a group with `tag:` is no longer written to `outflow.log`.
- `BC_compatible` initial conditions now describe the unperturbed formation: wells, well-node boundary conditions and external boundary conditions are left out of the steady-state solve.
- Well initial conditions under `BC_compatible`: temperature from the rock the well passes through, and pressure hydrostatic from `p_ini` at each well head.
- The first timestep now ends at `t_start + timestep`; it was previously solved at `t_start`, which shifted every later time by one initial timestep.
- Near a table/ramp time or `t_end`, the remaining time is split into two equal steps instead of leaving one short final step; landing on these times may take a step below `timestep_min`.
- Timestep growth never proposes a step below `timestep_min`.
- A run that fails to converge at `timestep_min` exits with status 1 (previously 0); through the API it raises `TimestepError`.
- A well whose roughness / diameter is above 0.05 now gets a warning that the Churchill friction factor is extrapolated beyond the Moody chart, instead of an error; a negative value is still an error.

### Fixed

- **`outflow.log` lag** — pressure and temperature were reported one
  timestep behind the flow rate.
- **Flux constraints** — on line and face groups, each node's flow was
  compared with the group total.
- **SUPG stabilization** — division by zero for near-zero (but non-zero)
  flux.
- **`BC_compatible` output** — the initial-condition solve wrote debug
  output frames and reset `outflow.log`.
- **`BC_compatible` convergence** — failed for wells with no pressure
  reference other than their own boundary conditions.

---

<!-- Version comparison links — verify and update when cutting a release -->
[In Development]: https://github.com/GEG-ETHZ/RAFT/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/GEG-ETHZ/RAFT/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/GEG-ETHZ/RAFT/tree/v0.2.0
