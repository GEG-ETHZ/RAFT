# Changelog

All notable changes to RAFT are documented here.
RAFT does not yet follow [Semantic Versioning](https://semver.org/) — everything is
pre-release. Starting with `1.0.0`, the project will follow SemVer strictly.

## [In Development]

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

- **Package name** — the package is now `geg-raft` (distribution, on PyPI)
  and `geg_raft` (import). Run it with `uv run -m geg_raft` and import it
  with `import geg_raft as raft`.
- **Laterals coupling example** — the boundary condition labels in
  `config_example_AGS_coupled_laterals.yaml` are now `well_1_inlet` and
  `well_1_outlet`.
- **Theory** — `theory.md` documents the lumped storage mass matrix and its
  diagonal tangents.

---

## [0.2.0] — 2026-10-15

Adds the coupling API for driving RAFT from an external simulator, and
reworks timestepping and `BC_compatible` initial conditions.

### Added

- **Coupling API** — `raft.RaftModel` drives RAFT from an external
  simulator in the same Python process: `solve` advances a trial from the
  last accepted state and returns boundary values, `get_fresh` accepts the
  step and writes its output.
- **External boundary conditions** — the `"external"` boundary condition
  function, set through the coupling API, with `"step"` or `"linear"`
  interpolation over each coupling step. Flow rates are given in kg/s for
  the whole group.
- **Config names without extension** — config files can be named without
  their `.yaml` extension, on the command line and in
  `subsurface.main`/`RaftModel`.
- **Verbosity levels** — `--verbosity {quiet,lite,full}` console output;
  `subsurface.main` and `RaftModel` take the same `verbosity="lite"`, etc.,
  replacing their `verbose=True/False` argument. `lite` prints one line per
  timestep attempt, including cut and retried steps.
- **Coupled example** — `config_example_AGS_coupled.yaml` and
  `scripts/coupling_example.py`: the AGS example coupled to a simple plant
  heat exchanger.

### Changed

- **Boundary condition labels** — groups are named with `label:` instead of
  `tag:`. Rename it in existing configs; a group with `tag:` is no longer
  written to `outflow.log`.
- **`BC_compatible` initial conditions** — now describe the unperturbed
  formation: wells, well-node boundary conditions and external boundary
  conditions are left out of the steady-state solve.
- **Well initial conditions** — under `BC_compatible`, temperature comes
  from the rock the well passes through, and pressure is hydrostatic from
  `p_ini` at each well head.
- **First timestep** — now ends at `t_start + timestep`; it was previously
  solved at `t_start`, which shifted every later time by one initial
  timestep.
- **Landing on table/ramp times** — near a table/ramp time or `t_end`, the
  remaining time is split into two equal steps instead of leaving one short
  final step; landing on these times may take a step below `timestep_min`.
- **Timestep growth** — never proposes a step below `timestep_min`.
- **Exit status** — a run that fails to converge at `timestep_min` exits
  with status 1 (previously 0); through the API it raises `TimestepError`.

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
