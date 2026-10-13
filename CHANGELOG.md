# Changelog

All notable changes to `TANGO Reservoir Simulator` are documented here.
`TANGO Reservoir Simulator` does not yet follow [Semantic Versioning](https://semver.org/) — everything is
pre-release. Once the first public release (`1.0.0`) is cut, the project will
adopt SemVer strictly.

## [In Development]

### Added
- Coupling API, `raft.RaftModel`, for driving RAFT from an external simulator in the same Python process: `solve` advances a trial from the last accepted state and returns boundary values, `get_fresh` accepts the step and writes its output.
- `"external"` boundary condition function, set through the coupling API, with `"step"` or `"linear"` interpolation over each coupling step. Flow rates are given in kg/s for the whole group.
- Config files can be named without their `.yaml` extension, on the command line and in `subsurface.main`/`RaftModel`.
- `--verbosity {quiet,lite,full}` console output levels; `subsurface.main` and `RaftModel` take the same `verbosity="lite"`, etc., replacing their `verbose=True/False` argument. `lite` prints one line per timestep attempt, including cut and retried steps.
- `config_example_AGS_coupled.yaml` and `scripts/coupling_example.py`: the AGS example coupled to a simple plant heat exchanger.

### Changed
- Boundary condition groups are named with `label:` instead of `tag:`. Rename it in existing configs; a group with `tag:` is no longer written to `outflow.log`.
- `BC_compatible` initial conditions now describe the unperturbed formation: wells, well-node boundary conditions and external boundary conditions are left out of the steady-state solve.
- Well initial conditions under `BC_compatible`: temperature from the rock the well passes through, and pressure hydrostatic from `p_ini` at each well head.
- The first timestep now ends at `t_start + timestep`; it was previously solved at `t_start`, which shifted every later time by one initial timestep.
- Near a table/ramp time or `t_end`, the remaining time is split into two equal steps instead of leaving one short final step; landing on these times may take a step below `timestep_min`.
- Timestep growth never proposes a step below `timestep_min`.
- A run that fails to converge at `timestep_min` exits with status 1 (previously 0); through the API it raises `TimestepError`.

### Fixed
- `outflow.log` reported pressure and temperature one timestep behind the flow rate.
- Flux constraints on line and face groups compared each node's flow with the group total.
- Division by zero in the SUPG stabilization for near-zero (but non-zero) flux.
- `BC_compatible` initial conditions wrote debug output frames and reset `outflow.log`.
- `BC_compatible` initial conditions failed to converge for wells with no pressure reference other than their own boundary conditions.

### Removed

<!-- Version comparison links — verify and update when cutting a release.
     Label is [Unreleased] (not [In Development]) so changelog tooling recognises it. -->
[Unreleased]: https://github.com/your-org/your-repo/-/commits/main
