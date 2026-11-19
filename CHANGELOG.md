# Changelog

All notable changes to RAFT are documented here.
RAFT does not yet follow [Semantic Versioning](https://semver.org/) — everything is
pre-release. Starting with `1.0.0`, the project will follow SemVer strictly.

## [In Development]

### Added

- **Linear elastic force balance** — `physics.matrix.mechanics: quasi_static`
  solves the matrix displacement in 2D (plane strain or plane stress),
  axisymmetric and 3D, from the in-situ stress state
  (`initial_conditions.stress`) and a drained `poisson_ratio`. Mechanics
  boundary conditions (`mechanics` blocks) prescribe displacement (component
  lists, `fixed`, `roller`) or traction increments. `Matrix.vtk` gains
  displacement and stress.
- **Poroelastic coupling** — with `flow` and `mechanics` on, displacement and
  pressure are solved monolithically with Biot's storage and strain coupling.
- **Thermal stress** — `heat` can now run with `mechanics` on, adding thermal
  stress to the force balance and thermal terms to the porosity. The monolithic
  solver includes temperature in the displacement system; `operator_split`
  uses the extrapolated temperature for it.
- **Stress recovery option** — `mechanics_controls.stress_recovery` selects
  the output stress: `element` (default), the element-mean stress written per
  cell; `average`, the stress at each element's nodes averaged over the
  elements sharing the node; or a consistent `L2` projection.
- **Fracture contact** — with `mechanics` on, fractures load the matrix
  through their faces: fracture pressure, the in-situ stress, and normal
  penalty contact (`mechanics_controls.penalty_factor`). `BC_compatible`
  initial conditions now include the force balance, so the fractures start
  closed under the in-situ stress; other assignments start undeformed and the
  coupled solver carries the closure. `Fracture.vtk` now shows each fracture as a strip of its two
  faces, separated by the aperture, with the normal jump and contact pressure
  when mechanics is on.
- **Stress-dependent fracture aperture** — with `mechanics` on, the fracture
  aperture follows the contact state through the Bandis–Barton law, with
  `material.fracture.aperture` as the unstressed aperture and the new
  `normal_stiffness` as the stress that halves it. The aperture sets fracture
  storage, conductance, leak-off, and heat exchange, and the displacement is
  coupled to fracture flow and heat in the Newton system. `Fracture.vtk`
  writes the aperture, evaluated at the nodes shared by neighbouring
  elements, and separates the faces by it.
- **Cohesive fractures** — `material.fracture.cohesive` bonds the fracture
  faces with a mixed-mode cohesive law (`linear`, `bilinear`, or `cubic`)
  that carries tension and shear until the bond breaks, with per-fracture
  tensile strength, shear strength, and fracture energy. Damage is
  irreversible; `initial_conditions.fracture_damage` sets it per fracture, 1
  for a debonded fracture. `mechanics_controls.damage_peak_smoothing_coefficient`
  smooths the traction where loading turns to unloading. `Fracture.vtk` gains
  the tangential jump and, with cohesion, the damage and the cohesive normal
  and shear tractions.

### Changed

- **Newton solve scaling** — each field's rows of the coupled tangent are
  scaled to a common magnitude before the solve, which removes the round-off
  that badly scaled displacement-pressure systems left in the pressure update.
- **Stagnated Newton acceptance** — a Newton solve whose residual has reached
  no new minimum in 10 iterations is accepted when it is within 10 times the
  tolerance, so round-off near zero flux no longer fails the step.

### Fixed

- **Fractures reaching the domain boundary** — the exterior faces of the
  matrix elements next to such a fracture stay in the boundary face sets, so
  boundary conditions apply at the nodes where the fracture meets the boundary.
  A fracture tip outside the domain is no longer moved onto the node where the
  fracture crosses the boundary, so that node is split instead of left as a tip.
  The fracture nodes on a boundary face join its node set, so fluid leaves
  through the fracture at a pressure boundary.
- **Leak-off enthalpy with matrix advection off** — each side of the
  fracture/well-matrix exchange takes the advected enthalpy only if it has
  advection on. The matrix no longer accumulates the enthalpy of leaked fluid
  that its Darcy flow cannot carry away, which heated the matrix near
  fractures and at outflow boundaries.
- **Multi-face flux boundary conditions** — a `flux` group spanning several
  faces or lines counted nodes shared between them twice, overestimating the
  applied flux near shared corners and edges.

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
