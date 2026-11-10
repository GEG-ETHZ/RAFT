# RAFT 0.2.0 — Reservoir Analysis for TANGO

A finite element method (FEM) simulator for coupled subsurface fluid flow and heat transport in geothermal reservoirs, specifically developed for native coupling with TANGO.

 Physical processes include:
 - Fluid flow
     - Darcy flow
     - Compressible equations of state:
        - Linearized Boussinesq approximation
        - Full EOS via CoolProp for water, super-critical CO2, etc.
     - Fracture flow and leak-off
     - Volumetric poroelastic effects
 - Heat flow
     - Diffusion
     - Advection
     - Fracture heat transfer with conductive and advective leak-off

## Table of Contents
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Building the documentation](#building-the-documentation)
- [Running the simulation](#running-the-simulation)
  - [Run options](#run-options)
  - [Outputs](#outputs)
  - [Thermoelastic cooling](#thermoelastic-cooling)
- [Coupling API](#coupling-api)
- [Theory](#theory)
- [Configuration Parameters](#configuration-parameters)
  - [`mesh`](#mesh)
  - [`regions`](#regions)
  - [`fractures` (optional field)](#fractures-optional-field)
  - [`wells` (optional field)](#wells-optional-field)
  - [`material`](#material)
  - [`initial_conditions`](#initial_conditions)
  - [`boundary_conditions`](#boundary_conditions)
  - [`physics`](#physics)
  - [`simulation`](#simulation)
- [Mesh generation](#mesh-generation)
  - [`domain`](#domain)
  - [`sizing`](#sizing)
  - [Command line options](#command-line-options)
- [Project Structure](#project-structure)
- [Developers](#developers)
  - [Running Checks](#running-checks)

## Getting Started
### Prerequisites
- Python version specified in `.python-version`
- uv (if not installed: [install uv](https://docs.astral.sh/uv/getting-started/installation/)  )<!--NOTE: link to Wiki for uv-->


### Installation
1. navigate to the raft directory and run `uv sync`
1. execute a script via: `uv run some_script.py`


### Building the documentation
1. `uv sync --group docs`
1. `just docs`
1. open `docs/_build/html/index.html` in a browser

This builds the Sphinx HTML docs (including this README and [theory.md](theory.md)) from source.


## Running the simulation
Run a simulation by passing a config file name:
```
uv run -m raft **config file name** **options**
```
For example, to run the example config with one progress line per timestep:
```
uv run -m raft config_example --verbosity lite
```

The config name may be given with or without its `.yaml` extension, and is looked up in the `config/` directory unless `--config-dir` points elsewhere. The mesh file named under `mesh: file:` is read from the same directory. Running without a config name prints the help.

### Run options
| Option | Effect |
| --- | --- |
| `config` | Config file name, with or without `.yaml`, e.g. `config_example` |
| `--config-dir DIR` | Directory holding the config and its mesh (default: `config/` in the repository) |
| `--verbosity {quiet,lite,full}` | Console output (default: `full`), see below |
| `--quiet` | Same as `--verbosity quiet` |
| `--test` | Run the test suite instead of a simulation (same as `uv run pytest`) |
| `-h`, `--help` | List the options |

Console output levels:
* `full`: everything, from the config parameters and defaults applied, through initial conditions, to every timestep's solver iterations and timestep changes.
* `lite`: one line per timestep attempt, with its step number, time, timestep size and progress. A step that is cut and retried prints a line for each attempt, so time lost to cuts stands out from steps that are simply slow.
* `quiet`: nothing.

The run exits with status 0 when it completes, and status 1 if a timestep fails to converge at `timestep_min`.

### Outputs
Results are written to `outputs/<config name>/`, or to `<output_directory>/<config name>/` if [`output_directory`](#simulation) is set:
* `domain.vtk` (and `domain_wells.vtk` with wells): material properties and boundary condition nodes, written once at the start.
* `Matrix.vtk.<n>`, plus `Fracture.vtk.<n>` and `Well.vtk.<n>` when fractures or wells exist: solution fields at output step `n`, if `output_vtk` is on. View them in [Paraview](https://www.paraview.org/).
* `outflow.log`: time, pressure, flow rate and temperature of each [labeled boundary condition group](#boundary_conditions), at every output step.
* `run.log`: the console output, if `output_log` is on. It is written at the chosen verbosity.

Output steps are every `output_frequency` accepted timesteps, starting from the initial condition (step 0).

Meshes are made with [Gmsh](https://Gmsh.info/), by hand or with the generator described under [Mesh generation](#mesh-generation), and must be exported as `.msh` files in Version 2 ASCII format. Gmsh requires a separate download and installation.

### Thermoelastic cooling
RAFT can model pressure- and temperature-dependent permeability using
`raft.matprops.get_effective_permeability`. The effective stress is calculated as

$$
\sigma' = \sigma'_0 + (1-\alpha_B)K_d\alpha_v(T-T_0)
             - \alpha_B(p-p_0),
$$

where $K_d$ is the drained bulk modulus, $\alpha_v$ is the solid thermal expansion
coefficient, and $\alpha_B$ is the Biot coefficient. The Bandis-Barton aperture law
then gives

$$
w = w_0\frac{1+B\sigma'_0}{1+B\sigma'},
\qquad
k_\mathrm{eff} = \frac{w^2}{12}.
$$

For anisotropic permeability, the scalar cubic-law value is multiplied by the
configured directional permeability ratios. The function also returns analytical
derivatives with respect to pressure and temperature for the coupled solver. If
`T_flow_lagged` is enabled, the temperature derivative is set to zero for the
current flow solve.

The permeability analysis script reconstructs these effective fields from saved
VTK output without rerunning the simulation. For the default 2D example, run:

```bash
uv run python scripts/analyze_permeability_outputs.py
```

This writes cell-wise `effective_stress`, `aperture`, and permeability fields to
`outputs/config_example_2D/permeability_analysis/`. To analyze another run, provide
the configuration and output directories explicitly:

```bash
uv run python scripts/analyze_permeability_outputs.py \
  --config outputs/thermoelastic_cooling.yaml \
  --output-dir outputs/thermoelastic_cooling \
  --analysis-dir outputs/thermoelastic_cooling/permeability_analysis
```

The thermoelastic cooling verification test uses a quasi-1D 2D case, compares the
simulation and reconstructed fields with the cold-injection analytical solution,
and writes the comparison plot to
`outputs/thermoelastic_cooling/thermoelastic_cooling_comparison.png`. Run it with:

```bash
uv run pytest tests/test_simulation.py::test_thermoelastic_cooling -q
```

## Theory
RAFT idealizes the subsurface as a continuous, fully-saturated, single-phase porous medium and solves a coupled thermo-hydraulic (pressure/temperature) finite element problem with a compressible, pressure- and temperature-dependent fluid equation of state. The governing equations, fluid EOS, finite element discretization, fracture coupling, mesh cutting algorithm, stabilization, and time-integration schemes are documented in [theory.md](theory.md).

## Configuration Parameters

The simulator uses a YAML configuration file to define simulation options, material properties, initial conditions, boundary conditions, and numerical controls. Below is the list of parameters and their options.

### `mesh`
* **`file`** *(string, default: `"box3D.msh"`)*: Name or relative path to the Gmsh `.msh` file (Version 2.x format). Meshes may be either B8 or T4 elements in 3D or Q4 or T3 in 2D. Meshes can consist of any combination of elements, so long as the element boundaries form a continuous approximation.
* **`dimensions`** *(integer, default: `3`)*: Spatial dimensions of the analysis (`2` or `3`).
* **`regions`** *(subcategories)*: See the `regions` section for specifying regions, or specify the subcategory `"list":` *(str)* and provide the path to a .txt file with a list of corresponding region for each element.
* **`fractures`** *(subcategories)*: See the `fractures` section for specifications.
* **`domain`** *(subcategories, optional)*: Outer boundary of the geometry, used only by the mesh generator (see [Mesh generation](#mesh-generation)). Ignored by the solver, which takes its domain from the mesh file.
* **`sizing`** *(subcategories, optional)*: Element sizes for the mesh generator (see [Mesh generation](#mesh-generation)). Ignored by the solver.

### `regions`
Regions allow you to assign different material properties to different sections of your domain. Each region is defined by a geometric shape which assign the region number to all elements whose centroids fall within the shape. Region numbers correspond to indices in the material property lists. Regions are an independent definition that does not rely on the mesh or domain, but conforming mesh boundaries are encouraged for more accurate results.

If regions are not defined, homogeneity is assumed (all elements are assigned to region 0).

**Region Definition:**
Each region specification requires three fields:
* **`shape`** *(string)*: Type of geometric boundary. Supported shapes:
  * `"sphere"`
  * `"cylinder"`
  * `"polyhedron"`
* **`values`** *(list)*: Shape-dependent parameters (see below).
* **`region_number`** *(integer)*: Region ID assigned to elements within the shape. Must correspond to a valid index in your material property lists.

**Shape-Specific Parameters:**

* **Sphere:**
```yaml
  - shape: "sphere"
    values: [[center_x, center_y, center_z], radius]
    region_number: 1
```
  * Center (list of 3 floats): Coordinates of sphere center. Note that this can be outside the domain.
  * Radius (float): Sphere radius.
* **Cylinder:**
```yaml
  - shape: "cylinder"
    values: [[center_x, center_y, center_z], radius, [axis_x, axis_y, axis_z], height]
    region_number: 1
```
  * Center (list of 3 floats): Coordinates of cylinder axis origin. Note that this can be outside the domain.
  * Radius (float): Cylinder radius.
  * Axis direction (list of 3 floats): Unit vector along cylinder axis (will be normalized automatically).
  * Height (float, optional): Cylinder half-height along axis direction. The cylinder extends equally in both directions from the center point. If omitted, height is unbounded.
* **Polyhedron:**
```yaml
  - shape: "polyhedron"
    values: [[x1, y1, z1], [x2, y2, z2], ..., [xn, yn, zn]]
    region_number: 1
```
  * Vertex list (list of lists): Coordinates of polyhedron vertices. A convex hull is computed from these points to define the region.
**Assignment Rules:**
* Regions are processed in the order specified in the config file.
* If an element centroid falls within multiple regions, it is assigned to the **last (highest-priority) region** that contains it.
* All elements are initially assigned to region 0; unassigned elements remain in region 0.
* Material properties are indexed by region number, so ensure your material lists have entries for each region used.
* Multiple shape definitions can be assigned the same region numbers.

### `fractures` (optional field)
Fractures use a conforming mesh approach and cut through the existing mesh, creating conforming nodes around them; see [Fracture Coupling](theory.md#fracture-coupling) for the underlying physics. These cuts may lead to poorly conditioned elements, so care should be taken when examining results. Mesh refinement around fractures is often necessary.

> **Current limitation:** Fractures are only supported in 2D analyses (`dimensions: 2`). 3D fracture support is not yet implemented (a warning is printed and the mesh is returned unmodified).


**Fracture Definition:**
Each fracture specification requires three fields:
* **`shape`** *(string)*: Type of geometric boundary. These shapes are only used in 3D, in 2D all fractures are represented as lines. Supported shapes:
  * `"ellipse"`
  * `"rectangle"`
* **`values`** *(list)*: $[[x_c, y_c, z_c ], [a,b], [\theta_x, \theta_y, \theta_z]]$
  * $[x_c, y_c, z_c ]$ *(list[float], no default)*: coordinates of the plane center. In 2D analysis, only $x_c$ and $y_c$ are considered.
  * $[a,b]$ *(list[float], no default)*: extents. For the ellipse, $a$ = major axis distance, $b$ = minor axis distance. For the rectangle, $a$ = half-length, $b$ = half-width. In 2D, $a$ is taken as the fracture half-length.
  * $[\theta_x, \theta_y, \theta_z]$ *(list[float], no default)*: plane angles in degrees. $\theta_x$ = plane angle with respect to the x-axis, $\theta_y$ = plane angle with respect to the y-axis, , $\theta_z$ = plane angle with respect to the z-axis. In 2D, only $\theta_x$ and $\theta_y$ are considered.

Hydraulic and thermal properties (`aperture`, `heat_transfer_factor`) are not specified here; they are defined per-fracture under `material.fracture` (see the `material` section below and [Fracture Coupling](theory.md#fracture-coupling) for their role in the governing equations).

> **IMPORTANT NOTE FOR FRACTURE SPECIFICATIONS**: The fracture mesh cutting algorithm can fail! Always check the quality of the cut elements in `domain.vtk`. If the algorithm fails, it can usually be solved by refining the mesh around fractures and/or junctions.

### `wells` (optional field)
Wells carry fluid from the surface into the rock mass and interact with the surrounding rock mass before the fluid reaches the injection point. Wells always interact thermally and may interact hydraulically in an open perforation interval. See [Well Coupling](theory.md#well-coupling) for the underlying physics. Mesh refinement around wells is often necessary.

> **Current limitation:** Wells are currently only supported in 3D analyses (`dimensions: 3`). This will be extended to support axisymmetric analyses in 2D in the future.

**Well Definition:**
Each well specification requires two fields:
* **`alignment`** *(list of list of floats as [start, end] point pairs)*: Each pair defines a line segment (points matching the spatial dimensions).
  * The nodes and connectivity are found by snapping to the path of existing mesh edges closest to that segment. Lines may extend beyond the reservoir boundary.
  * If the exact path does not exist among the element edges, the line snaps to the shortest edge path connecting the start and end node. This throws a warning and reports the true vs. prescribed line lengths.
* **``dL_max``** *(float, default: L/10)*: Maximum element length for sections of the well which exceed the reservoir boundaries.

Wells that share physical nodes in the mesh are automatically connected. Thus, perforation intervals or branching wells can be specified by having the wells share start or end points.

**Mesh sizing:** when the mesh is built with [raft_to_geo](#mesh-generation), `mesh.sizing.well` sets the element size around the wells, in the same profile form as `frac` and `bc`:

```yaml
mesh:
  sizing:
    outer: 25
    well: [[5, 10], [5, 50]]   # 5 at the well, growing to 10 over 5 m, then to `outer` over 50 m
```

Well alignments are discretized in the mesh, so element edges follow them and wells sharing a start or end point share the node that connects them. Alignments extending past the reservoir are clipped to [mesh: domain](#domain) for meshing, with a warning naming the well; `dL_max` applies to those outside sections and so does not affect the mesh.

Hydraulic and thermal properties (`diameter`, `perforation_conductance`, `heat_transfer_factor`, `roughness`) are not specified here; they are defined per-well under `material.wells` (see the `material` section below and [Well Coupling](theory.md#well-coupling) for their role in the governing equations).

### `material`

> **CONSISTENT UNITS**: In general, RAFT does not require any specific units, so long as the units are internally consistent. For example, lengths can be specified in feet, timesteps in months, temperature in Farenheit, and energy in J, so long as the thermal conductivity is then specified in J/ft/month/F. The exception this comes when using the real equation of state for the fluid or the viscosity correlations, in which case standard SI units of m, s, kg, J, K, and Pa **must** be used.

* **`solid`**: Rock mass properties. Rock mass properties are defined per region. If regions are defined, RAFT expects a list with a number of entries equal to the number of regions. Region numbers correspond to property indices.
  * **`bulk_mod`** *(float, default: `20.0e9`)*: Solid bulk modulus in Pascals ($\text{Pa}$).
  * **`density`** *(float, default: `2600.0`)*: Rock density in $\text{kg/m}^3$.
  * **`therm_cond`** *(float or list, default: `3.0`)*: Rock thermal conductivity in $\text{W/m}\cdot\text{K}$. Can be a scalar for isotropic conductivity, or a list of length equal to the dimensions for anisotropic conductivity (e.g. `[3.0, 2.0, 0.1]`). For anisotropic properties, the list corresponds to the local x, y, z axes which can be rotated with `rotation_x/y/z`.
  * **`specific_heat`** *(float, default: `800.0`)*: Rock specific heat capacity in $\text{J/kg}\cdot\text{K}$.
  * **`therm_expansion`** *(float, default: `12.0e-6`)*: Rock volumetric linear thermal expansion coefficient in $1/\text{K}$.
  * **`porosity`** *(float, default: `0.01`)*: Reservoir porosity (dimensionless fraction).
  * **`biot_coeff`** *(float, default: `0.5`)*: Biot-Willis coefficient (dimensionless).
  * **`permeability`** *(float or list, default: `1.0e-16`)*: Permeability in $\text{m}^2$. Can be a scalar for isotropic permeability, or a list of length equal to the dimensions for anisotropic permeability (e.g. `[1.0e-13, 1.0e-11, 1.0e-12]`). For anisotropic properties, the list corresponds to the local x, y, z axes which can be rotated with `rotation_x/y/z`.
  * **`stress_init`** *(float, default: `32.0e6`)*: Initial effective stress used by the pressure- and temperature-dependent aperture/permeability law, in Pa.
  * **`rotation_x`** *(float, default: `0.0`)*: Rotation angle of the local coordinate system around the x-axis in degrees.
  * **`rotation_y`** *(float, default: `0.0`)*: Rotation angle of the local coordinate system around the y-axis in degrees.
  * **`rotation_z`** *(float, default: `0.0`)*: Rotation angle of the local coordinate system around the z-axis in degrees.

These rotation parameters define a 3D rotation matrix $R = R_x \cdot R_y \cdot R_z$ to rotate the local coordinate system along which the rock mass anisotropy is defined. The rotation is applied to both anisotropic permeability ($\boldsymbol{k}_f = R \operatorname{diag}(k_f) R^T$) and anisotropic solid thermal conductivity ($\Lambda_s = R \operatorname{diag}(\lambda_s) R^T$).

* **`fluid`**: Fluid phase properties.
  * **`fluid_type`** *(string, default: `"WATER"`)*: Only required for a real fluid EOS. Set the working fluid in the reservoir. Required when using nonlinear properties, as it controls the property lookup in CoolProp. E.g. `"WATER"` for water or `"CO2"` for supercritical CO2.
  * **`density`** *(float, default: `1000.0`)*: **Density determines whether to use the Boussinesq approximation or a real EOS.**
    * `density: 0` - Indicates to use the real EOS. All other fluid properties are ignored except for `viscosity`, `therm_cond`, and `fluid_type`.
    * `density: non-zero float` - sets reference density $\rho_0$ in $\text{kg/m}^3$ at reference state $p_0, T_0$.
  * **`reference_temperature`** *(float, default: `293.0`)*: Reference temperature $T_0$ for the Boussinesq approximation in $\text{K}$. Not used when nonlinear density is specified.
  * **`reference_pressure`** *(float, default: `1.0e5`)*: Reference pressure $p_0$ for the Boussiesq approximation in $\text{Pa}$. Not used when nonlinear density is specified.
  * **`bulk_mod`** *(float, default: `2.0e9`)*: Fluid bulk modulus in Pascals ($\text{Pa}$). Used by Boussinesq approximation, ignored when using real EOS.
  * **`therm_expansion`** *(float, default: `200.0e-6`)*: Fluid volumetric thermal expansion coefficient in $\text{K}^{-1}$. Used by Boussinesq approximation, ignored when using real EOS.
  * **`specific_heat`** *(float, default: `4180.0`)*: Fluid specific heat capacity in $\text{J/kg}\cdot\text{K}$. Used by Boussinesq approximation, ignored when using real EOS.
  * **`therm_cond`** *(float, default: `0.6`)*: Fluid thermal conductivity in $\text{W/m}\cdot\text{K}$.
  * **`viscosity`** *(float, default: `1.0e-3`)*: Fluid dynamic viscosity in $\text{Pa}\cdot\text{s}$.
    * **Default behaviour:** Viscosity is a constant set to the specific *float* value. Recommended for cases with small variations in pressure or temperature.
    * **True behaviour:** Set to `0` to instead use a temperature- and pressure-dependent, real-fluid viscosity looked up from CoolProp for the configured `fluid_type`. Recommended for super-critical CO2.
    * **Pure thermal behabiour:** Set to `-1` to instead use a temperature-dependent viscosity with the form $\mu = A e^{-B/T + CT + DT^2}$. Curently only works for water. For water, $ A = 1.856·10^{−11} \text{mPa}\cdot \text{s}, B = 4209 \text{K}, C = 0.04527 \text{K}^{-1}$ and $D = -3.376\times 10^{-5} \text{K}^{-2}$.


> **IMPORTANT NOTE FOR NONLINEAR FLUID PROPERTIES**: When using non-linear material properties, the magnitude of pressure and temperature matter. Temperature __must__ be specified in Kelvin, while pressure boundary conditions should be specified above the fluid's saturation point. Temperature and pressure are clipped to $274$–$1900\,\text{K}$ and $0.1$–$700\,\text{MPa}$ before the lookup, to keep unconverged solver iterates within CoolProp's valid range; pressures should stay comfortably above the lower bound, since it sits close to water's saturation curve at higher temperatures.

* **`fracture`** *(required if `fractures` are defined in `mesh`)*: Hydraulic and thermal properties of the fracture(s), used in the [fracture governing equations](theory.md#fracture-coupling). Each property is a list indexed the same way as fractures are listed under `mesh.fractures`.
  * **`aperture`** *(list[float], no default)*: Hydraulic aperture of the fracture, $w$, used in the cubic-law in-plane flow and leak-off terms.
  * **`heat_transfer_factor`** *(list[float], no default)*: Multiplier, $F$, on the fracture-matrix heat transfer coefficient, which assumes laminar flow between parallel plates. Use 1 for the standard laminar value, values below 1 to reduce heat exchange (0 is perfectly insulated), and large values to approach local thermal equilibrium between fracture and matrix.

* **`wells`** *(required if `wells` are defined in `mesh`)*: Hydraulic and thermal properties of the well(s), used in the [well governing equations](theory.md#well-coupling). Each property is a list indexed the same way as wells are listed under `mesh.wells`.
  * **`diameter`** *(list[float], no default)*: Hydraulic diameter of the well, $D$, which determines frictional resistance.
  * **`roughness`** *(list[float], no default)*: Roughness (units of length) used to calculate friction in turbulent flow.
  * **`perforation_conductance`** *(list[float], no default)*: Conductance from the well to the reservoir in ($m^2/m$). The pressure drop out of the well is typically a function of the skin effect and perforation radius, as well as damaged zones. A value of 0 is an unperforated well. A value of 1 typically large enough to hydraullically link the well and the reservoir while neglecting the pressure drop caused by perforations and skin effects.
  * **`heat_transfer_factor`** *(list[float], no default)*: Multiplier, $F$, on the well-rock heat transfer coefficient. The coefficient follows the flow regime: constant in laminar flow (Re < 2000), the Gnielinski correlation in turbulent flow (Re > 2300, including the effect of `roughness`), and interpolated between. Use 1 for an uninsulated well, values below 1 for insulated sections (0 is perfectly insulated); values above 1 are permitted.
> **Injecting and Producing via Wells**: For the well to interact hydraulically with the reservoir (i.e. inject or produce fluid), the connected well chain must have at least one interval with non-zero perforation conductance.

### `initial_conditions`
* **`assignment`** (*string, default*: `"constant"`): determine how initial conditions are assigned.
  * `"constant"`: Uniform conditions assumed throughout for pressure and temperature. Inputs required under **`initial_conditions`**:
    * **`T_ini`** *(float, default: `20.0`)*: Initial temperature in $^\circ\text{C}$.
    * **`p_ini`** *(float, default: `1.0e5`)*: Initial pressure in $\text{Pa}$.
  * `"gradient"`: A gradient is applied in the direction of a reference vector. Not that this may not be at equilibrium in the case of heterogeneous properties. Inputs required under **`initial_conditions`**:
    * **`T_ini`** *(float, default: `20.0`)*: Reference temperature at minimum point along gradient vector in domain in $^\circ\text{C}$.
    * **`p_ini`** *(float, default: `1.0e5`)*: Reference pressure at minimum point along gradient vector in domain in $\text{Pa}$.
    * **`T_grad`** *(float, default: `0.0`)*: Temperature gradient along the gradient vector in $^\circ\text{C/m}$.
    * **`p_grad`** *(float, default: `0.0`)*: Pressure gradient along the gradient vector in $\text{Pa/m}$.
    * **`grad_direction_T`** *(list[float], default: `[0,0,-1]`)*: Vector in the direction of the temperature gradient. Automatically normalized.
    * **`grad_direction_p`** *(list[float], default: `[0,0,-1]`)*: Vector in the direction of the pressure gradient. Automatically normalized.
  * `"BC_compatible"`: Solve the non-linear transport problem for the steady-state conditions within the domain under the boundary conditions applied at time $t=t_0$. The solved pressure and temperature fields are applied as initial conditions and reference conditions. Invokes the appropriate solver based on input material properties, regardless of user timestep solver specification.
    * The steady state describes the unperturbed formation: wells, boundary conditions on well nodes, and `"external"` boundary conditions are left out of this solve.
    * Wells then take the temperature of the rock they pass through; outside the domain, they take the temperature of the nearest well node inside it. Well pressure is hydrostatic along each well, starting from `p_ini` at the well head (the first alignment point). Connected wells share the head of the lowest-numbered well.

### `boundary_conditions`
`boundary_conditions` is a list of groups at the root of the config. Each group applies to one geometry feature (specified using one of four geometric methods: node indices, coordinate proximity, named faces, or lines), and can carry a `hydraulics` block specifying pressure or fluid flux, a `thermal` block specifying temperature or heat flux, or both.

**Sample boundary condition group**
```yaml
- coords: [[x1, y1, z1], [x2, y2, z2]]  (geometry specification - required)
  label: "inlet"                        (label - optional)
  hydraulics:                           (field)
      type: "flux"                      (type - required)
      values: [[0, 0.001], 100]         (values - required)
      function: "ramp"                  (function - optional)
      constraints:                      (constraints - optional)
          maximum_pressure: 16.0e+6
          minimum_pressure: 1.0e+4
  thermal:                              (field)
      type: "temperature"               (type - required)
      values: [[0, 0.001], 100]         (values - required)
      function: "ramp"                  (function - optional)
```

**Geometry Specification (Required):**
Each boundary condition group must specify nodes using one or a combination of the following methods:

* **`nodes`** *(list of integers, or string)*: Direct node IDs where the boundary condition is applied. The string `"all"` is also accepted to fix all nodes in the mesh, or `"matrix"` to fix all nodes in the reservoir while leaving fracture and wellbore nodes unconstrained.
* **`coords`** *(list of coordinates)*: List of coordinate points; the mesh node closest to each point will be selected. Each coordinate should be a list matching the spatial dimensions (e.g., `[x, y, z]` for 3D, `[x, y]` for 2D).
* **`faces`** *(list of strings)*: Named boundary faces from preprocessing. Supported face identifiers: `"x_min"`, `"x_max"`, `"y_min"`, `"y_max"`, `"z_min"`, `"z_max"`. All nodes on the specified faces will be included. Fluxes are integrated over the face.
  * Fluid fluxes: Specify `values` as mass flux in $(\text{kg}/\text{s})/\text{m}^2$.
  * Temperature fluxes: Specify `values` in $\text{W} / \text{m}^2$.
* **`lines`** *(list of float `[start, end]` point pairs)*: Each pair defines a line segment (points matching the spatial dimensions, as in `coords`). The nodes and connectivity are found by snapping to the path of existing mesh edges closest to that segment. The line is snipped to the domain boundaries. If the exact path does not exist among the element edges, the line snaps to the shortest edge path connecting the start and end node. This throws a warning and reports the true vs. prescribed line lengths. You can view the nodes with `line` boundary conditions in `domain.vtk` under the `line_BC_nodes` field. Fluxes are integrated over the prescribed line.
  * Fluid fluxes: Specify `values` as mass flux in $(\text{kg}/\text{s})/\text{m}$.
  * Temperature fluxes: Specify `values` in $\text{W} / \text{m}$.
* **`well`** *(list `[index (int), location (string)]`)*: Specify conditions on a well, only valid if `wells` is specified in the mesh. The well boundary condition specification takes the index of the well (corresponding to its 0-indexed position as specified in `mesh`) and the location as a string. Location only accepts `inlet` or `outlet` and respectively map to the first and last coordinate specified by the well alignment.

**Field Blocks (At least one required):**
* **`hydraulics`**: controls fluid pressure/flux at this group.
  * `type: "pressure"`: Dirichlet condition (fixed pressure).
  * `type: "flux"`: Neumann condition (fixed fluid flux).
* **`thermal`**: controls temperature/heat flux at this group.
  * `type: "temperature"`: Dirichlet condition (fixed temperature).
  * `type: "flux"`: Neumann condition (fixed heat flux).

A group with only `hydraulics` affects pressure alone; a group with only `thermal` affects temperature alone; a group with both applies both conditions over the same geometry. Each block carries its own `function`/`values` (and, for `hydraulics`, an optional `constraints` block — see **Constraints** below).

**Value (Required) and Function (Optional) Specification:**
 The function applied to the nodes is determined by the field `function`. Each group should contain a field named `"values"` which depends on the type of function being applied.
* `function`: `"constant"`
  * Constant value. Default if function not specified.
  * `values`: `[specified value]` Single value float applied to all nodes in the group.
* `function`: `"linear"`
  * Linearly changing with time.
  * `values` : `[slope, intercept]` defines $y(t) = \text{slope} \cdot t + \text{intercept}$.
* `function`: `"gradient"`
  * Linearly changing in space following $y = y_0 + m (g - \min(g))$.
  * `values` : `[direction, reference value, slope]`
    * `direction`: 3x1 list of floats defining a vector in the gradient direction, $\vec{s}$ . Automatically normalized. Defines local coordinate system $g = \vec{X}\cdot \vec{s}$.
    * `reference value`: The reference value of the gradient at the minimum of $g$ within the specified nodes, $y_0$.
    * `slope`: gradient $m$ along $g$.
* `function`: `"ramp"`
  * Linear increase to a value $y_{ramp}$ over time period $t_{ramp}$.
  * `values` : `[[y0, y1], t_ramp]` defines:
        $\quad y(t) = y_0 + \frac{t}{t_{ramp}}(y_1 -y_0)\quad \forall t<t_{ramp}$
        $\quad y(t) = y_1 \quad \forall t\geq t_{ramp}$.
* `function`: `"table"`
  * Accepts a series of times and values, linearly interpolating between them. 0 is assumed before the first listed time. The function is assumed constant beyond the final time listed in the table.
  * `values` : `[[t_0, t_1, ... , t_n], [y_0, y_1, ... , y_n]]` in which $t_i$ are a list of increasing times and $y_i$ are the corresponding value at that time.
  * RAFT indexes all times $t_i$ to ensure that timesteps occur at those specific times and the timestepping code does not step over them.
* `function`: `"external"`
  * The value is set by an external simulator at each coupling step through the [Coupling API](#coupling-api).
  * Requires a `label`, unique among all labeled groups. Allowed types: `hydraulics` `"pressure"` or `"flux"`, and `thermal` `"temperature"`. `constraints` are allowed on a `"flux"` group.
  * A flow rate is given to the API as a total for the group in kg/s and spread over the group's nodes, line or face.
  * `interpolation` *(string, default: `"step"`)*: `"step"` holds the new value over the whole coupling step; `"linear"` ramps from the last accepted value to the new one. The first coupling step is always stepwise.
  * `values` is optional; if given, it is used as a constant when the config is run directly with `uv run -m raft`.

**Label Specification (Optional):**
Flow boundary condition groups can optionally be marked with a `label: "groupname"` entry. If they are marked with a label, the group pressure, flow, and temperatures will be output to *outflow.log* in the output directory. Outputs from each group are summarized as:
* groupname_pressure: arithmetic mean of pressure over the group nodes. Note that this mean is not weighted by nodal area, and thus caution should be used when examining this value over pressure faces with non-uniform meshes.
* groupname_flow: sum of flow rates over the group nodes.
* groupname_temperature: flow-weighted average temperature over the group nodes.

**Constraints**
Constraints are limits placed on a `hydraulics` block with `type: "flux"`. A block with `constraints` cannot use the `"gradient"` function.

* **`maximum_pressure`** / **`minimum_pressure`** *(float, optional)*: Pressure limits at the wellbore nodes. Either may be omitted to leave that side unbounded.

The group follows its specified fluid flux until any node's pressure reaches `maximum_pressure` or `minimum_pressure`, at which point the group switches to a fixed-pressure condition according to which bound was violated. It switches back to the original flux condition once the flux implied by holding that pressure would exceed the originally specified flux target. Each group is evaluated collectively - any one pressure node exceeding the constraints triggers the swap. For fluid flux constraints, line and face geometry types are integrated over the line/face and the prescribed integrated total flux is compared against the current integrated total flux.

Only one switch is permitted per timestep to avoid oscillation between the two states, so tighten the timestep if a run needs faster response at the switch point. Wellbore groups are included when labeling groups for `outflow.log`.

### `physics`
Physics switches for each domain: `matrix`, `fractures`, and `wells`. Fractures and wells inherit any field not set for them from `matrix`, so a block is only needed where a domain differs.

```yaml
physics:
  matrix:          # frozen pressure field, conduction only
    flow: off
    heat: transient
    advection: off
  wells:
    flow: steady
    advection: on
```

* **`flow`** *(string, default: `"transient"`)*: Pressure equation mode, `"off"`, `"steady"` or `"transient"`.
* **`heat`** *(string, default: `"transient"`)*: Temperature equation mode, `"off"`, `"steady"` or `"transient"`.
* **`advection`** *(on/off, default: `off`)*: Heat advection. Enthalpy outflow at fluid boundaries without a thermal BC only applies on nodes in domains with advection on.

An `off` field holds its initial values in that domain, and its equations are not assembled, so it costs no solver time. Its values still enter property evaluations (e.g. density), so advection with `flow: off` transports heat with the Darcy flux of the frozen pressure field. A field that is `off` in every domain skips its solve entirely, giving a single-physics solver.

Fracture/well-matrix coupling is assembled for a field if either side solves it, and includes advective exchange if either side has advection on.

### `simulation`
 Simulation controls
* **`t_start`** *(float, default: `0.0`)*: Start time of the simulation.
* **`t_end`** *(float, default: `100.0`)*: End time of the simulation.
* **`timestep`** *(float, default: `1.0`)*: Simulation timestep.
* **`timestep_min`** *(float, default: `1.0`)*: Minimum allowed timestep.
* **`timestep_max`** *(float, default: `1.0`)*: Maximum allowed timestep.

Output Controls
* **`output_frequency`** *(integer, default: `1`)*: Controls how often outputs are written (every $N$ timesteps).
* **`output_vtk`** *(integer/boolean, default: `0`)*: If set to `1`, enables writing mesh and solution fields to `.vtk` files for visualization.
* **`output_viscosity`** *(integer/boolean, default: `0`)*: If set to `1`, enables output of fluid viscosity fields.
* **`output_log`** *(integer/boolean, default: `0`)*: If set to `1`, logs console output to a file in the output directory.
* **`output_directory`** *(string, optional)*: Custom output directory path. Defaults to `outputs/` under the repository root.

Physical controls

* **`rate_order`** *(integer, default: `1`)*: The approximation order of rate of change. Accepts `1` for a 1st order backwards Euler approximation, or `2` for a 2nd order backwards Euler approximation.
* **`gravity`** *(float, default: `0.0`)*: Set the magnitude of the acceleration due to gravity in $\text{m}/\text{s}^2$, typically `-9.81`. Gravity is assumed to act in the z-axis in 3D and the y-axis in 2D.
* **`axisymmetric_on`** *(integer/boolean, default: `0`)*: Set to `1` to solve the axisymmetric case. Only valid in 2D analysis.

Numerical solver controls
* **`solver_type`** *(string, default: `"linearized"`)*: Type of time integration solver. Options are `"linearized"` or `"fully_coupled"`. The `fully_coupled` solver solves both pressure and temperature monolithically for the current timestep. The `linearized` solver uses an operator splitting scheme, extrapolating temperature to the current timestep, solving pressure with the extrapolated temperature, then updating temperature with the new pressure.  See [Time Discretization and Solution Scheme](theory.md#time-discretization-and-solution-scheme) for more details.
* **`rel_error_T`** *(float, default: `1.0e-2`)*: Relative error tolerance used for temperature extrapolation checks (`"linearized"` solver only).
* **`iterative_controls`** *(dictionary)*: Non-linear Newton-Raphson solver settings. Because fluid properties are nonlinear functions of pressure and temperature, both solver types use these controls: `"fully_coupled"` for its single monolithic solve, and `"linearized"` for each of its two per-field (pressure, then temperature) sub-solves.
  * **`NR_tol`** *(float, default: `1.0e-6`)*: Newton-Raphson residual tolerance.
  * **`iter_max`** *(integer, default: `100`)*: Maximum number of iterations per timestep.
  * **`aitken_relaxation`** *(integer/boolean, default: `1`)*: Turn on (`1`) or off (`0`) Aitken's $\Delta^2$ relaxation for non-linear convergence.
  * **`NR_step`** *(float, default: `1.0`)*: Newton-Raphson step/line-search size; scales each iteration's solution update before any Aitken relaxation is applied.

## Coupling API
`raft.RaftModel` couples RAFT to an external simulator running in the same Python process that sets the boundary conditions. The RaftModel relies on the external simulator to determine when a coupling timestep is converged and requires an explicit call to accept the trial state and proceed to the next timestep. Every `solve` restarts from the last accepted state, so it can be repeated as often as needed.

```python
from raft import RaftModel

model = RaftModel("config_example_AGS_coupled.yaml", t_start=0.0)
for t_start, t_end in coupling_steps:
    while not converged:
        bcs = {"inlet": {"p": p_in, "T": T_in, "q": q_in}, "outlet": {"p": p_out, "T": T_out, "q": q_out}}
        results = model.solve(t_start, t_end, bcs)
        ...  # update the plant from results["outlet"]["T"], etc.
    model.get_fresh()  # accept the step and write its output
```

* **`RaftModel(config_name, t_start, config_dir=None, verbosity="full")`**: preprocesses the config, assigns initial conditions at `t_start` and writes output frame 0. The config's `t_start` is replaced and its `t_end` is unused. `verbosity` takes `"quiet"`, `"lite"` or `"full"`, as for `--verbosity`.
* **`solve(t_start, t_end, bcs)`**: advances from the last accepted state to exactly `t_end` and returns the boundary values there. `t_start` must equal the last accepted time.
  * `bcs` must contain every label of a group with an [`"external"`](#boundary_conditions) field, and only those. Each entry gives `p` (Pa), `T` (K) and `q` (kg/s, positive into the domain); RAFT uses the ones matching the group's types and ignores the rest.
  * Returns, for each label, `p` and `T` (arithmetic mean over the group's nodes, not flow-weighted), `q` (summed over the nodes, kg/s, positive into the domain) and `mode`, the active hydraulic condition: `"flux"`, `"pressure"` (for a pressure group, or a flux group held at a pressure [constraint](#boundary_conditions)) or `"none"` for a thermal-only group.
  * If the solver fails at `timestep_min`, `solve` raises `raft.subsurface.TimestepError` and the accepted state is unchanged, so the step can be retried, for example with a shorter coupling step.
* **`get_fresh()`**: accepts the last `solve` and writes one output frame at its end time, so `output_frequency` counts coupling steps. Calling it without a new `solve` raises an error.
* **`solve_stats`**: what the last `solve` did internally, as `raft.subsurface.StepStats`:
  * `accepted_steps` and `dt_accepted`: the number and sizes of the internal timesteps taken.
  * `attempts`: every timestep solve, including rejected ones; `cut_extrapolation` and `cut_newton` count rejections for temperature extrapolation error and Newton–Raphson failure.
  * `constraint_switches`: extra solves after a wellbore constraint switched. `dt_min_proceeds`: steps accepted at `timestep_min` despite an error above tolerance.
  * `wall_time`: seconds spent in the `solve`.
* **`well_profiles()`**: one `WellProfile` per well, for the pending trial, or else the last accepted step. Nodes run from the well's first alignment point: `s` (distance along the well), `coords`, `p`, `T` and `inside` (whether the node lies in the rock domain). Element values sit at `s_mid`: `q` (mass flow rate in the direction of increasing `s`, kg/s), `reynolds` and `heat_transfer_coefficient` (well to rock). These are the same values written to `Well.vtk`.

Notes:
* RAFT chooses its own timesteps inside each coupling step and lands exactly on `t_end`. A coupling step longer than the internal timestep takes several internal steps, and each `solve` repeats all of them, so the iteration cost grows with the number of internal steps per coupling step.
* `run.log` is written only while RAFT's own calls run; anything the external simulator prints is not captured.
* `scripts/coupling_example.py` runs the AGS example against a simple plant heat exchanger: `uv run python scripts/coupling_example.py`.

## Mesh generation

Gmsh is not a part of RAFT, but to simplify mesh generation the script `scripts/raft_to_geo.py` is provided which generates a Gmsh `.geo` and `.msh` files. Gmsh is not provided with raft and requires separate download and installation. Meshing by hand or from another source is still supported, this script is provided as a convenient option to keep the mesh and config files synchronized. The script `raft_to_geo.py` reads the config file and generates corresponding files to ensure that the mesh contains discretized boundary condition geometries, fractures, and region boundaries:
  - nodes are placed exactly where `coords` boundary conditions exist
  - element edges are placed exactly along the specifications of `lines`, fracture lines and faces in 2D and 3D respectively, well alignments in 3D, and region boundary edge lines and faces in 2D and 3D respectively

Explicit discretization of these features reduces the error introduced by the mesh and ensures consistency as the mesh is refined.

**Running the script:**
```bash
uv run just geo **config file name** **options**
```
Example:
```
uv run just geo config_example_2D -o 2D_generated
```

This generates `config/2D_generated.geo` and the matching `.msh` using the specifications from `config_example_2D`. The file name defaults to the name specified in [mesh: file](#mesh), unless an alternative name is passed with the flag `-o **alternative name**`. Existing files are never overwritten without `-f` flag.

The ```just geo``` recipe runs `uv run --with Gmsh scripts/raft_to_geo.py <config> --mesh`, which pulls in Gmsh for that command only, and passes any extra flags through. Script names default to `config/<config name>.yaml`. The `.msh` output is Version 2.2 ASCII with T3 triangles in 2D and tetrahedra in 3D. Quadrilateral Q4 elements can be specified in 2D with `--quads`. `raft_to_geo.py` does not support meshing arbitrary hexahedral B8 elements, but they are supported by RAFT if supplied by custom meshing.

### `domain`
The outer boundary, required by the script. `box` takes two opposite corners; `polyhedron`, `sphere` and `cylinder` take the same values as the equivalent [region](#regions) shapes.

```yaml
mesh:
  domain:
    shape: "box"
    values: [[-51, 0], [102, 75]]   # [[xmin, ymin, zmin], [xmax, ymax, zmax]]; drop z in 2D
```

### `sizing`
Element sizes, each overridable from the command line. Only `outer` is required.

```yaml
mesh:
  sizing:
    outer: 10                                    # size far from any feature
    frac: [[0.05, 0.05, 1, 10], [0.2, 5, 5, 5]]  # size profile off the fractures
    bc: 0.5                                      # a bare number is one stage over `transition`
    well: [[5, 10], [5, 50]]                     # size profile off the wells (3D only)
    transition: 25        # stage length for bare-number profiles (default: 5*outer)
    region_max: 3         # coarsest size on a region boundary
    boundary_growth: 0.25 # how strongly region boundaries follow nearby features
    quads: false          # Q4 elements instead of T3 (2D only)
```

A profile $[[s_1, ..., s_n], [L_1, ..., L_n]]$ sets the element size to $s_1$ at the feature, then ramps $s_1 \rightarrow s_2$ over $L_1$, $s_2 \rightarrow s_3$ over $L_2$, and finally $s_n \rightarrow$ `outer` over $L_n$. Sizes need not increase, so repeating a value holds that size for a set distance: the example above holds 0.05 for 0.2 m, then grows to 1 over 5 m, to 10 over the next 5 m, and to `outer` over the 5 m after that. `frac`, `bc` and `well` are independent, so fractures may be finer than boundary conditions or wells, or the other way around.

Region boundaries are refined according to their proximity to fractures, wells and boundary conditions, so an interface near a fracture is resolved more finely than one far away. `region_max` caps that refinement and `boundary_growth` sets how strongly it follows the nearby feature. Stretches of a region boundary that lie on the domain boundary are left at `outer`.

### Command line options
| Option | Effect |
| --- | --- |
| `config` | Config file; a bare name resolves to `config/<name>.yaml` |
| `-o`, `--output` | Output `.geo`; a bare name resolves into `config/` (default: the stem of `mesh.file`) |
| `--mesh` | Also run Gmsh and write the `.msh` |
| `-f`, `--force` | Overwrite existing output files |
| `--size-outer` | Overrides `sizing.outer` |
| `--size-frac`, `--dist-frac` | Fracture sizes and stage lengths, comma separated (e.g. `--size-frac 0.05,0.05,1,10 --dist-frac 0.2,5,5,5`) |
| `--size-bc`, `--dist-bc` | The same for point and line boundary conditions |
| `--size-well`, `--dist-well` | The same for wells |
| `--transition` | Stage length when only sizes are given |
| `--size-region-max` | Overrides `sizing.region_max` |
| `--boundary-growth` | Overrides `sizing.boundary_growth` |
| `--quads` | Q4 elements instead of T3 (2D only) |


## Project Structure
```
└── tango-reservoir-simulator
    ├── README.md
    ├── CONTRIBUTING.md
    ├── CHANGELOG.md
    ├── theory.md
    ├── pyproject.toml
    ├── justfile
    ├── .gitignore
    ├── .python-version
    ├── .pre-commit-config.yaml
    ├── .gitlab-ci.yml
    ├── config/
    ├── outputs/
    ├── scripts/
    │   ├── coupling_example.py
    │   ├── debug_fracture_angles.py
    │   ├── raft_to_geo.py
    │   ├── plot_fluid_props.py
    │   └── plot_fluid_props_functionofpressure.py
    ├── src/
    │   └── raft/
    │       ├── __init__.py
    │       ├── __main__.py
    │       ├── api.py
    │       ├── logging_utils.py
    │       ├── matprops.py
    │       ├── matrixassembly.py
    │       ├── meshprocessing.py
    │       ├── parseboundaryconditions.py
    │       ├── postprocessing.py
    │       ├── preprocessing.py
    │       ├── py.typed
    │       ├── shapefunctions.py
    │       ├── solvers.py
    │       ├── spyder_runfile.py
    │       ├── subsurface.py
    │       └── writevtk.py
    ├── tests/
    │   ├── __init__.py
    │   ├── test_simulation.py
    │   ├── test_checks.py
    │   ├── test_wellbore.py
    │   ├── test_configs/
    │   └── test_demo.py
    └── uv.lock
```

## Developers
View [CONTRIBUTING.md](./CONTRIBUTING.md) and install the git hooks: `uv run prek install` and `uv run prek install --hook-type pre-push`

### Running Checks
- **Tests:** `uv run pytest`

- **Linting:** `uv run ruff check .`
- **Formatting:** `uv run ruff format .` (use `--check` to verify without modifying)
- **Type checking:** `uv run mypy .`
