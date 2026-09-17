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
- [Running the simulation](#running-the-simulation)
- [Theory](#theory)
- [Configuration Parameters](#configuration-parameters)
  - [`mesh`](#mesh)
  - [`regions`](#regions)
  - [`fractures` (optional field)](#fractures-optional-field)
  - [`material`](#material)
  - [`initial_conditions`](#initial_conditions)
  - [`boundary_conditions`](#boundary_conditions)
  - [`simulation`](#simulation)
- [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Project Structure](#project-structure)
- [Developers](#developers)
  - [Branching Strategy](#branching-strategy)
  - [Running Checks](#running-checks)

## Running the simulation
Run a simulation:
    `uv run -m raft **some-config-file.yaml**`

replace the config file with a specific config file name, such as:
    `uv run -m raft config_example.yaml`

Run the test suite:
    `uv run pytest`

Creating meshes: [GMSH](https://gmsh.info/)

Viewing results files: [Paraview](https://www.paraview.org/)

Config files are read from the "config" directory. Meshes are made via [GMSH](https://gmsh.info/) and should be exported to .msh files in Version 2 ASCII format. Results files are saved by default to the "outputs" directory in a folder under the config name. The results files are output as .vtk files viewable in [Paraview](https://www.paraview.org/), and an outflow.log file which includes times, pressure, flow rates, and temperatures. See the [boundary conditions](#boundary_conditions) section for how to tag which boundary are output.

## Theory
RAFT idealizes the subsurface as a continuous, fully-saturated, single-phase porous medium and solves a coupled thermo-hydraulic (pressure/temperature) finite element problem with a compressible, pressure- and temperature-dependent fluid equation of state. The governing equations, fluid EOS, finite element discretization, fracture coupling, mesh cutting algorithm, stabilization, and time-integration schemes are documented in [theory.md](theory.md).

## Configuration Parameters

The simulator uses a YAML configuration file to define simulation options, material properties, initial conditions, boundary conditions, and numerical controls. Below is the list of parameters and their options.

### `mesh`
* **`file`** *(string, default: `"box3D.msh"`)*: Name or relative path to the Gmsh `.msh` file (Version 2.x format). Meshes may be either B8 or T4 elements in 3D or Q4 or T3 in 2D. Meshes can consist of any combination of elements, so long as the element boundaries form a continuous approximation.
* **`dimensions`** *(integer, default: `3`)*: Spatial dimensions of the analysis (`2` or `3`).
* **`regions`** *(subcategories)*: See the `regions` section for specifying regions, or specify the subcategory `"list":` *(str)* and provide the path to a .txt file with a list of corresponding region for each element.
* **`fractures`** *(subcategories)*: See the `fractures` section for specifications.

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

If regions are not defined, homogeneity is assumed (all elements are assigned to region 0).

**Fracture Definition:**
Each fracture specification requires three fields:
* **`shape`** *(string)*: Type of geometric boundary. These shapes are only used in 3D, in 2D all fractures are represented as lines. Supported shapes:
  * `"ellipse"`
  * `"rectangle"`
* **`values`** *(list)*: $[[x_c, y_c, z_c ], [a,b], [\theta_1, \theta_2]]$
  * $[x_c, y_c, z_c ]$ *(list[float], no default)*: coordinates of the plane center. In 2D analysis, only $x_c$ and $y_c$ are considered.
  * $[a,b]$ *(list[float], no default)*: extents. For the ellipse, $a$ = major axis distance, $b$ = minor axis distance. For the rectangle, $a$ = half-length, $b$ = half-width. In 2D, $a$ is taken as the fracture half-length.
  * $[\theta_1, \theta_2]$ *(list[float], no default)*: plane angles. $\theta_1$ = plane angle with respect to the x-axis, $\theta_2$ = plane angle with respect to the y-axis.

Hydraulic and thermal properties (`aperture`, `nusselt`) are not specified here; they are defined per-fracture under `material.fracture` (see the `material` section below and [Fracture Coupling](theory.md#fracture-coupling) for their role in the governing equations).

> **IMPORTANT NOTE FOR FRACTURE SPECIFICATIONS**: The fracture mesh cutting algorithm can fail! Always check the quality of the cut elements in `domain.vtk`. If the algorithm fails, it can usually be solved by refining the mesh around fractures and/or junctions.

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
    * **Pure thermal behabiour:** Set to `-1` to instead use a temperature-dependent viscosity with the form $\mu = A e^{-B/T + CT + DT^2}$. Curently only works for water. For water, $ A = 1.856·10−11 \text{mPa}\cdot \text{s}, B = 4209 \text{K}, C = 0.04527 \text{K}^{-1}$ and $D = -3.376\times 10^{-5} \text{K}^{-2}$.


> **IMPORTANT NOTE FOR NONLINEAR FLUID PROPERTIES**: When using non-linear material properties, the magnitude of pressure and temperature matter. Temperature __must__ be specified in Kelvin, while pressure boundary conditions should be specified above the fluid's saturation point. Temperature and pressure are clipped to $274$–$1900\,\text{K}$ and $0.1$–$700\,\text{MPa}$ before the lookup, to keep unconverged solver iterates within CoolProp's valid range; pressures should stay comfortably above the lower bound, since it sits close to water's saturation curve at higher temperatures.

* **`fracture`** *(required if `fractures` are defined in `mesh`)*: Hydraulic and thermal properties of the fracture(s), used in the [fracture governing equations](theory.md#fracture-coupling). Each property is a list indexed the same way as fractures are listed under `mesh.fractures`.
  * **`aperture`** *(list[float], no default)*: Hydraulic aperture of the fracture, $w$, used in the cubic-law in-plane flow and leak-off terms.
  * **`nusselt`** *(list[float], no default)*: Nusselt number, $Nu$, controlling the fracture-matrix convective heat transfer coefficient.


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

### `boundary_conditions`
`boundary_conditions` is a list of groups at the root of the config. Each group applies to one geometry feature (specified using one of four geometric methods: node indices, coordinate proximity, named faces, or lines), and can carry a `hydraulics` block specifying pressure or fluid flux, a `thermal` block specifying temperature or heat flux, or both.

**Sample boundary condition group**
```yaml
- coords: [[x1, y1, z1], [x2, y2, z2]]  (geometry specification - required)
  tag: "inlet"                          (label - optional)
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

* **`nodes`** *(list of integers)*: Direct node IDs where the boundary condition is applied.
* **`coords`** *(list of coordinates)*: List of coordinate points; the mesh node closest to each point will be selected. Each coordinate should be a list matching the spatial dimensions (e.g., `[x, y, z]` for 3D, `[x, y]` for 2D).
* **`faces`** *(list of strings)*: Named boundary faces from preprocessing. Supported face identifiers: `"x_min"`, `"x_max"`, `"y_min"`, `"y_max"`, `"z_min"`, `"z_max"`. All nodes on the specified faces will be included. Fluxes are integrated over the face.
  * Fluid fluxes: Specify `values` in $(\text{m}^3/ \text{s}) / \text{m}^2$.
  * Temperature fluxes: Specify `values` in $\text{W} / \text{m}^2$.
* **`lines`** *(list of `[start, end]` point pairs)*: Each pair defines a line segment (points matching the spatial dimensions, as in `coords`). The nodes and connectivity are found by snapping to the path of existing mesh edges closest to that segment. The line is snipped to the domain boundaries. If the exact path does not exist among the element edges, the line snaps to the shortest edge path connecting the start and end node. This throws a warning and reports the true vs. prescribed line lengths. You can view the nodes with `line` boundary conditions in `domain.vtk` under the `line_BC_nodes` field. Fluxes are integrated over the prescribed line.
  * Fluid fluxes: Specify `values` in $(\text{m}^3/ \text{s}) / \text{m}$.
  * Temperature fluxes: Specify `values` in $\text{W} / \text{m}$.

**Field Blocks (At least one required):**
* **`hydraulics`**: controls fluid pressure/flux at this group.
  * `type: "pressure"`: Dirichlet condition (fixed pressure).
  * `type: "flux"`: Neumann condition (fixed fluid flux).
* **`thermal`**: controls temperature/heat flux at this group.
  * `type: "temperature"`: Dirichlet condition (fixed temperature).
  * `type: "flux"`: Neumann condition (fixed heat flux).

A group with only `hydraulics` affects pressure alone; a group with only `thermal` affects temperature alone; a group with both applies both conditions over the same geometry. Each block carries its own `function`/`values` (and, for `hydraulics`, an optional `constraints` block — see [Wellbore Constraints](#wellbore-constraints) below).

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

**Label Specification (Optional):**
Flow boundary condition groups can optionally be marked with an `tag: "groupname"` entry. If they are marked with a tag, the group pressure, flow, and temperatures will be output to *outflow.log* in the output directory. Outputs from each group are summarized as:
* groupname_pressure: arithmetic mean of pressure over the group nodes. Note that this mean is not weighted by nodal area, and thus caution should be used when examining this value over pressure faces with non-uniform meshes.
* groupname_flow: sum of flow rates over the group nodes.
* groupname_temperature: flow-weighted average temperature over the group nodes.

**Constraints**
Constraints are limits placed on a `hydraulics` block with `type: "flux"`. A block with `constraints` cannot use the `"gradient"` function.

* **`maximum_pressure`** / **`minimum_pressure`** *(float, optional)*: Pressure limits at the wellbore nodes. Either may be omitted to leave that side unbounded.

The group follows its specified fluid flux until any node's pressure reaches `maximum_pressure` or `minimum_pressure`, at which point the group switches to a fixed-pressure condition according to which bound was violated. It switches back to the original flux condition once the flux implied by holding that pressure would exceed the originally specified flux target. Each group is evaluated collectively - any one pressure node exceeding the constraints triggers the swap. For fluid flux constraints, line and face geometry types are integrated over the line/face and the prescribed integrated total flux is compared against the current integrated total flux.

Only one switch is permitted per timestep to avoid oscillation between the two states, so tighten the timestep if a run needs faster response at the switch point. Wellbore groups are included when tagging groups for `outflow.log`.

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
* **`advection_on`** *(integer/boolean, default: `0`)*: Set to `1` to enable heat advection; `0` to disable.
* **`gravity`** *(float, default: `0.0`)*: Set the magnitude of the acceleration due to gravity in $\text{m}/\text{s}^2$, typically `-9.81`. Gravity is assumed to act in the z-axis in 3D and the y-axis in 2D.

* **`steady_pressure`** *(integer/boolean, default: `0`)*: Set to `1` to solve steady-state pressure equations; `0` for transient.
* **`steady_temperature`** *(integer/boolean, default: `0`)*: Set to `1` to solve steady-state temperature equations; `0` for transient.
* **`axisymmetric_on`** *(integer/boolean, default: `0`)*: Set to `1` to solve the axisymmetric case. Only valid in 2D analysis.

Numerical solver controls
* **`solver_type`** *(string, default: `"linearized"`)*: Type of time integration solver. Options are `"linearized"` or `"fully_coupled"`. The `fully_coupled` solver solves both pressure and temperature monolithically for the current timestep. The `linearized` solver uses an operator splitting scheme, extrapolating temperature to the current timestep, solving pressure with the extrapolated temperature, then updating temperature with the new pressure.  See [Time Discretization and Solution Scheme](theory.md#time-discretization-and-solution-scheme) for more details.
* **`rel_error_T`** *(float, default: `1.0e-2`)*: Relative error tolerance used for temperature extrapolation checks (`"linearized"` solver only).
* **`iterative_controls`** *(dictionary)*: Non-linear Newton-Raphson solver settings. Because fluid properties are nonlinear functions of pressure and temperature, both solver types use these controls: `"fully_coupled"` for its single monolithic solve, and `"linearized"` for each of its two per-field (pressure, then temperature) sub-solves.
  * **`NR_tol`** *(float, default: `1.0e-6`)*: Newton-Raphson residual tolerance.
  * **`iter_max`** *(integer, default: `100`)*: Maximum number of iterations per timestep.
  * **`aitken_relaxation`** *(integer/boolean, default: `1`)*: Turn on (`1`) or off (`0`) Aitken's $\Delta^2$ relaxation for non-linear convergence.
  * **`NR_step`** *(float, default: `1.0`)*: Newton-Raphson step/line-search size; scales each iteration's solution update before any Aitken relaxation is applied.

## Prerequisites
- Python version specified in `.python-version`
- uv (if not installed - see below)

### Installation
1. [install uv](https://docs.astral.sh/uv/getting-started/installation/) <!--NOTE: link to Wiki for uv-->
1. run `uv sync`
1. (Contributors) install the git hooks: `uv run prek install` and `uv run prek install --hook-type pre-push` — see [Setup](CONTRIBUTING.md#setup) for what they enforce
1. execute a script via: `uv run some_script.py`

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
    │   ├── debug_fracture_angles.py
    │   ├── plot_fluid_props.py
    │   └── plot_fluid_props_functionofpressure.py
    ├── src/
    │   └── raft/
    │       ├── __init__.py
    │       ├── __main__.py
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
