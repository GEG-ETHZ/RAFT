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
     - Buoyancy$^1$
 - Heat flow
     - Diffusion
     - Advection (Streamline Upwind Pretrov Galerkin (SUPG) stabilization)

$^1$Buyoancy assumes the Boussinesq approximation. Only the perturbation to density is included. Hydrostatic pressures are not accounted or solved for.

## Running the simulation
Run a simulation:
    uv run -m raft **some-config-file.yaml**

replace the config file with a specific config file name, such as:
    uv run -m raft config_example.yaml

Run the test suite:
    uv run pytest

## Configuration Parameters

The simulator uses a YAML configuration file to define simulation options, material properties, initial conditions, boundary conditions, and numerical controls. Below is the list of parameters and their options.

### `mesh`
* **`file`** *(string, default: `"box3D.msh"`)*: Name or relative path to the Gmsh `.msh` file (Version 2.x format). Meshes may be either B8 or T4 elements, but must use entirely one element. Mixed meshes are currently not supported.
* **`dimensions`** *(integer, default: `3`)*: Spatial dimensions of the analysis (`2` or `3`). 2D is not currently supported.
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


### `material`
* **`solid`**: Rock mass properties. Rock mass properties are defined per region. If regions are defined, RAFT expects a list with a number of entries equal to the number of regions. Region numbers correspond to property indices.
  * **`bulk_mod`** *(float, default: `20.0e9`)*: Solid bulk modulus in Pascals ($\text{Pa}$).
  * **`density`** *(float, default: `2600.0`)*: Rock density in $\text{kg/m}^3$.
  * **`therm_cond`** *(float or list, default: `3.0`)*: Rock thermal conductivity in $\text{W/m}\cdot\text{K}$. Can be a scalar for isotropic conductivity, or a list of length equal to the dimensions for anisotropic conductivity (e.g. `[3.0, 2.0, 0.1]`).
  * **`specific_heat`** *(float, default: `800.0`)*: Rock specific heat capacity in $\text{J/kg}\cdot\text{K}$.
  * **`therm_expansion`** *(float, default: `12.0e-6`)*: Rock volumetric linear thermal expansion coefficient in $1/\text{K}$.
  * **`porosity`** *(float, default: `0.01`)*: Reservoir porosity (dimensionless fraction).
  * **`biot_coeff`** *(float, default: `0.5`)*: Biot-Willis coefficient (dimensionless).
  * **`permeability`** *(float or list, default: `1.0e-16`)*: Permeability in $\text{m}^2$. Can be a scalar for isotropic permeability, or a list of length equal to the dimensions for anisotropic permeability (e.g. `[1.0e-13, 1.0e-11, 1.0e-12]`).
  * **`rotation_x`** *(float, default: `0.0`)*: Rotation angle of the local coordinate system around the x-axis in degrees.
  * **`rotation_y`** *(float, default: `0.0`)*: Rotation angle of the local coordinate system around the y-axis in degrees.
  * **`rotation_z`** *(float, default: `0.0`)*: Rotation angle of the local coordinate system around the z-axis in degrees.

These rotation parameters define a 3D rotation matrix $R = R_x \cdot R_y \cdot R_z$ to rotate the local coordinate system along which the rock mass anisotropy is defined. The rotation is applied to both anisotropic permeability ($K = R \operatorname{diag}(k) R^T$) and anisotropic solid thermal conductivity ($\Lambda_s = R \operatorname{diag}(\lambda_s) R^T$).

* **`fluid`**: Fluid phase properties.
  * **`bulk_mod`** *(float, default: `2.0e9`)*: Fluid bulk modulus in Pascals ($\text{Pa}$).
  * **`density`** *(float, default: `1000.0`)*: Fluid density in $\text{kg/m}^3$.
  * **`therm_cond`** *(float, default: `0.6`)*: Fluid thermal conductivity in $\text{W/m}\cdot\text{K}$.
  * **`therm_expansion`** *(float, default: `200.0e-6`)*: Fluid volumetric thermal expansion coefficient in $1/\text{K}$.
  * **`specific_heat`** *(float, default: `4180.0`)*: Fluid specific heat capacity in $\text{J/kg}\cdot\text{K}$.
  * **`viscosity`** *(float, default: `1.0e-3`)*: Fluid dynamic viscosity in $\text{Pa}\cdot\text{s}$.

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
  * **`"BC_compatible"`: Solve the non-linear transport problem for the steady-state conditions within the domain under the boundary conditions applied at time $t=t_0$. Pressurization is time-dependent and thus neglected. Buoyancy requires an reference temperature and is thus neglected. The solved pressure and temperature fields are applied as initial conditions and reference temperatures.

### `boundary_conditions`
Boundary conditions are specified in groups. Each group can be applied using one of three geometric specification methods: node indices, coordinate proximity, or named faces. Each group can also specify either constant or time-dependent values.

**Boundary Condition Types:**
* **`pressure_fixed`**: Dirichlet boundary conditions for fluid pressure.
* **`temperature_fixed`**: Dirichlet boundary conditions for temperature.
* **`fluid_flux`**: Neumann boundary conditions for fluid flow flux.
* **`temperature_flux`**: Neumann boundary conditions for heat flux.

**Geometry Specification Methods:**
Each boundary condition group must specify nodes using exactly one of the following methods:

* **`nodes`** *(list of integers)*: Direct node IDs where the boundary condition is applied.
* **`coords`** *(list of coordinates)*: List of coordinate points; the node closest to each point will be selected. Each coordinate should be a list matching the spatial dimensions (e.g., `[x, y, z]` for 3D, `[x, y]` for 2D).
* **`faces`** *(list of strings)*: Named boundary faces from preprocessing. Supported face identifiers: `"x_min"`, `"x_max"`, `"y_min"`, `"y_max"`, `"z_min"`, `"z_max"`. All nodes on the specified faces will be included. Use caution when applying Neumann conditions to a face, as surface integrals over the face for a specified flux are not yet implemented. Instead, the values are applied directly into the forcing vector.

**Value Specification:**
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
  * `values` : `[y_ramp, t_ramp]` defines:
        $\quad y(t) =  \frac{t}{t_{ramp}}y_{ramp} \quad \forall t<t_{ramp}$
        $\quad y(t) = y_{ramp} \quad \forall t\geq t_{ramp}$.

**Output Specification:**
*(Optional)* Flow boundary condition groups can optionally be marked with an `tag: "groupname"` entry. If they are marked with a tag, the group pressure, flow, and temperatures will be output to *outflow.log* in the output directory. Outputs from each group are summarized as:
* groupname_pressure: arithmetic mean of pressure over the group nodes. Node that this is not weighted by nodal area, and thus caution should be used when examining this value over pressure faces with non-uniform meshes.
* groupname_flow: sum of flow rates over the group nodes.
* groupname_temperature: flow-weighted average temperature over the group nodes.


### `simulation`
* **`t_start`** *(float, default: `0.0`)*: Start time of the simulation.
* **`t_end`** *(float, default: `100.0`)*: End time of the simulation.
* **`timestep`** *(float, default: `1.0`)*: Simulation timestep.
* **`timestep_min`** *(float, default: `1.0`)*: Minimum allowed timestep.
* **`timestep_max`** *(float, default: `1.0`)*: Maximum allowed timestep.
* **`output_frequency`** *(integer, default: `1`)*: Controls how often outputs are written (every $N$ timesteps).
* **`output_vtk`** *(integer/boolean, default: `0`)*: If set to `1`, enables writing mesh and solution fields to `.vtk` files for visualization.
* **`output_directory`** *(string, optional)*: Custom output directory path. Defaults to `outputs/` under the repository root.
* **`rate_order`** *(integer, default: `1`)*: The approximation order of rate of change (typically `1`).
* **`advection_on`** *(integer/boolean, default: `0`)*: Set to `1` to enable heat advection; `0` to disable.
* **`pressurization_on`** *(integer/boolean, default: `0`)*: Set to `1` to enable thermal pressurization coupling; `0` to disable.
* **`buoyancy_on`** *(integer/boolean, default: `0`)*: Set to `1` to enable density buoyancy effects; `0` to disable.
* **`steady_pressure`** *(integer/boolean, default: `0`)*: Set to `1` to solve steady-state pressure equations; `0` for transient.
* **`steady_temperature`** *(integer/boolean, default: `0`)*: Set to `1` to solve steady-state temperature equations; `0` for transient.
* **`rel_error_T`** *(float, default: `1.0e-2`)*: Relative error tolerance used for temperature extrapolation checks.

## Prerequisites
- Python version specified in `.python-version`
- uv (if not installed - see below)

### Installation
1. [install uv](https://docs.astral.sh/uv/getting-started/installation/) <!--NOTE: link to Wiki for uv-->
1. run `uv sync` to install dependencies. Re-run whenever dependencies in `pyproject.toml` change
1. execute a script via: `uv run some_script.py`

## Project Structure
```
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
