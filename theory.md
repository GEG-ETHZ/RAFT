# Theory — Governing Equations and Numerical Methods

RAFT idealizes the subsurface as a continuous, fully-saturated, single-phase porous medium and solves a coupled thermo-hydraulic (pressure/temperature) finite element problem. This section summarizes the strong and weak forms, spatial and temporal discretization, stabilization, and the linearized solution scheme actually implemented by the solvers in [`src/raft/matrixassembly.py`](src/raft/matrixassembly.py) and [`src/raft/solvers.py`](src/raft/solvers.py).

## Table of Contents
- [Symbol Glossary](#symbol-glossary)
- [Strong Form Governing Equations](#strong-form-governing-equations)
- [Weak Form and Finite Element Discretization](#weak-form-and-finite-element-discretization)
- [Fracture Coupling](#fracture-coupling)
- [Fracture Mesh Cutting Algorithm](#fracture-mesh-cutting-algorithm)
- [Advection Stabilization (SUPG)](#advection-stabilization-supg)
- [Time Discretization and Solution Scheme](#time-discretization-and-solution-scheme)
- [Adaptive Timestep Control](#adaptive-timestep-control)
- [Tangent Matrices](#tangent-matrices)

## Symbol Glossary

| Symbol | Definition | Unit | Type |
|---|---|---|---|
| $p$ | Fluid pressure | $Pa$ | Primary variable |
| $T$ | Temperature | $^\circ C$ | Primary variable |
| $\rho_f$ | Fluid density | $kg/m^3$ | Material property |
| $\rho_s$ | Rock matrix density | $kg/m^3$ | Material property |
| $M$ | Biot storage modulus | $Pa$ | Derived property |
| $\boldsymbol{k}$ | Permeability tensor | $m^2$ | Material property |
| $\mu$ | Viscosity | $Pa\cdot s$ | Material property |
| $\vec{g}$ | Gravity vector | $m/s^2$ | Boundary condition |
| $\alpha_f$ | Volumetric coefficient of thermal fluid expansion | $^\circ C^{-1}$ | Material property |
| $\alpha_s$ | Volumetric coefficient of thermal matrix expansion | $^\circ C^{-1}$ | Material property |
| $\alpha_e$ | Effective coefficient of pressurization | $^\circ C^{-1}$ | Derived property |
| $\phi$ | Porosity | - | Material property |
| $\beta$ | Biot effective pressure coefficient | - | Material/derived property |
| $K_d$ | Drained bulk modulus of porous media | $Pa$ | Material/derived property |
| $K_s$ | Bulk modulus of solid matrix | $Pa$ | Material property |
| $K_f$ | Bulk modulus of fluid | $Pa$ | Material property |
| $Q_i$ | Volumetric fluid injection per injection point | $m^n/s$ | Boundary condition |
| $\vec{q}$ | Darcy flux | $m^3/s/m^2$ | Derived property |
| $c_{pf}$ | Specific heat capacity of fluid | $J/kg/^\circ C$ | Material property |
| $c_{ps}$ | Specific heat capacity of matrix | $J/kg/^\circ C$ | Material property |
| $m_e$ | Effective heat capacity | $J/m^3/^\circ C$ | Derived property |
| $\boldsymbol{\lambda}_f$ | Fluid thermal conductivity tensor | $W/m/^\circ C$ | Material property |
| $\boldsymbol{\lambda}_s$ | Matrix thermal conductivity tensor | $W/m/^\circ C$ | Material property |
| $\boldsymbol{\lambda}_e$ | Effective thermal conductivity tensor | $W/m/^\circ C$ | Derived property |
| $\Omega$ | Domain | | Domain |
| $\Gamma$ | Boundary | | Boundary |

## Strong Form Governing Equations
Flow is governed by conservation of mass with Darcy's law, assuming a fully-saturated, single-phase medium:

$$\frac{1}{M}\dot{p} = \nabla\cdot\left(-\frac{\boldsymbol{k}_f}{\mu}(\nabla p + \rho_f \vec{g})\right) + \alpha_e \dot{T} + Q_i\delta(\vec{X}_i) \quad \forall \vec{X}\in\Omega$$

Buoyancy is introduced through a Boussinesq approximation in which the fluid density is linearized about the initial (in-situ) temperature, $\rho_f = \rho_{f0}(1-\alpha_f \Delta T)$ with $\Delta T = T - T_0$. Substituting and neglecting $\nabla\cdot\rho_{f0}\vec{g}$ (which is not a source of flow) gives the working form of conservation of mass:

$$\frac{1}{M}\dot{p} = -\nabla\cdot\left(\frac{\boldsymbol{k}_f}{\mu}\nabla p\right) + \frac{\rho_{f0}\alpha_f \boldsymbol{k}_f}{\mu}\nabla\cdot\left(\Delta T\,\vec{g}\right) + \alpha_e \dot{T} + Q_i\delta(\vec{X}_i)$$

The derived hydraulic (poroelastic) parameters are

$$\beta = 1-\frac{K_d}{K_s}, \qquad \alpha_e = \beta\alpha_s + \phi(\alpha_f - \alpha_s), \qquad \frac{1}{M} = \frac{\phi}{K_f} + \frac{\beta-\phi}{K_s}, \qquad K_d = (1-\beta)K_s$$

Temperature is governed by conservation of energy assuming local thermal equilibrium between rock and fluid (a single temperature field for the porous medium — not valid across highly conductive fracture apertures, where a separate fracture/matrix convective exchange is used instead, see [`fractures`](README.md#fractures-optional-field)):

$$m_e \dot{T} = \rho_f c_f \frac{\boldsymbol{k}_f}{\mu}\nabla p\cdot\nabla T - \lambda_e \nabla^2 T \quad \forall \vec{X}\in\Omega$$

with effective thermal parameters mixed by porosity

$$m_e = \rho_s c_s(1-\phi) + \rho_f c_f \phi, \qquad \lambda_e = \lambda_s(1-\phi) + \lambda_f \phi$$

The coupled problem seeks $p(\vec{X},t)$ and $T(\vec{X},t)$ satisfying the above everywhere in $\Omega$, subject to initial conditions $p_0(\vec{X})$, $T_0(\vec{X})$, Dirichlet boundary conditions $\overline{p}(t)$, $\overline{T}(t)$ on $\Gamma_D$, and Neumann (flux) boundary conditions on $\Gamma_N$:

$$\vec{q} = -\frac{\boldsymbol{k}_f}{\mu}\nabla p = \overline{\vec{q}}(t), \qquad \vec{q}_T = -\lambda_e \nabla T = \overline{\vec{q}}_T(t)$$

where $\Gamma = \Gamma_N \cup \Gamma_D$, $\Gamma_N \cap \Gamma_D = \emptyset$. These correspond directly to the `type: "pressure"`/`type: "temperature"` (Dirichlet) and `type: "flux"` (Neumann) [boundary condition types](README.md#boundary_conditions).

## Weak Form and Finite Element Discretization
Multiplying by admissible variations $\delta p$, $\delta T$, integrating over the domain, and applying the divergence theorem to the Laplacian terms (the Dirichlet contribution vanishes since $\delta p = \delta T = 0$ there, and the Neumann contribution reduces to the prescribed flux) yields the weak forms:

$$0 = -\int_\Omega \delta p\,\frac{1}{M}\dot{p}\,d\Omega + \oint_{\Gamma_N}\delta p\,\overline{\vec{q}}\cdot\vec{n}\,d\Gamma_N + \int_\Omega \nabla\delta p\cdot\left(\frac{\boldsymbol{k}_f}{\mu}\nabla p\right)d\Omega + \int_\Omega \delta p\,\frac{\rho_{f0}\alpha_f}{\mu}\boldsymbol{k}_f\vec{g}\cdot\nabla\Delta T\,d\Omega + \int_\Omega \delta p\,\alpha_e \dot{T}\,d\Omega + \int_\Omega \delta p\,Q_i\delta(\vec{X}_i)\,d\Omega$$

$$0 = -\int_\Omega \delta T\,m_e\dot{T}\,d\Omega + \int_\Omega \delta T\,\rho_f c_f \frac{\boldsymbol{k}_f}{\mu}\nabla p\cdot\nabla T\,d\Omega + \oint_{\Gamma_N}\delta T\,\overline{\vec{q}}_T\cdot\vec{n}\,d\Gamma_N + \int_\Omega \nabla\delta T\cdot\lambda_e\nabla T\,d\Omega$$

The fields are discretized with standard isoparametric shape functions, $p^h = \textbf{N}(\vec{X})\textbf{p}(t)$ and $T^h = \textbf{N}(\vec{X})\textbf{T}(t)$, with gradients $\nabla p^h = \textbf{B}(\vec{X})\textbf{p}(t)$, $B_{ij} = \partial N_j/\partial x_i$ ([`src/raft/shapefunctions.py`](src/raft/shapefunctions.py)). Pressure and temperature may use different-order shape functions. Substituting into the weak forms gives element residual vectors $\textbf{R}_p = 0$ and $\textbf{R}_T = 0$, assembled in [`src/raft/matrixassembly.py`](src/raft/matrixassembly.py):

$$\textbf{R}_p = \textbf{F}_S + \textbf{F}_{\Gamma p} + \textbf{F}_k + \textbf{F}_b + \textbf{F}_z + \textbf{F}_w = 0$$

| Term | Definition | Description |
|---|---|---|
| $\textbf{F}_S$ | $-\textbf{C}_p\dot{\textbf{p}}$, $\textbf{C}_p = \int_\Omega \textbf{N}^\top \frac{1}{M}\textbf{N}\,d\Omega$ | Storage vector |
| $\textbf{F}_{\Gamma p}$ | $\oint_{\Gamma_N}\textbf{N}^\top \overline{\vec{q}}\cdot\vec{n}\,d\Gamma_N$ | External flux vector |
| $\textbf{F}_k$ | $\int_\Omega \textbf{B}^\top \frac{\boldsymbol{k}_f}{\mu}\textbf{B}\,\textbf{p}\,d\Omega$ | Pressure diffusion (Darcy) vector |
| $\textbf{F}_b$ | $\int_\Omega \textbf{N}^\top \frac{\rho_{f0}\alpha_f}{\mu}\boldsymbol{k}_f \textbf{g}^\top \textbf{B}\,\Delta\textbf{T}\,d\Omega$ | Buoyancy vector |
| $\textbf{F}_z$ | $\int_\Omega \textbf{N}^\top \alpha_e \textbf{N}\dot{\textbf{T}}\,d\Omega$ | Pressurization vector |
| $\textbf{F}_w$ | $\int_\Omega \textbf{N}^\top Q_i\delta(\vec{X}_i)\,d\Omega$ | Point flux source vector |

$$\textbf{R}_T = \textbf{F}_C + \textbf{F}_A + \textbf{F}_{\Gamma T} + \textbf{F}_\lambda = 0$$

| Term | Definition | Description |
|---|---|---|
| $\textbf{F}_C$ | $-\textbf{C}_T\dot{\textbf{T}}$, $\textbf{C}_T = \int_\Omega \textbf{N}^\top m_e \textbf{N}\,d\Omega$ | Heat storage vector |
| $\textbf{F}_A$ | $\int_\Omega \textbf{N}^\top \rho_f c_f \frac{\boldsymbol{k}_f}{\mu}\textbf{B}^\top\textbf{p}^\top\textbf{B}\textbf{T}\,d\Omega$ | Heat advection vector |
| $\textbf{F}_{\Gamma T}$ | $\oint_{\Gamma_N}\textbf{N}^\top \overline{\vec{q}}_T\cdot\vec{n}\,d\Gamma_N$ | External heat flux vector |
| $\textbf{F}_\lambda$ | $\int_\Omega \textbf{B}^\top \lambda_e \textbf{B}\textbf{T}\,d\Omega$ | Heat conduction vector |

## Fracture Coupling
Fractures are modelled as lower-dimensional manifolds (line elements in 2D) embedded in, and conforming to, the surrounding matrix mesh, rather than as full-dimensional low-permeability regions. Each fracture element solves a reduced-dimensional pressure/temperature system restricted to the fracture plane, and exchanges flow and heat with the matrix elements it cuts through. This is assembled as a coupled fracture/matrix system in [`src/raft/matrixassembly.py`](src/raft/matrixassembly.py) and requires a conforming cut of the bulk mesh, produced in [`src/raft/meshprocessing.py`](src/raft/meshprocessing.py) (currently 2D only — see [`fractures`](README.md#fractures-optional-field)).

**Governing equations.** The fracture void is treated as fully fluid-filled (no solid phase), so the mass and energy balances are the [bulk equations](#strong-form-governing-equations) with the solid/Biot contributions dropped, restricted to the fracture's local tangential coordinate $s$ (i.e. $\nabla \to \partial/\partial s = \vec{u}\cdot\nabla$ for fracture tangent $\vec{u}$), an effective in-plane permeability from the cubic law, and sink/source terms exchanging flow and heat with the matrix:

$$\frac{w}{K_f}\dot{p}_f = \frac{\partial}{\partial s}\left(\frac{w^3}{12\mu}\frac{\partial p_f}{\partial s}\right) + \frac{w^3}{12\mu}\rho_{f0}\alpha_f\,g_s\,\frac{\partial \Delta T_f}{\partial s} - \underbrace{\frac{w}{12\mu}(p_f - p_m)}_{\text{leak-off}} + Q_i\delta(\vec{X}_i)$$

$$w\rho_f c_f\,\dot{T}_f = w\rho_f c_f \frac{w^2}{12\mu}\frac{\partial p_f}{\partial s}\frac{\partial T_f}{\partial s} + \lambda_f \frac{\partial^2 T_f}{\partial s^2} - \underbrace{h(T_f - T_m)}_{\text{convective exchange}}$$

where $w$ is the hydraulic aperture, $K_f$ is the fluid bulk modulus (fracture storage is pure fluid compressibility — the fracture walls are treated as rigid, so there is no analogue of the matrix's Biot storage modulus $M$ or pressurization term $\alpha_e\dot T$), $g_s = \vec{g}\cdot\vec{u}$ is the gravity component along the fracture, $\lambda_f$ is the fluid thermal conductivity, and $p_m$/$T_m$ are the matrix pressure/temperature at the fracture face. The **cubic law** gives the effective in-plane (longitudinal) permeability $k_f = w^2/12$, so in-plane Darcy flux is $\vec{q}_\parallel = -\frac{w^3}{12\mu}\nabla p$. **Leak-off** to the matrix uses the same aperture-derived permeability over the half-aperture, $q_\perp = \frac{w}{12\mu}(p_f - p_m)$. **Heat exchange** with the matrix is modeled as a convective boundary condition — rather than assuming local thermal equilibrium across the aperture, which is invalid in highly conductive fractures — as $q_{T\perp} = h(T_f - T_m)$ with $h = \frac{Nu\cdot\lambda_f}{2w}$, where $Nu$ is the Nusselt number (`material.fracture.nusselt`). Steady, fully-developed laminar flow gives $Nu \approx 3.66$–$4.36$ (Bejan, 2013), higher during the thermal transient. Local thermal equilibrium can be enforced by setting an arbitrarily high Nusselt number.

Discretizing with the same shape functions as the bulk mesh gives fracture storage/conductance/capacity matrices analogous to $\textbf{C}_p$, $\textbf{K}_{Dp}$, $\textbf{C}_T$, $\textbf{K}_{DT}$ (using $w/K_f$, $w^3/(12\mu)$, $\rho_f c_f w$, and $\lambda_f w$ respectively in place of the bulk integrands), plus fracture–matrix coupling matrices built from the leak-off and $h$ terms above, all assembled in `integrate_linear_matrices` (`is_fracture=True` pass). Heat advection (including SUPG stabilization) is likewise assembled along the fracture. Fracture hydraulic and thermal properties (`aperture`, `nusselt`) are set per-fracture in [`material.fracture`](README.md#material).

## Fracture Mesh Cutting Algorithm
Fracture coupling requires a mesh that conforms to each fracture line, so RAFT cuts an ordinary matrix mesh to fit (2D only).

* **Level sets.** Fracture geometry is calculated by level sets, a normal and tangential level set describe the signed distance from the fracture.
* **Fracture element size follows the matrix mesh.** Fracture nodes aren't placed at any independent resolution — a new node is only added where the fracture line actually crosses an edge of the existing matrix mesh (a sign change in $\phi$ between the edge's two endpoints). So the fracture mesh has exactly as many elements as matrix element edges it passes through, and its element size locally matches the surrounding matrix mesh.
* **Snapping tolerance.** To prevent sliver elements, fracture nodes snap to the existing matrix node locations when they are close. This can create local deiviations from the linear fracture geomery. The tolerance is determined by the lesser of a domain scale and a local mesh scale. The domain scale is $\approx 0.005\times$ the domain's characteristic length. The mesh scale is the 1st percentile of the local matrix edge length found in a strip along the fracture line. To improve the geometry of the fracture representation and prevent local deviations, use a finer mesh along the fracture line.
* **Junction placement.** Fracture junctions are placed before any cutting. If the intersection isn't already a mesh node, the containing matrix element is located and a new node is inserted there, splitting that element — so any two crossing fractures always share a common node from the start, before either is cut in.
* **Matrix cutting.** Matrix elements are split and cut, preserving Q4 geometry when possible, but reverting to T3s when not. This may affect accuracy. Review the mesh quality heuristics (Scaled Jacobian, Skewness, Aspect Ratio). Fracture typically require finer meshes around them to fully resolve the normal boundary layers.
* **Matrix–fracture correspondence.** Once cutting is complete, each fracture element is matched to the corresponding matrix elements which is the fracture-matrix coupling occurs.

## Advection Stabilization (SUPG)
Standard Galerkin FEM produces spurious oscillations for advection-dominated heat transport (Brooks & Hughes, 1982). RAFT applies Streamline Upwind Petrov–Galerkin (SUPG) stabilization to the temperature residual, adding

$$\textbf{F}_{stab} = \int_\Omega \textbf{B}^\top\cdot \vec{a}\,\tau\,\mathcal{R}(T^h)\,d\Omega, \qquad \vec{a} = \rho_{f0} c_{pf}\frac{\boldsymbol{k}_f}{\mu}\textbf{B}\textbf{p}$$

with stabilization coefficient (Shakib, Hughes & Zohan, 1991) and element Péclet number

$$\tau = \frac{\ell^e}{2\lVert\vec{a}\rVert}\left(1 + \frac{9}{Pe^2} + \left(\frac{\ell^e}{2\lVert\vec{a}\rVert}h\right)^2\right)^{-0.5}, \qquad Pe = \frac{\ell^e \lVert\vec{a}\rVert}{\lambda_e}$$

where $\ell^e$ is the characteristic element size and $\mathcal{R}(T^h) = \rho_f c_f \frac{\boldsymbol{k}_f}{\mu}\textbf{B}^\top\textbf{p}^\top\textbf{B}\textbf{T} - \lambda_e\textbf{G}\textbf{T} - m_e\textbf{N}\dot{\textbf{T}}$ is the strong-form residual of the discretized temperature field ($\textbf{G}$ is the matrix of second shape function derivatives, which vanishes identically for linear elements). This is enabled per-simulation via [`advection_on`](README.md#simulation).

## Time Discretization and Solution Scheme
Time derivatives use a general multistep discretization $\dot{\textbf{p}} = \sum_{i=0}^k c_{vi}\textbf{p}^{n-i}$ (and analogously for $\textbf{T}$), implemented in [`rate_coefficients`](src/raft/solvers.py). For first-order ($k=1$, the default [`rate_order`](README.md#simulation)), $c_{v0} = 1/\Delta t$, $c_{v1} = -1/\Delta t$. Second-order backwards difference time discretization can also be specified. RAFT implements two time-integration solvers, selected via [`solver_type`](README.md#simulation):

**`"linearized"` (default) — pressure-first, semi-implicit, extrapolated linearization**, in [`lin_extrap_time_integration`](src/raft/solvers.py). Nonlinear coupling terms are evaluated using a temperature extrapolated from previously-solved states, $\hat{\textbf{T}}^n$, rather than the (unknown) current-timestep temperature. The extrapolation order is the same as the time discretization order (1st linear or 2nd quadratic). In the linear case, $\hat{\textbf{T}}^n = (1+ \frac{\Delta t^n}{\Delta t^{n-1}})\textbf{T}^{n-1} - \frac{\Delta t^n}{\Delta t^{n-1}}\textbf{T}^{n-2}$. This linearizes the pressure equation, which is solved first as a single linear system:

$$\left[\textbf{C}_p \frac{1}{\Delta t} - \textbf{K}_{Dp}\right]\textbf{p}^n = -\textbf{C}_p c_{v1}\textbf{p}^{n-1} + \textbf{F}_{\Gamma p} + \textbf{F}_b(\hat{\textbf{T}}^n) + \textbf{F}_z(\hat{\textbf{T}}^n) + \textbf{F}_w \equiv \textbf{A}_p \textbf{p}^n = \textbf{b}_p$$

The resulting $\textbf{p}^n$ is substituted directly into the (now-linear) temperature system, which is solved second:

$$\left[c_{v0}\textbf{C}_T - \textbf{K}_{AT}(\textbf{p}^n) - \textbf{K}_{DT} - \textbf{K}_{stabT}(\textbf{p}^n)\right]\textbf{T}^n = \textbf{F}_{\Gamma T} - c_{v1}\textbf{C}_T\textbf{T}^{n-1} \equiv \textbf{A}_T \textbf{T}^n = \textbf{b}_T$$

Pressure is solved first (rather than temperature) because at typical reservoir permeabilities ($k_f \gtrsim 10^{-16}\,m^2$) the characteristic timescale of pressure diffusion (hours–days) is much shorter than that of thermal transport (weeks-years), so pressure is expected to evolve faster than temperature between timesteps. This is a one-pass solve. The extrapolated temperature is checked against the actually-solved temperature and convergence is reported against [`rel_error_T`](README.md#simulation). If the error between the expolated and solved temperatures is too large, the timestep is reduced in proportion to the error and the timestep is repeated until acceptable error is achieved or the minimum timestep is reached.

**`"fully_coupled"` — monolithic Newton-Raphson**, in [`fully_coupled_time_integration`](src/raft/solvers.py). Pressure and temperature are assembled into a single combined system and evaluated at $t^n$ (fully implicit), then solved iteratively with Newton-Raphson using the tangent matrices below, subject to [`iterative_controls`](README.md#simulation): `NR_tol`, `iter_max`, and optional Aitken $\Delta^2$ relaxation ([`_aitken_relax`](src/raft/solvers.py)) to accelerate/stabilize convergence between iterations. If convergence is not achieved, the timestep is reduced and the non-linear solve is attempted again. If the minimum timestep is reached and the solver is unable to find a converged solution, the simulation ends.

## Adaptive Timestep Control
The timestep is adapted every step based on solution error/convergence, implemented in [`_update_time`](src/raft/subsurface.py) and driven from the main loop in [`main`](src/raft/subsurface.py). The three most recent timesteps are tracked as a rolling history $\Delta t = [\Delta t^n, \Delta t^{n-1}, \Delta t^{n-2}]$, since the second-order rate/extrapolation coefficients above need timestep history beyond just the current step.

"Convergence" is checked differently per [`solver_type`](README.md#simulation):
* `"linearized"`: the relative difference between the extrapolated temperature $\hat{\textbf{T}}^n$ used to solve pressure and the actual solved temperature $\textbf{T}^n$ is compared against [`rel_error_T`](README.md#simulation). This is a step-quality check on the linearization, not an iterative nonlinear solve.
* `"fully_coupled"`: the Newton-Raphson residual norm falling below `NR_tol` within `iter_max` iterations (see [`iterative_controls`](README.md#simulation)).

If a step does not converge, and the timestep is already at `timestep_min`, the two solvers diverge in behavior: `"linearized"` accepts the step anyway (with a warning) and proceeds, since it has no iterative fallback; `"fully_coupled"` aborts the simulation. Otherwise, the timestep is cut and the step is retried at the same start time:
* `"linearized"`: $\Delta t^n \leftarrow \max\left(0.85\,\dfrac{4}{1 + 3\,\varepsilon_T/\texttt{rel\_error\_T}}\,\Delta t^n,\ \Delta t_{min}\right)$, where $\varepsilon_T$ is the extrapolation error — an error-proportional cut (similar in form to classic embedded-RK adaptive step-size control), so a larger error produces a sharper cut.
* `"fully_coupled"`: a fixed cut factor of $0.5$ is applied.

If the step converges, the timestep history is shifted ($\Delta t^{n-1}\!\to\!\Delta t^{n-2}$, $\Delta t^n\!\to\!\Delta t^{n-1}$) and the timestep is grown geometrically for the next step, $\Delta t^{n+1} = \min(1.25\,\Delta t^n,\ \Delta t_{max})$ — a fixed growth factor, not error-based.

Finally, the proposed step is clamped so it never overshoots a *critical time*: any boundary-condition breakpoint (a `"table"` time or a `"ramp"` end time — see [Value Specification](README.md#boundary_conditions)) or `t_end`, collected up front by [`get_critical_times`](src/raft/subsurface.py). If $t + \Delta t^n$ would step past the next critical time, $\Delta t^n$ is shortened to land exactly on it, so boundary-condition discontinuities always fall on a solved timestep rather than being smoothed over within one.

## Tangent Matrices
The `"fully_coupled"` solver requires the Jacobian of the residuals with respect to the nodal degrees of freedom, assembled from the following blocks (mass matrices are lumped to prevent transient spurious oscillations):

$$\textbf{K}_{pp} = \frac{\partial\textbf{R}_p}{\partial\textbf{p}^n} = -c_{v0}\textbf{C}_p + \textbf{K}_{Dp}, \qquad \textbf{K}_{Dp} = \int_\Omega \textbf{B}^\top \frac{\boldsymbol{k}_f}{\mu}\textbf{B}\,d\Omega \ \text{(hydraulic conductivity matrix)}$$

$$\textbf{K}_{pT} = \frac{\partial\textbf{R}_p}{\partial\textbf{T}^n} = \textbf{K}_b + c_{v0}\textbf{K}_z, \qquad \textbf{K}_b = \int_\Omega \textbf{N}^\top \frac{\rho_{f0}\alpha_f}{\mu}\boldsymbol{k}_f\textbf{g}^\top\textbf{B}\,d\Omega \ \text{(buoyancy)}, \quad \textbf{K}_z = \int_\Omega \textbf{N}^\top \alpha_e \textbf{N}\,d\Omega \ \text{(pressurization)}$$

$$\textbf{K}_{Tp} = \frac{\partial\textbf{R}_T}{\partial\textbf{p}^n} = \textbf{K}_{Ap} + \textbf{K}_{stabP}, \qquad \textbf{K}_{Ap} = \int_\Omega \textbf{N}^\top \rho_f c_f \frac{\boldsymbol{k}_f}{\mu}\textbf{B}^\top\textbf{T}^\top\textbf{B}\,d\Omega \ \text{(pressure advection tangent)}$$

$$\textbf{K}_{TT} = \frac{\partial\textbf{R}_T}{\partial\textbf{T}^n} = -c_{v0}\textbf{C}_T + \textbf{K}_{AT} + \textbf{K}_{DT} + \textbf{K}_{stabT}$$

where $\textbf{K}_{DT} = \int_\Omega \textbf{B}^\top \lambda_e \textbf{B}\,d\Omega$ is the heat conduction matrix, $\textbf{K}_{AT}$ the heat advection matrix, and $\textbf{K}_{stabP}$, $\textbf{K}_{stabT}$ the SUPG stabilization tangent contributions. When viscosity is non-constant (see [`material.fluid.viscosity`](README.md#material)), additional $\textbf{K}_{pT}^\mu$ and $\textbf{K}_{TT}^\mu$ terms are included via $\partial\mu/\partial T$.
