"""Assembles global linear FEM matrices."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.sparse import csc_matrix

from .matprops import get_viscosity
from .shapefunctions import GNMATRIX_FUNCS, NMATRIX_FUNCS, getB, getquadrature


def _make_sparse(
    vals: npt.NDArray[np.float64],
    row_vec: npt.NDArray[np.int32],
    col_vec: npt.NDArray[np.int32],
    nn: int,
) -> csc_matrix:
    return csc_matrix((vals, (row_vec, col_vec)), shape=(nn, nn))


def _assemble_global_matrices(
    vec: dict[str, npt.NDArray[np.float64]], conn: npt.NDArray[np.int32], nn: int, nne: int
) -> dict[str, csc_matrix]:
    """Assemble vector of rows and columns all at once and assemble sparse matrix."""
    # For vectorized sparse matrix assembly, we feed in 3 vectors:
    # vec: each entry of the element-by-element sparse matrices
    # row_vec: each entry indicates the row that the entry in vec belongs to
    # col_vec:each entry indicates the column that the entry in vec belongs to
    #
    # Rather than assemble the row and column vectors element by element,
    # this function computes them in one go
    row_vec = np.repeat(conn, nne, axis=1).ravel()
    col_vec = np.tile(conn, (1, nne)).ravel()

    return {name: _make_sparse(vec[name], row_vec, col_vec, nn) for name in vec}


def _get_quad_points_and_shape_functions(
    num_quad_base: int,
    nsd: int,
    etype: str,
) -> tuple[npt.NDArray[np.float64], int, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate list of weights, shape functions, and shape function gradients at quadrature points."""
    # ----- Quadrature points
    Q, W = getquadrature(num_quad_base, nsd, etype)
    nq = len(W)

    # ---- Evaluate element type function
    getN = NMATRIX_FUNCS[etype]  # Shape function matrix
    getGN = GNMATRIX_FUNCS[etype]  # Gradients of shape function matrix
    # Evaluate N and GN at the parent coordinates once
    N_q = []
    GN_q = []
    for q in range(nq):
        xi = Q[q, :]
        N_q.append(getN(xi))
        GN_q.append(getGN(xi))

    return W, nq, np.array(N_q), np.asarray(GN_q)


def _initiate_vectors(list_of_matrices: list[str], total_entries: int) -> dict[str, npt.NDArray[np.float64]]:
    """Create the dictionary of vectorized matrix entries."""
    return {name: np.zeros(total_entries) for name in list_of_matrices}


def _store_vectorized_element_matrix(
    vec: dict[str, npt.NDArray[np.float64]],
    element_matrices: dict[str, npt.NDArray[np.float64]],
    start: int,
    stop: int,
) -> None:
    """Ravel element matrices and store them in the element-entry vector."""
    for name, Ee in element_matrices.items():
        vec[name][start:stop] = Ee.ravel()


def _initialize_element_matrices(list_of_matrices: list[str], nne: int) -> dict[str, npt.NDArray[np.float64]]:
    """Create a dict of zero-initialized (nne, nne) element matrices, keyed by name."""
    return {name: np.zeros((nne, nne)) for name in list_of_matrices}


def _vectorized_quad_point_integration(
    left_matrix: npt.NDArray[np.float64],
    integrand: npt.NDArray[np.float64],
    right_matrix: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    scalar: bool,
) -> npt.NDArray[np.float64]:
    """Perform vectorized matrix integration over quadrature points.

    Takes in a left matrix that gets transposed, a middle integrand, and a right matrix.
    The left matrix should have dimensions nq x a x nne.
    The right matrix should have dimensions nq x b x nne.
    The integrand should have dimensions a x b or nq x a x b.
    """
    if scalar:
        integrand = np.asarray(integrand, dtype=np.float64)
    left_T = left_matrix.transpose(0, 2, 1)

    # choose between multiply types for scalar vs. non-scalar material properties
    middle = integrand * right_matrix if scalar else integrand @ right_matrix

    K = left_T @ middle * weights.reshape(-1, 1, 1)

    result: npt.NDArray[np.float64] = K.sum(0)
    return result


def integrate_linear_matrices(
    mesh: dict[str, Any], mat: dict[str, Any], controls: dict[str, Any]
) -> tuple[dict[str, csc_matrix[Any]], dict[str, Any], dict[str, Any]]:
    """Integrate linear matrices outside of time integration loop."""
    # ----- integration controls
    mu_linear = not bool(controls["viscosity_nonconstant"])

    # ----- fluid properties, universal
    if mu_linear:
        mu = mat["fluid"]["viscosity"]
        alpha_f = mat["fluid"]["therm_expansion"]
    rho_f = mat["fluid"]["density"]

    # ----- mesh data
    nn, ne, nne, nsd = mesh["nn"], mesh["ne"], mesh["nne"], mesh["nsd"]
    conn = mesh["conn"]  # (ne, nne)  - connectivity matrix
    nodes = mesh["nodes"]  # (nn,  nsd) - nodal degrees of freedom
    region_lists = mesh["region_lists"]
    num_regions = mesh["num_regions"]
    etype = mesh["element_type"]

    # ----- gravity vector
    ng = np.zeros((nsd, 1))
    ng[nsd - 1] = 1.0  # gravity normal
    g = -9.81 * ng

    # ----- Vectorized matrix assembly
    # Assemble lists of rows, columns, and entries and then assemble the sparse
    # matrix all at once.
    # For the scalar pressure and temperature, there are nne * nne entries
    # per element.
    list_of_matrices = ["Kpp", "KM", "KpT", "KbT", "C", "KTT"] if mu_linear else ["KM", "KpT", "C", "KTT"]
    vec = _initiate_vectors(list_of_matrices, ne * nne * nne)

    # Matrices to assemble:
    # Kpp = Pressure conductance matrix
    # KbT = Pressure buoyancy matrix - body
    # KM =  Pressure storage matrix
    # KpT =  Pressurization matrix
    # C = Thermal capacity matrix
    # KTT =  Thermal conductance matrix

    # Evaluate quadrature points and get weights and shape functions.
    W, nq, N_q, GN_q = _get_quad_points_and_shape_functions(mesh["n_quadrature"], nsd, etype)

    # ----- Matrix integration element loop
    for r in range(num_regions):
        # solid properties - by region
        alpha_e = np.asarray(mat["solid"]["therm_expansion_eff"][r])
        comp_f = np.asarray(1 / mat["solid"]["storage_mod"][r])
        kf = mat["solid"]["permeability_eff"][r]
        m_e = mat["solid"]["volumetric_heat_capacity_eff"][r]
        lambda_e = mat["solid"]["therm_cond_eff"][r]

        if mu_linear:
            kf_b = ng.T @ np.dot(kf, ng) / mu * alpha_f * rho_f @ g.T  # kf component in gravity direction

        # Check for region isotropy
        isotropic_cond = bool(isinstance(lambda_e, (float, int)))
        isotropic_perm = bool(isinstance(kf, (float, int)))

        for e in region_lists[r]:
            enodes = conn[e, :]  # local element nodes
            xI = nodes[enodes, :]
            B_q, detJe_q = getB(GN_q, xI, nsd)
            wJe_q = W * detJe_q  # Weight * det(Jacobian), used in every integrand

            # Element matrices
            element_matrices = _initialize_element_matrices(list_of_matrices, nne)

            # Vectorized integrand computation
            # N_q: [nq, 1, nne]
            # B_q: [nq, nsd, nne]
            # wJe: [nq,]

            # Vectorized integration performs the following loop:
            # for q in range(nq):
            #     B = B_q[q]
            #     N = N_q[q]
            #     wJe = wJe_q[q]

            #     element_matrices["KM"] += N.T @ (comp_f * N) * wJe
            #     element_matrices["Kpp"] += B.T @ np.dot(kf / mu, B) * wJe
            #     element_matrices["KpT"] += N.T @ (alpha_e * N) * wJe
            #     element_matrices["KbT"] += N.T @ (kf_b @ g.T) @ B * wJe
            #     element_matrices["C"] += N.T @ (m_e * N) * wJe
            #     element_matrices["KTT"] += B.T @ np.dot(lambda_e, B) * wJe

            element_matrices["KM"] = _vectorized_quad_point_integration(N_q, comp_f, N_q, wJe_q, True)
            element_matrices["KpT"] = _vectorized_quad_point_integration(N_q, alpha_e, N_q, wJe_q, True)
            element_matrices["C"] = _vectorized_quad_point_integration(N_q, m_e, N_q, wJe_q, True)
            element_matrices["KTT"] = _vectorized_quad_point_integration(
                B_q, lambda_e, B_q, wJe_q, isotropic_cond
            )
            if mu_linear:
                element_matrices["Kpp"] = _vectorized_quad_point_integration(
                    B_q, kf / mu, B_q, wJe_q, isotropic_perm
                )
                element_matrices["KbT"] = _vectorized_quad_point_integration(N_q, kf_b, B_q, wJe_q, False)

            # ----- Assemble element contributions into global matrices
            start = e * nne * nne  # vectorized index count start
            stop = start + nne * nne  # vectorized index count end
            _store_vectorized_element_matrix(vec, element_matrices, start, stop)

    return (
        _assemble_global_matrices(vec, conn, nn, nne),
        mat,
        mesh,
    )


@dataclass
class _SUPGQuantities:
    """Flux, Peclet number, and SUPG stabilization terms at an element's quadrature points.

    Computed once per element by `_compute_supg_quantities` and shared by every
    physics block that needs them (base advection, viscosity correction, and the
    fully-coupled pressure/temperature off-diagonal terms), instead of recomputing
    or re-threading them through each block individually.
    """

    gradp: npt.NDArray[np.float64]  # nq x nsd x 1
    flux: npt.NDArray[np.float64]  # nq x nsd x 1
    flux_T: npt.NDArray[np.float64]  # nq x 1 x nsd # noqa
    flux_mag: npt.NDArray[np.float64]  # nq,
    tau: npt.NDArray[np.float64]  # nq x 1 x 1
    Pe: npt.NDArray[np.float64]  # Peclet number
    integrand_Kstab: npt.NDArray[np.float64]  # tau * flux @ flux_T, used for K_stab_TT # noqa


def _compute_supg_quantities(
    B_q: npt.NDArray[np.float64],
    pe: npt.NDArray[np.float64],
    advec_conduct: npt.NDArray[np.float64],
    isotropic_perm: bool,
    isotropic_cond: bool,
    lambda_e: npt.NDArray[np.float64] | float,
    L_vec: npt.NDArray[np.float64],
    nq: int,
) -> _SUPGQuantities:
    """Compute the Darcy flux and SUPG (Shakib, Hughes, & Zohan (1991)) stabilization terms."""
    gradp = B_q @ pe  # nq x nsd x 1

    flux = -advec_conduct * gradp if isotropic_perm else -advec_conduct @ gradp
    flux_T = flux.transpose(0, 2, 1)  # quadrature-point-wise transpose, calculate once
    flux_mag = np.sqrt(flux[:, 0, 0] ** 2 + flux[:, 1, 0] ** 2 + flux[:, 2, 0] ** 2)  # flux magnitude
    indx_noflow = np.nonzero(flux_mag == 0)
    # prevent divide by zero
    flux_mag[indx_noflow] = 1.0e-15
    flux[indx_noflow, :, :] = 1.0e-15

    # Effective length weighted by fluxes
    h_eff = flux_mag / np.sqrt(
        (flux[:, 0, 0] / L_vec[0, 0]) ** 2
        + (flux[:, 1, 0] / L_vec[1, 0]) ** 2
        + (flux[:, 2, 0] / L_vec[2, 0]) ** 2
    )
    if isotropic_cond:
        lambda_e_projected = lambda_e * np.ones((nq, 1, 1))
    else:
        # anisotropic case - project conductivity along flowlines
        n_stream = flux / flux_mag.reshape(-1, 1, 1)
        n_stream_T = n_stream.transpose(0, 2, 1)
        lambda_e_projected = (n_stream_T @ lambda_e) @ n_stream  # nq x 1 x 1

    # Peclet number
    Pe = h_eff * flux_mag / lambda_e_projected.reshape(1, 1, -1)

    # SUPG stabilization coefficient
    tau = h_eff / (2 * flux_mag) * (1 + 9 / Pe**2) ** (-0.5)
    tau[:, :, indx_noflow] = 0
    tau = tau.reshape(-1, 1, 1)

    integrand_Kstab = tau * flux @ flux_T

    return _SUPGQuantities(
        gradp=gradp,
        flux=flux,
        flux_T=flux_T,
        flux_mag=flux_mag,
        tau=tau,
        Pe=Pe,
        integrand_Kstab=integrand_Kstab,
    )


def _integrate_pressure_flow_terms(
    N_q: npt.NDArray[np.float64],
    B_q: npt.NDArray[np.float64],
    wJe_q: npt.NDArray[np.float64],
    kf: npt.NDArray[np.float64] | float,
    mu_q: npt.NDArray[np.float64],
    kf_b: npt.NDArray[np.float64],
    isotropic_perm: bool,
) -> dict[str, npt.NDArray[np.float64]]:
    """Integrate the viscosity-dependent pressure conductance (Kpp) and buoyancy (KbT) matrices."""
    return {
        "Kpp": _vectorized_quad_point_integration(B_q, kf / mu_q, B_q, wJe_q, isotropic_perm),
        "KbT": _vectorized_quad_point_integration(N_q, kf_b, B_q, wJe_q, False),
    }


def _integrate_viscosity_temperature_correction(
    N_q: npt.NDArray[np.float64],
    B_q: npt.NDArray[np.float64],
    wJe_q: npt.NDArray[np.float64],
    gradT: npt.NDArray[np.float64],
    supg: _SUPGQuantities,
    mu_q: npt.NDArray[np.float64],
    dmu_dT_q: npt.NDArray[np.float64],
) -> dict[str, npt.NDArray[np.float64]]:
    """Integrate the viscosity-derivative correction to the temperature advection/stabilization matrices."""
    FA = supg.flux_T @ gradT
    integrand_KATT_mu = -1 / mu_q @ FA * dmu_dT_q

    Fstab = (supg.tau * supg.flux) @ (supg.flux_T @ gradT)

    # tau derivative wrt mu(T) correction factor
    correction_factor = 0.5 * (supg.Pe**2 + 18) / (supg.Pe**2 + 9)
    integrand_Kstab_mu = -2 / mu_q * Fstab * dmu_dT_q * correction_factor.reshape(-1, 1, 1)

    return {
        "KATT_mu": _vectorized_quad_point_integration(N_q, integrand_KATT_mu, N_q, wJe_q, False),
        "K_stab_TT_mu": _vectorized_quad_point_integration(B_q, integrand_Kstab_mu, N_q, wJe_q, False),
    }


def _integrate_coupled_pressure_terms(
    N_q: npt.NDArray[np.float64],
    B_q: npt.NDArray[np.float64],
    wJe_q: npt.NDArray[np.float64],
    gradT: npt.NDArray[np.float64],
    supg: _SUPGQuantities,
    advec_conduct: npt.NDArray[np.float64],
    isotropic_perm: bool,
    nsd: int,
) -> dict[str, npt.NDArray[np.float64]]:
    """Integrate the fully-coupled temperature-wrt-pressure off-diagonal matrices (KATp, K_stab_Tp)."""
    Rth_dp = -advec_conduct * gradT if isotropic_perm else -advec_conduct @ gradT
    integrand_KATp = Rth_dp.transpose(0, 2, 1)  # nq x 1 x nsd

    # Stabilization wrt p
    Rth = supg.flux_T @ gradT
    inner_term = Rth * np.eye(nsd) + supg.flux @ gradT.transpose(0, 2, 1)

    flux_mag_sq = supg.flux_mag.reshape(-1, 1, 1) ** 2
    pe_factor = supg.Pe**2 / (supg.Pe**2 + 9)
    tau_correction_Tp = pe_factor.reshape(-1, 1, 1) * (Rth / flux_mag_sq) * (supg.flux @ supg.flux_T)
    inner_term = inner_term - tau_correction_Tp

    integrand_Kstab_Tp = (
        -supg.tau * advec_conduct * inner_term if isotropic_perm else -supg.tau * (inner_term @ advec_conduct)
    )

    return {
        "KATp": _vectorized_quad_point_integration(N_q, integrand_KATp, B_q, wJe_q, False),
        "K_stab_Tp": _vectorized_quad_point_integration(B_q, integrand_Kstab_Tp, B_q, wJe_q, False),
    }


def _integrate_viscosity_pressure_correction(
    N_q: npt.NDArray[np.float64],
    B_q: npt.NDArray[np.float64],
    wJe_q: npt.NDArray[np.float64],
    supg: _SUPGQuantities,
    kf: npt.NDArray[np.float64] | float,
    mu_q: npt.NDArray[np.float64],
    dmu_dT_q: npt.NDArray[np.float64],
    kf_b: npt.NDArray[np.float64],
    del_T: npt.NDArray[np.float64],
    isotropic_perm: bool,
) -> dict[str, npt.NDArray[np.float64]]:
    """Integrate the viscosity-wrt-temperature correction to the pressure/buoyancy matrices."""
    # contribution of viscosity to pressure terms
    Fk = kf / mu_q * supg.gradp if isotropic_perm else kf / mu_q @ supg.gradp  # (nq, nsd, 1)
    Fb = kf_b @ B_q @ del_T  # (nq, 1, 1)

    #   Darcy's law component
    integrand_KpT_mu_p = -(dmu_dT_q / mu_q) * Fk  # (nq, nsd, 1), elementwise
    #   Buoyancy component
    integrand_KpT_mu_b = -1.0 / mu_q * Fb @ dmu_dT_q

    return {
        "KpT_mu_p": _vectorized_quad_point_integration(B_q, integrand_KpT_mu_p, N_q, wJe_q, False),
        "KpT_mu_b": _vectorized_quad_point_integration(N_q, integrand_KpT_mu_b, N_q, wJe_q, False),
    }


def integrate_nonlinear_matrices(
    mesh: dict[str, Any],
    mat: dict[str, Any],
    primary_variables: dict[str, Any],
    controls: dict[str, Any],
) -> dict[str, Any]:
    """Integrate non-linear matrices inside time integration loop.

    KATT = Temperature advection tangent matrix wrt temperature
    K_stab_TT = Advection stabilization wrt temperature
    KATp = Temperature advection tangent matrix wrt pressure
    K_stab_Tp = Advection stabilization wrt pressure
    Kpp, KbT, KpT_mu_p, KpT_mu_b, KATT_mu, K_stab_TT_mu = viscosity-nonlinear corrections, see helpers below.
    """
    p = primary_variables["p"][:, 0]
    T = primary_variables["T"][:, 0]
    # ----- integration controls
    mu_nonlinear = controls["viscosity_nonconstant"]
    mu_linear = not mu_nonlinear
    fully_coupled = controls["fully_coupled"]

    # ----- fluid properties
    mu = mat["fluid"]["viscosity"]
    cp_f = mat["fluid"]["specific_heat"]
    rho_f = mat["fluid"]["density"]

    # ----- mesh data
    nn, ne, nne, nsd = mesh["nn"], mesh["ne"], mesh["nne"], mesh["nsd"]
    conn = mesh["conn"]  # (ne, nne)  - connectivity matrix
    nodes = mesh["nodes"]  # (nn,  nsd) - nodal degrees of freedom
    region_lists = mesh["region_lists"]
    num_regions = mesh["num_regions"]
    etype = mesh["element_type"]

    # ----- Pre-allocate sparse matrices (use lil for efficient assembly)
    list_of_matrices = ["KATT", "K_stab_TT"]
    if fully_coupled:
        list_of_matrices.extend(["KATp", "K_stab_Tp"])

    if mu_nonlinear:
        list_of_matrices.extend(["Kpp", "KbT", "KpT_mu_b", "KpT_mu_p", "KATT_mu", "K_stab_TT_mu"])
        alpha_f = mat["fluid"]["therm_expansion"]

    # ----- gravity vector
    ng = np.zeros((nsd, 1))
    ng[nsd - 1] = 1.0  # gravity normal
    g = -9.81 * ng

    vec = _initiate_vectors(list_of_matrices, ne * nne * nne)

    # Evaluate quadrature points and get weights and shape functions.
    W, nq, N_q, GN_q = _get_quad_points_and_shape_functions(mesh["n_quadrature"], nsd, etype)

    # ----- Matrix integration element loop
    for r in range(num_regions):
        # solid properties - by region
        kf = mat["solid"]["permeability_eff"][r]
        lambda_e = mat["solid"]["therm_cond_eff"][r]

        # Check for region isotropy
        isotropic_cond = bool(isinstance(lambda_e, (float, int)))
        isotropic_perm = bool(isinstance(kf, (float, int)))

        # advection velocity = advection_conductivity * pressure gradient
        if mu_linear:
            advec_conduct = cp_f * rho_f * kf / mu  # advection conductivity

        for e in region_lists[r]:
            enodes = conn[e, :]  # local element nodes
            xI = nodes[enodes, :]
            B_q, detJe_q = getB(GN_q, xI, nsd)
            wJe_q = W * detJe_q  # Weight * det(Jacobian), used in every integrand

            # Element matrices initialize
            element_matrices = _initialize_element_matrices(list_of_matrices, nne)

            Te = T[enodes].reshape(-1, 1)  # element nodal temperatures
            pe = p[enodes].reshape(-1, 1)  # element nodal pressures
            enodes_coords = nodes[enodes, :]
            L_vec = (enodes_coords.max(axis=0) - enodes_coords.min(axis=0)).reshape(-1, 1)  # element lengths

            # ------------------------------------ Pressure and fluid flow
            if mu_nonlinear:
                # If mu is nonlinear, the pressure diffusion is also non-linear and must
                # be re-integrated at each timestep.
                T_q = N_q @ Te  # quadrature point temperatures
                mu_q, dmu_dT_q = get_viscosity(T_q, mu)  # quadrature point viscosities

                advec_conduct = cp_f * rho_f * kf / mu_q  # advection conductivity
                kf_b = (
                    ng.T @ np.dot(kf, ng) / mu_q * alpha_f * rho_f @ g.T
                )  # kf component in gravity direction

                # Integrate the tangent matrices
                # Kpp = dFk / dp - Darcy's Law
                # KbT = dFb / dT - Buoyancy
                element_matrices.update(
                    _integrate_pressure_flow_terms(N_q, B_q, wJe_q, kf, mu_q, kf_b, isotropic_perm)
                )

            # ----------------------------------- Advection and stabilization
            # Calculate fluxes, stabilization coefficicent, pressure gradients and store in supg dataclass
            supg = _compute_supg_quantities(
                B_q, pe, advec_conduct, isotropic_perm, isotropic_cond, lambda_e, L_vec, nq
            )

            # advection matrix
            element_matrices["KATT"] = _vectorized_quad_point_integration(N_q, supg.flux_T, B_q, wJe_q, False)
            # advection stabilization
            element_matrices["K_stab_TT"] = _vectorized_quad_point_integration(
                B_q, supg.integrand_Kstab, B_q, wJe_q, False
            )

            gradT = B_q @ Te

            # ---------------------------------- Viscosity correction
            if mu_nonlinear:
                # Integrate tangent matrices:
                # KATT_mu = dFA / dmu * dmu / dT
                # K_stab_TT_mu = dF_stab / dmu * dmu / dT
                element_matrices.update(
                    _integrate_viscosity_temperature_correction(N_q, B_q, wJe_q, gradT, supg, mu_q, dmu_dT_q)
                )

            # ----------------------------------- Fully coupled off diagonal terms
            if fully_coupled:
                # Integrate the off-diagonal tangents wrt pressure
                # KATp = dFA / dp
                # K_stab_Tp = dF_stab / dp
                element_matrices.update(
                    _integrate_coupled_pressure_terms(
                        N_q, B_q, wJe_q, gradT, supg, advec_conduct, isotropic_perm, nsd
                    )
                )

                # viscosity wrt T
                if mu_nonlinear:
                    # If non-linear viscosity, integrate the viscosity tangents
                    # KpT_mu_p = dFk / dmu * dmu / dT - for Darcy's law
                    # KpT_mu_b = dFb / dmu * dmu / dT - for buoyancy
                    T_0e = mat["T_0"][enodes]
                    del_T = Te - T_0e.reshape(-1, 1)
                    element_matrices.update(
                        _integrate_viscosity_pressure_correction(
                            N_q, B_q, wJe_q, supg, kf, mu_q, dmu_dT_q, kf_b, del_T, isotropic_perm
                        )
                    )

            # ----- Assemble element contributions into global matrices
            start = e * nne * nne
            stop = start + nne * nne
            _store_vectorized_element_matrix(vec, element_matrices, start, stop)

    # ----- Return and convert to sparse matrix
    return _assemble_global_matrices(vec, conn, nn, nne)


def integrate_NLpressure_matrices(
    mesh: dict[str, Any],
    mat: dict[str, Any],
    primary_variables: dict[str, Any],
    controls: dict[str, Any],  # noqa
) -> dict[str, Any]:
    """Integrate non-linear fluid flow matrices inside time integration loop."""
    T = primary_variables["T"][:, 0]  # noqa: F841

    # ----- fluid properties
    mu_coeff = mat["fluid"]["viscosity"]
    alpha_f = mat["fluid"]["therm_expansion"]
    rho_f = mat["fluid"]["density"]

    # ----- mesh data
    nn, ne, nne, nsd = mesh["nn"], mesh["ne"], mesh["nne"], mesh["nsd"]
    conn = mesh["conn"]  # (ne, nne)  - connectivity matrix
    nodes = mesh["nodes"]  # (nn,  nsd) - nodal degrees of freedom
    region_lists = mesh["region_lists"]
    num_regions = mesh["num_regions"]
    etype = mesh["element_type"]

    # ----- gravity vector
    ng = np.zeros((nsd, 1))
    ng[nsd - 1] = 1.0  # gravity normal
    g = -9.81 * ng

    # ----- Pre-allocate sparse matrices (use lil for efficient assembly)
    list_of_matrices = ["Kpp", "KbT"]

    vec = _initiate_vectors(list_of_matrices, ne * nne * nne)

    # Kpp = Pressure conductance matrix
    # KbT = Pressure buoyancy matrix - body

    # Evaluate quadrature points and get weights and shape functions.
    W, nq, N_q, GN_q = _get_quad_points_and_shape_functions(mesh["n_quadrature"], nsd, etype)

    # ----- Matrix integration element loop
    for r in range(num_regions):
        # solid properties - by region
        kf = mat["solid"]["permeability_eff"][r]

        isotropic_perm = bool(isinstance(kf, (float, int)))

        for e in region_lists[r]:
            enodes = conn[e, :]  # local element nodes
            xI = nodes[enodes, :]
            B_q, detJe_q = getB(GN_q, xI, nsd)
            wJe_q = W * detJe_q  # Weight * det(Jacobian), used in every integrand

            # Element matrices
            element_matrices = _initialize_element_matrices(list_of_matrices, nne)

            # vectorized integration
            Te = T[enodes].reshape(-1, 1)  # element nodal temperatures
            T_q = N_q @ Te  # quadrature point temperatures
            mu_q, _ = get_viscosity(T_q, mu_coeff)  # quadrature point viscosities

            kf_b = ng.T @ np.dot(kf, ng) / mu_q * alpha_f * rho_f @ g.T  # kf component in gravity direction

            element_matrices["Kpp"] = _vectorized_quad_point_integration(
                B_q, kf / mu_q, B_q, wJe_q, isotropic_perm
            )
            element_matrices["KbT"] = _vectorized_quad_point_integration(N_q, kf_b, B_q, wJe_q, False)

            # ----- Assemble element contributions into global matrices
            start = e * nne * nne
            stop = start + nne * nne
            _store_vectorized_element_matrix(vec, element_matrices, start, stop)

    # ----- Return and convert to sparse matrix
    return _assemble_global_matrices(vec, conn, nn, nne)


def integrate_boundary_flux_matrices(
    mesh: dict[str, Any],
    conn_face: list[list[int]],
    node_order: Sequence[int],
) -> dict[str, csc_matrix[Any]]:
    """Integrate lower dimensional boundary integrals for flux BCs."""
    ne = np.size(conn_face, 0)
    nne = np.size(conn_face, 1)
    nsd = mesh["nsd"] - 1
    nodes = mesh["nodes"]  # (nn,  nsd) - nodal degrees of freedom
    etype = mesh["element_type"]

    # ----- Local face numbering
    # Map each global node id in conn_face to a local index.
    # Used for assembly.
    face_node_ids = np.unique(np.asarray(conn_face, dtype=np.int32))
    global_to_local = {int(g): i for i, g in enumerate(face_node_ids)}
    nn_face = len(face_node_ids)

    conn_face_local = np.array(
        [[global_to_local[int(n)] for n in enodes] for enodes in conn_face],
        dtype=np.int32,
    )  # conn_face_local is used for assembly then mapped back to global

    match etype:
        case "B8":
            etype = "Q4"
            nq = 2
        case "T4":
            etype = "T3"
            nq = 1
        case _:
            raise NotImplementedError(f"Face elements of {etype}.")

    # ----- Vectorized matrix assembly
    # Assemble lists of rows, columns, and entries and then assemble the sparse
    # matrix all at once.
    # For the scalar pressure and temperature, there are nne * nne entries
    # per element.
    list_of_matrices = ["A"]
    vec = _initiate_vectors(list_of_matrices, ne * nne * nne)

    # Matrices to assemble:
    # A = area weight for flux distribution over the face
    integrand = np.asarray(1.0)

    # Evaluate quadrature points and get weights and shape functions.
    W, nq, N_q, GN_q = _get_quad_points_and_shape_functions(mesh["n_quadrature"], nsd, etype)

    # ----- Matrix integration element loop
    for e in range(ne):
        enodes = conn_face[e]  # global node ids, used for geometry

        # calculate face normal
        xI = nodes[enodes, :]
        n = np.cross(xI[1, :] - xI[0, :], xI[2, :] - xI[0, :])
        # Find the in-plane coordinates - assumes faces are aligned with cartesian axes
        ax = np.argmax(abs(n))
        xI = np.delete(xI, ax, 1)

        detJe_q = getB(GN_q, xI, nsd)[1]
        wJe_q = W * detJe_q  # Weight * det(Jacobian), used in every integrand

        # Element matrices
        element_matrices = _initialize_element_matrices(list_of_matrices, nne)
        # Boundary faces may be misordered relative to their global position,
        # relative in negative jacobians.
        # This shouldn't be an issue because all elements are defined with the same
        # relative numbering. Flux directions can then be controlled via
        # the sign of the applied traction.
        element_matrices["A"] = _vectorized_quad_point_integration(N_q, integrand, N_q, abs(wJe_q), True)

        # ----- Assemble element contributions into global matrices
        # Build global (row, col) index pairs for this element block once
        start = e * nne * nne
        stop = start + nne * nne
        _store_vectorized_element_matrix(vec, element_matrices, start, stop)

    matrices = _assemble_global_matrices(vec, conn_face_local, nn_face, nne)
    # Map local ordering back to global ordering using the input node_order
    local_idx = np.searchsorted(face_node_ids, np.asarray(node_order, dtype=np.int32))
    return {name: mat[local_idx, :][:, local_idx] for name, mat in matrices.items()}
