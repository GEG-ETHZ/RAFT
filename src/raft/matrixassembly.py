"""Assembles global linear FEM matrices."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.sparse import csc_matrix

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
    integrand_arr = np.asarray(integrand, dtype=np.float64)
    left_T = left_matrix.transpose(0, 2, 1)

    # choose between multiply types for scalar vs. non-scalar material properties
    middle = integrand_arr * right_matrix if scalar else integrand_arr @ right_matrix

    K = left_T @ middle * weights.reshape(-1, 1, 1)

    return np.array(np.sum(K, 0))


def integrate_linear_matrices(
    mesh: dict[str, Any], mat: dict[str, Any]
) -> tuple[dict[str, csc_matrix[Any]], dict[str, Any], dict[str, Any]]:
    """Integrate linear matrices outside of time integration loop."""
    # ----- fluid properties, universal
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
    list_of_matrices = ["Kpp", "KM", "KpT", "KbT", "C", "KTT"]
    vec = _initiate_vectors(list_of_matrices, ne * nne * nne)

    # Matrices to assemble:
    # Kpp = Pressure conductance matrix
    # KM =  Pressure storage matrix
    # KpT =  Pressurization matrix
    # C = Thermal capacity matrix
    # KTT =  Thermal conductance matrix
    # KbT = Pressure buoyancy matrix - body

    # Evaluate quadrature points and get weights and shape functions.
    W, nq, N_q, GN_q = _get_quad_points_and_shape_functions(mesh["n_quadrature"], nsd, etype)

    # ----- Matrix integration element loop
    for r in range(num_regions):
        # solid properties - by region
        alpha_e = mat["solid"]["therm_expansion_eff"][r]
        comp_f = 1 / mat["solid"]["storage_mod"][r]
        kf = mat["solid"]["permeability_eff"][r]
        m_e = mat["solid"]["volumetric_heat_capacity_eff"][r]
        lambda_e = mat["solid"]["therm_cond_eff"][r]

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
            element_matrices["Kpp"] = _vectorized_quad_point_integration(
                B_q, kf / mu, B_q, wJe_q, isotropic_perm
            )
            element_matrices["KpT"] = _vectorized_quad_point_integration(N_q, alpha_e, N_q, wJe_q, True)
            element_matrices["KbT"] = _vectorized_quad_point_integration(N_q, kf_b, B_q, wJe_q, False)
            element_matrices["C"] = _vectorized_quad_point_integration(N_q, m_e, N_q, wJe_q, True)
            element_matrices["KTT"] = _vectorized_quad_point_integration(
                B_q, lambda_e, B_q, wJe_q, isotropic_cond
            )

            # ----- Assemble element contributions into global matrices
            start = e * nne * nne  # vectorized index count start
            stop = start + nne * nne  # vectorized index count end
            _store_vectorized_element_matrix(vec, element_matrices, start, stop)

    return (
        _assemble_global_matrices(vec, conn, nn, nne),
        mat,
        mesh,
    )


def integrate_nonlinear_matrices(
    mesh: dict[str, Any],
    mat: dict[str, Any],
    primary_variables: dict[str, Any],
) -> dict[str, Any]:
    """Integrate non-linear matrices inside time integration loop."""
    p = primary_variables["p"][:, 0]
    T = primary_variables["T"][:, 0]  # noqa: F841

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
    vec = _initiate_vectors(list_of_matrices, ne * nne * nne)

    # KATT = Temperature advection tangent matrix wrt temperature
    # K_stab_TT = l Advection stabilization wrt temperature

    # These following two matrices are not used in the linearized extrapolation.
    # They are the derivatives w.r.t pressure.
    # They are commented out as they are only needed in a fully-coupled solver.
    # potential fully-coupled version in the future.

    # K_stab_TP = Advection stabilization wrt pressure
    # KATP = Temperature advection tangent matrix wrt pressure

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
        advec_conduct = cp_f * rho_f * kf / mu  # advection conductivity

        for e in region_lists[r]:
            enodes = conn[e, :]  # local element nodes
            xI = nodes[enodes, :]
            B_q, detJe_q = getB(GN_q, xI, nsd)
            wJe_q = W * detJe_q  # Weight * det(Jacobian), used in every integrand

            pe = p[enodes].reshape(-1, 1)  # element nodal pressures
            # Te = T[enodes].reshape(-1, 1)  # element nodal temperatures
            enodes_coords = nodes[enodes, :]
            L_vec = (enodes_coords.max(axis=0) - enodes_coords.min(axis=0)).reshape(-1, 1)  # element lengths

            # Element matrices
            element_matrices = _initialize_element_matrices(list_of_matrices, nne)

            # ----- Vectorized quadrature point integration loop
            # Calculate integrand at all quadrature points in nq x a x b arrays
            #
            # gradients
            gradp = B_q @ pe  # nq x nsd x 1

            # fluxes
            flux = -advec_conduct * gradp if isotropic_perm else -advec_conduct @ gradp
            flux_mag = np.sqrt(flux[:, 0, 0] ** 2 + flux[:, 1, 0] ** 2 + flux[:, 2, 0] ** 2)
            indx_noflow = np.nonzero(flux_mag == 0)
            # prevent divide by zero
            flux_mag[indx_noflow] = 1.0e-15
            flux[indx_noflow, :, :] = 1.0e-15

            # SUPG Stabilization
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

            # SUPG stabilization coefficient (Shakib, Hughes, & Zohan (1991))
            tau = h_eff / (2 * flux_mag) * (1 + 9 / Pe**2) ** (-0.5)
            tau[:, :, indx_noflow] = 0

            # integrands
            integrand_KA = flux.transpose(0, 2, 1)  # nq x 1 x nsd
            integrand_Kstab = tau.reshape(-1, 1, 1) * flux @ flux.transpose(0, 2, 1)

            # ---- Integrate over all quadrature points
            element_matrices["KATT"] = _vectorized_quad_point_integration(
                N_q, integrand_KA, B_q, wJe_q, False
            )
            element_matrices["K_stab_TT"] = _vectorized_quad_point_integration(
                B_q, integrand_Kstab, B_q, wJe_q, False
            )

            # Vectorized integration forms the following loop:
            # for q in range(nq):
            #     B = B_q[q]
            #     N = N_q[q]
            #     wJe = wJe_q[q]

            #     # gradients
            #     gradp = B @ pe
            #     # gradT = B @ Te

            #     # fluxes
            #     flux = -np.dot(advec_conduct, gradp)
            #     flux_mag = math.sqrt(flux[0, 0] ** 2 + flux[1, 0] ** 2 + flux[2, 0] ** 2)

            #     # SUPG Stabilization
            #     if flux_mag == 0:
            #         tau = 0
            #     else:
            #         # Effective length weighted by fluxes
            #         h_eff = flux_mag / math.sqrt(
            #             (flux[0, 0] / L_vec[0, 0]) ** 2
            #             + (flux[1, 0] / L_vec[1, 0]) ** 2
            #             + (flux[2, 0] / L_vec[2, 0]) ** 2
            #         )

            #         if isotropic_cond:
            #             lambda_e_projected = lambda_e
            #         else:
            #             # Steamline vector
            #             n_stream = flux / flux_mag
            #             # Project conductivity along the vector
            #             lambda_e_projected = n_stream.T @ lambda_e @ n_stream

            #         # Pecelt number
            #         Pe = h_eff * flux_mag / lambda_e_projected
            #         # SUPG stabilization coefficient (Shakib, Hughes, & Zohan (1991))
            #         tau = h_eff / (2 * flux_mag) * (1 + 9 / Pe**2) ** (-0.5)

            #     # Resdiual of the discretized form, dropping 2nd order terms
            #     # Rth = flux.T @ gradT
            #     # 2nd order terms are usually negligible, but may be necessary for higher order elements.
            #     # - lambda_e * G * Te
            #     # - m_e * cv1 * N * Te
            #     # G = d^2/dx^2 ( N )

            #     # Integrate
            #     element_matrices["KATT"] += N.T @ (flux.T @ B) * wJe
            #     element_matrices["K_stab_TT"] += B.T * tau @ flux @ (flux.T @ B) * wJe
            #     # with 2nd order terms
            #     # B.T * tau * advec_conduct @ gradp * ( flux.T*B  - lambda_e * G - m_e * cv1 * N ) * wJe

            #     # Derivatives wrt pressure
            #     # KATPe += N.T * advec_conduct @ gradT.T @ B * wJe
            #     # K_stab_TPe += B.T * tau * advec_conduct * (Rth
            #                    + advec_conduct * gradp.T @ gradT) @ B * wJe

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
        element_matrices["A"] = _vectorized_quad_point_integration(
            N_q, np.asarray(1.0), N_q, abs(wJe_q), True
        )

        # ----- Assemble element contributions into global matrices
        # Build global (row, col) index pairs for this element block once
        start = e * nne * nne
        stop = start + nne * nne
        _store_vectorized_element_matrix(vec, element_matrices, start, stop)

    matrices = _assemble_global_matrices(vec, conn_face_local, nn_face, nne)
    # Map local ordering back to global ordering using the input node_order
    local_idx = np.searchsorted(face_node_ids, np.asarray(node_order, dtype=np.int32))
    return {name: mat[local_idx, :][:, local_idx] for name, mat in matrices.items()}
