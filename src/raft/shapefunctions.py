"""Get the finite element shape functions."""

import numpy as np
import numpy.typing as npt
from numba import njit  # type: ignore[import-untyped]


@njit  # type: ignore[untyped-decorator]
def getNmatrix(parent_coords: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Get the shape function matrix from parent coordinates."""
    # This function assumes B8 linear elements and should be updated for different element types.
    #
    # parent_coords: [1x3] parent coordinates at which the shape function is computed
    # N            : [1x n dof e] Shape function matrix

    N = np.empty((1, 8))
    n1 = 0.5 - 0.5 * parent_coords
    n2 = 0.5 + 0.5 * parent_coords
    N[0, 0] = n1[0] * n1[1] * n1[2]
    N[0, 1] = n2[0] * n1[1] * n1[2]
    N[0, 2] = n2[0] * n2[1] * n1[2]
    N[0, 3] = n1[0] * n2[1] * n1[2]
    N[0, 4] = n1[0] * n1[1] * n2[2]
    N[0, 5] = n2[0] * n1[1] * n2[2]
    N[0, 6] = n2[0] * n2[1] * n2[2]
    N[0, 7] = n1[0] * n2[1] * n2[2]

    return N


@njit  # type: ignore[untyped-decorator]
def _inverseJ(J: npt.NDArray[np.float64], nsd: int) -> npt.NDArray[np.float64]:
    Jinv = np.empty((nsd, nsd), dtype=np.float64)
    if nsd == 2:
        Jinv = (
            1.0
            / (J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
            * np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]])
        )
    else:
        # Unpack the matrix elements
        m00, m01, m02 = J[0]
        m10, m11, m12 = J[1]
        m20, m21, m22 = J[2]

        # Compute the determinant
        det = m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) + m02 * (m10 * m21 - m11 * m20)

        inv_det = 1.0 / det

        # Compute the adjugate matrix elements multiplied by 1/det
        Jinv = np.array(
            [
                [
                    (m11 * m22 - m12 * m21) * inv_det,
                    (m02 * m21 - m01 * m22) * inv_det,
                    (m01 * m12 - m02 * m11) * inv_det,
                ],
                [
                    (m12 * m20 - m10 * m22) * inv_det,
                    (m00 * m22 - m02 * m20) * inv_det,
                    (m02 * m10 - m00 * m12) * inv_det,
                ],
                [
                    (m10 * m21 - m11 * m20) * inv_det,
                    (m01 * m20 - m00 * m21) * inv_det,
                    (m00 * m11 - m01 * m10) * inv_det,
                ],
            ]
        )

    return Jinv


@njit(cache=True)  # type: ignore[untyped-decorator]
def getBmatrix(
    parents_coords: npt.NDArray[np.float64], phys_coords: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Get the gradient and Jacobian matrices from parent and physical coordinates."""
    # This function assumes B8 linear elements and should be updated for different element types.
    #
    # parent_coords : [1 x nsd] parent coordinates at which the shape function is computed
    # phys_coords   : [ndofe x nsd] physical coordinates of the element nodes
    # B             : [nsd x ndofe] Gradient matrix
    # J             : [nsd x nsd] Jacobian matrix

    nsd = len(parents_coords)

    xi = parents_coords[0]
    eta = parents_coords[1]
    if nsd == 3:
        zeta = parents_coords[2]

    # Derivatives of shape functions in parent coordinates
    ez = eta * zeta
    xz = xi * zeta
    xe = xi * eta
    dNdxi = np.empty((3, 8))

    dNdxi[0, 0] = 0.125 * (-1 + eta + zeta - ez)
    dNdxi[0, 1] = 0.125 * (1 - eta - zeta + ez)
    dNdxi[0, 2] = 0.125 * (1 + eta - zeta - ez)
    dNdxi[0, 3] = 0.125 * (-1 - eta + zeta + ez)
    dNdxi[0, 4] = 0.125 * (-1 + eta - zeta + ez)
    dNdxi[0, 5] = 0.125 * (1 - eta + zeta - ez)
    dNdxi[0, 6] = 0.125 * (1 + eta + zeta + ez)
    dNdxi[0, 7] = 0.125 * (-1 - eta - zeta - ez)

    dNdxi[1, 0] = 0.125 * (-1 + xi + zeta - xz)
    dNdxi[1, 1] = 0.125 * (-1 - xi + zeta + xz)
    dNdxi[1, 2] = 0.125 * (1 + xi - zeta - xz)
    dNdxi[1, 3] = 0.125 * (1 - xi - zeta + xz)
    dNdxi[1, 4] = 0.125 * (-1 + xi - zeta + xz)
    dNdxi[1, 5] = 0.125 * (-1 - xi - zeta - xz)
    dNdxi[1, 6] = 0.125 * (1 + xi + zeta + xz)
    dNdxi[1, 7] = 0.125 * (1 - xi + zeta - xz)

    dNdxi[2, 0] = 0.125 * (-1 + xi + eta - xe)
    dNdxi[2, 1] = 0.125 * (-1 - xi + eta + xe)
    dNdxi[2, 2] = 0.125 * (-1 - xi - eta - xe)
    dNdxi[2, 3] = 0.125 * (-1 + xi - eta + xe)
    dNdxi[2, 4] = 0.125 * (1 - xi - eta + xe)
    dNdxi[2, 5] = 0.125 * (1 + xi - eta - xe)
    dNdxi[2, 6] = 0.125 * (1 + xi + eta + xe)
    dNdxi[2, 7] = 0.125 * (1 - xi + eta - xe)

    J = dNdxi @ phys_coords

    Jinv = _inverseJ(J, nsd)
    B = Jinv @ dNdxi

    return B, J


def getquadrature(qt: int, ndim: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Returns the Gauss quadrature points and weights for the
    specified order of integration and number of dimensions.
    """
    #   qt:     number of quadrature points
    #   nsd:    number of space dimensions
    #   W:      [qt^ndim x 1] vector of quadrature weights
    #   Q:      [qt^ndim x ndim] quadrature coordinates in parent coordinate system

    rpt = np.zeros((qt, 1))
    rwt = np.zeros((qt, 1))
    if qt == 1:
        rpt[0] = 0.0

        rwt[0] = 2.0
    elif qt == 2:
        rpt[0] = 0.577350269189626
        rpt[1] = -0.577350269189626

        rwt[0] = 1.0
        rwt[1] = 1.0
    elif qt == 3:
        rpt[0] = 0.774596669241483
        rpt[1] = -0.774596669241483
        rpt[2] = 0

        rwt[0] = 0.555555555555556
        rwt[1] = 0.555555555555556
        rwt[2] = 0.888888888888889
    else:
        raise Exception("Unsupported number of quadrature points")

    # Create grid
    grids = np.meshgrid(*([rpt] * ndim), indexing="ij")

    # Create points
    Q = np.stack(grids, axis=-1).reshape(-1, ndim)

    # Create weights
    wgrids = np.meshgrid(*([rwt] * ndim), indexing="ij")
    W = np.prod(wgrids, axis=0).ravel()

    return Q, W
