"""Get the finite element shape functions."""

import numpy as np
import numpy.typing as npt


def getNmatrix(parent_coords: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Get the shape function matrix from parent coordinates."""
    # This function assumes B8 linear elements and should be updated for different element types.
    #
    # parent_coords: [1x3] parent coordinates at which the shape function is computed
    # N            : [1x n dof e] Shape function matrix

    N = np.zeros((1, 8))
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


def getBmatrix(
    parents_coords: npt.NDArray[np.float64], phys_coords: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Get the gradient and Jacobian matrices from parent and physical coordinates."""
    # This function assumes B8 linear elements and should be updated for different element types.
    #
    # parent_coords : [1 x 3] parent coordinates at which the shape function is computed
    # phys_coords   : [ndofe x 3] physical coordinates of the element nodes
    # B             : [nsd x ndofe] Gradient matrix
    # J             : [nsd x nsd] Jacobian matrix

    xi = parents_coords[0]
    eta = parents_coords[1]
    zeta = parents_coords[2]

    # Derivatives of shape functions in parent coordinates
    dNdxi = (
        1
        / 8.0
        * np.array(
            [
                [
                    -1 + eta + zeta - eta * zeta,
                    1 - eta - zeta + eta * zeta,
                    1 + eta - zeta - eta * zeta,
                    -1 - eta + zeta + eta * zeta,
                    -1 + eta - zeta + eta * zeta,
                    1 - eta + zeta - eta * zeta,
                    1 + eta + zeta + eta * zeta,
                    -1 - eta - zeta - eta * zeta,
                ],
                [
                    -1 + xi + zeta - xi * zeta,
                    -1 - xi + zeta + xi * zeta,
                    1 + xi - zeta - xi * zeta,
                    1 - xi - zeta + xi * zeta,
                    -1 + xi - zeta + xi * zeta,
                    -1 - xi - zeta - xi * zeta,
                    1 + xi + zeta + xi * zeta,
                    1 - xi + zeta - xi * zeta,
                ],
                [
                    -1 + xi + eta - xi * eta,
                    -1 - xi + eta + xi * eta,
                    -1 - xi - eta - xi * eta,
                    -1 + xi - eta + xi * eta,
                    1 - xi - eta + xi * eta,
                    1 + xi - eta - xi * eta,
                    1 + xi + eta + xi * eta,
                    1 - xi + eta - xi * eta,
                ],
            ],
            dtype=np.float64,
        )
    )

    J = dNdxi @ phys_coords

    # Jinv = 1/(J[1,1]*J[2,2] - J[1,2]*J[2,2])*np.array([ [J[2,2], -J[1,2]], [-J[2,1], J[1,1]]  ])
    # B = Jinv @ dNdxi

    B = np.linalg.solve(J, dNdxi).astype(np.float64)

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
