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

    B = np.linalg.solve(J, dNdxi)

    return B, J
