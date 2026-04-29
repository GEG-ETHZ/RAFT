"""Get the Gauss quadrature points and weights."""

import numpy as np


def getquadrature(qt: int, ndim: int) -> np.NDarray[np.float64]:
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
