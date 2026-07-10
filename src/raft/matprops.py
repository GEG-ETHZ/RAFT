"""All material property correlations are stored here."""

import numpy as np
import numpy.typing as npt


def get_viscosity(
    T: npt.NDArray[np.float64], viscosity_coefficients: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Get visocsity using a 4 coefficient approximation."""
    A = viscosity_coefficients[0]
    B = viscosity_coefficients[1]
    C = viscosity_coefficients[2]
    D = viscosity_coefficients[3]

    # Cap limits on temperature
    low_limit = 200.0
    up_limit = 1.0e3

    T = np.maximum(T, low_limit)
    T = np.minimum(T, up_limit)

    mu = A * np.exp(B / T + C * T + D * T**2)

    out_of_limits = (low_limit >= T) | (up_limit <= T)
    dmudT = np.where(out_of_limits, 0.0, mu * (-B / T**2 + C + 2 * D * T))

    return mu, dmudT
