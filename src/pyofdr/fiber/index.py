"""Index perturbations sampled on the PyOFDR spatial grid."""

import numpy as np

from pyrayleigh import ornstein_uhlenbeck_index_from_normals
from pyrayleigh.backends import real_arrays


def index_samples_from_normals(normals, dz, sigma, correlation_length, *, xp=None):
    """OU samples on fixed spacing dz [m], from fixed independent N(0, 1) draws."""
    xp, (normals, sigma, correlation_length) = real_arrays(
        normals, sigma, correlation_length, xp=None if xp is np else xp)
    if normals.ndim != 1:
        raise ValueError("normals must be a vector")
    count = normals.shape[0]
    if count == 0:
        return xp.zeros_like(normals)
    # The closing knot is unused and needs no extra random draw.
    padded = xp.concat((normals, xp.zeros((1,), dtype=xp.float64)))
    profile = ornstein_uhlenbeck_index_from_normals(
        count*float(dz), sigma, correlation_length, dz, padded, xp=xp)
    return profile.values[:-1]
