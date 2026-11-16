"""Index perturbations sampled on the PyOFDR spatial grid."""

import numpy as np

from pyrayleigh import index_segments_profile, ornstein_uhlenbeck_index_from_normals
from pyrayleigh.backends import real_arrays


def index_samples_from_segments(segments, z, *, xp=None):
    """Sum static index changes on half-open intervals, clipped at the fiber input."""
    xp, values = real_arrays(z, *(segment["delta_n"] for segment in segments),
                             xp=None if xp is np else xp)
    intervals = []
    for segment in segments:
        start = max(0., segment["start"])
        if segment["end"] > start:
            intervals.append((start, segment["end"], segment["delta_n"]))
    profile = index_segments_profile(intervals, xp=xp)
    return profile.value(values[0], xp=xp)


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
