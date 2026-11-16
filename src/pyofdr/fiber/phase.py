import numpy as np

from pyrayleigh.backends import real_arrays


def phase_at_bins(field, dz, prefactor, xp=None):
    """Round-trip phase at z[k] from the field in cells 0 through k-1."""
    xp, (field, dz, prefactor) = real_arrays(
        field, dz, prefactor, xp=None if xp is np else xp)
    if field.ndim != 1:
        raise ValueError("field must be a vector")
    if field.shape[0] == 0:
        return xp.zeros_like(field)
    phase = prefactor * xp.cumulative_sum(field[:-1], axis=0) * dz
    return xp.concat((xp.zeros_like(field[:1]), phase))
