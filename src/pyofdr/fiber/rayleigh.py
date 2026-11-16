"""Rayleigh backscatter rates on the PyOFDR spatial grid."""

import numpy as np

from pyrayleigh.backends import real_arrays
from pyrayleigh.units import db_to_linear


def rayleigh_rates_from_segments(coefficient_dB, segments, z, *, xp=None):
    """Power per metre; the last matching half-open segment overrides the base rate."""
    xp, values = real_arrays(z, coefficient_dB,
                             *(seg["rayleigh_coefficient_dB"] for seg in segments),
                             xp=None if xp is np else xp)
    z, base = values[:2]
    rate = xp.ones_like(z) * db_to_linear(base, xp=xp)
    for seg, value in zip(segments, values[2:]):
        inside = (z >= seg["start"]) & (z < seg["end"])
        rate = xp.where(inside, db_to_linear(value, xp=xp), rate)
    return rate
