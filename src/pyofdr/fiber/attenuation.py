"""Fiber round-trip attenuation.

Amplitude envelope for round trip. Power goes as exp(-2*alpha*z)
but we work with fields so its exp(-alpha*z).
"""

from __future__ import annotations

import math

import numpy as np

from pyrayleigh.backends import real_arrays
from pyrayleigh.units import attenuation_from_db_per_km

# log(10) is just a constant; precompute it so we don't depend on
# any particular array module here
_LN10 = math.log(10.0)


def attenuation_samples_from_segments(coefficient_dB_km, segments, z, *, xp=None):
    """Loss [dB/km] on half-open intervals; the last matching segment wins."""
    xp, values = real_arrays(z, coefficient_dB_km,
                             *(seg["attenuation_dB_per_km"] for seg in segments),
                             xp=None if xp is np else xp)
    z, base = values[:2]
    loss = xp.ones_like(z) * base
    for seg, value in zip(segments, values[2:]):
        inside = (z >= seg["start"]) & (z < seg["end"])
        loss = xp.where(inside, value, loss)
    return loss


def dB_per_km_to_neper_per_m(alpha_dB_km, *, xp=None):
    """dB/km to Np/m conversion."""
    return attenuation_from_db_per_km(alpha_dB_km, xp=None if xp is np else xp)


def round_trip_attenuation(z, alpha_dB_km, xp=None):
    """Round-trip amplitude exp(-alpha*z), with alpha the power loss [1/m]."""
    xp, (z, alpha_dB_km) = real_arrays(z, alpha_dB_km, xp=None if xp is np else xp)
    alpha = dB_per_km_to_neper_per_m(alpha_dB_km, xp=xp)
    return xp.exp(-alpha * z)


def round_trip_attenuation_varying(z, alpha_dB_km_z, dz, xp=None):
    """Envelope for a z-dependent loss, exp(-integral(alpha(z') dz')).

    alpha_dB_km_z is an array of the same length as z, one loss value
    per spatial bin. Left-Riemann cumulative integral, so the per-bin
    discretization error is O(dz * d(alpha)/dz). For typical SMF
    (alpha ~ 0.2 dB/km, dz ~ mm) this is ~1e-6 dB, well below any
    realistic loss budget. The loss at bin k applies to [z[k], z[k+1]).
    """
    xp, (z, alpha_dB_km_z, dz) = real_arrays(
        z, alpha_dB_km_z, dz, xp=None if xp is np else xp)
    loss_dB = cumulative_loss_dB(alpha_dB_km_z, dz, xp=xp)
    return xp.exp(-loss_dB * _LN10 / 10.0)


def cumulative_loss_dB(alpha_dB_km_z, dz, xp=None):
    """One-way distributed loss from the input to each bin, in dB."""
    xp, (alpha_dB_km_z, dz) = real_arrays(
        alpha_dB_km_z, dz, xp=None if xp is np else xp)
    if alpha_dB_km_z.ndim != 1:
        raise ValueError("loss samples must be a vector")
    if alpha_dB_km_z.shape[0] == 0:
        return xp.zeros_like(alpha_dB_km_z)
    loss = xp.cumulative_sum(alpha_dB_km_z[:-1], axis=0) * (dz / 1000.0)
    return xp.concat((xp.zeros_like(alpha_dB_km_z[:1]), loss))
