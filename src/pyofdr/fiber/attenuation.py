"""Fiber round-trip attenuation.

Amplitude envelope for round trip. Power goes as exp(-2*alpha*z)
but we work with fields so its exp(-alpha*z).
"""

from __future__ import annotations

import math

import numpy as np

# log(10) is just a constant; precompute it so we don't depend on
# any particular array module here
_LN10 = math.log(10.0)


def dB_per_km_to_neper_per_m(alpha_dB_km):
    """dB/km to Np/m conversion."""
    return alpha_dB_km * _LN10 / (10.0 * 1000.0)


def round_trip_attenuation(z, alpha_dB_km, xp=np):
    """Amplitude attenuation envelope exp(-alpha*z).

    `xp` is the array module to use (numpy by default). Pass cupy
    or jax.numpy here when running on a non-numpy backend.
    """
    alpha = dB_per_km_to_neper_per_m(alpha_dB_km)
    return xp.exp(-alpha * z)


def round_trip_attenuation_varying(z, alpha_dB_km_z, dz, xp=np):
    """Envelope for a z-dependent loss, exp(-integral(alpha(z') dz')).

    alpha_dB_km_z is an array of the same length as z, one loss value
    per spatial bin. Left-Riemann cumulative integral, so the per-bin
    discretization error is O(dz * d(alpha)/dz). For typical SMF
    (alpha ~ 0.2 dB/km, dz ~ mm) this is ~1e-6 dB, well below any
    realistic loss budget. The loss at bin k applies to [z[k], z[k+1]).
    """
    loss_dB = cumulative_loss_dB(alpha_dB_km_z, dz, xp=xp)
    return xp.exp(-loss_dB * _LN10 / 10.0)


def cumulative_loss_dB(alpha_dB_km_z, dz, xp=np):
    """One-way distributed loss from the input to each bin, in dB."""
    loss = xp.zeros_like(alpha_dB_km_z, dtype=float)
    loss[1:] = xp.cumsum(alpha_dB_km_z[:-1]) * (dz / 1000.0)
    return loss
