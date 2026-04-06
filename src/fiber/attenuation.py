"""Fiber round-trip attenuation.

Amplitude envelope for round trip. Power goes as exp(-2*alpha*z)
but we work with fields so its exp(-alpha*z).
"""

from __future__ import annotations

import numpy as np


def dB_per_km_to_neper_per_m(alpha_dB_km):
    """dB/km to Np/m conversion."""
    return alpha_dB_km * np.log(10) / (10.0 * 1000.0)


def round_trip_attenuation(z, alpha_dB_km):
    """Amplitude attenuation envelope exp(-alpha*z)."""
    alpha = dB_per_km_to_neper_per_m(alpha_dB_km)
    return np.exp(-alpha * z)
