"""Small unit conversion helpers."""

from utils.constants import C


def dB_to_linear(dB):
    """Power dB to linear"""
    return 10 ** (dB / 10)


def wavelength_range_to_freq_range(center_wl, range_wl):
    #Convert wavelenght range to frequency range
    return C / center_wl**2 * range_wl
