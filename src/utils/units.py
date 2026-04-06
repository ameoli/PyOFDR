from utils.constants import C
import numpy as np


def wavelength_to_frequency(wl):
    """Wavelenght [m] to frequency [Hz]"""
    return C / wl


def dB_to_linear(dB):
    """Power dB to linear"""
    return 10 ** (dB / 10)


def dB_to_amplitude(dB):
    """Power dB to amplitude (field)"""
    return 10 ** (dB / 20)


def wavelength_range_to_freq_range(center_wl, range_wl):
    #Convert wavelenght range to frequency range
    return C / center_wl**2 * range_wl
