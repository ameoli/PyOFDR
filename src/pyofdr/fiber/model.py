"""Build PyRayleigh objects from the validated PyOFDR configuration."""

from pyrayleigh import GuideOptics
from pyrayleigh.backends import real_arrays
from pyrayleigh.units import dispersion_to_gvd_at_wavelength, wavelength_to_frequency


def build_guide(config, *, xp=None):
    """Reference optical parameters; PyRayleigh uses total axial strain."""
    fiber = config["fiber"]
    temperature = config["temperature"]
    xp, (wavelength, index, dispersion, pe, alpha, xi) = real_arrays(
        config["source"]["center_wavelength"], fiber["n_core"], fiber["dispersion_D"],
        config["strain"]["photoelastic_coefficient"], temperature["thermal_expansion"],
        temperature["thermo_optic"], xp=xp)
    thermal = xi + pe*alpha
    return GuideOptics(
        frequency_ref=wavelength_to_frequency(wavelength, xp=xp),
        effective_index=index,
        group_index=index,
        gvd=dispersion_to_gvd_at_wavelength(dispersion, wavelength, xp=xp),
        strain_phase_factor=1. - pe,
        strain_group_factor=1. - pe,
        temperature_phase_factor=thermal,
        temperature_group_factor=thermal,
        thermal_expansion=alpha,
    )
