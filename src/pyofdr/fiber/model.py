"""Build PyRayleigh objects from the validated PyOFDR configuration."""

from pyrayleigh import GuideOptics, Section
from pyrayleigh.backends import real_arrays
from pyrayleigh.units import (attenuation_from_db_per_km, db_to_linear,
                             dispersion_to_gvd_at_wavelength, wavelength_to_frequency)


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


def build_sections(config, guide, *, xp=None):
    """Split distributed loss and Rayleigh segments; the last override wins."""
    fiber = config["fiber"]
    attenuation = fiber["attenuation_segments"]
    rayleigh = fiber["rayleigh_segments"]
    parameters = [guide.frequency_ref, fiber["attenuation_dB_per_km"], fiber["rayleigh_coefficient_dB"]]
    parameters.extend(segment["attenuation_dB_per_km"] for segment in attenuation)
    parameters.extend(segment["rayleigh_coefficient_dB"] for segment in rayleigh)
    xp, _ = real_arrays(*parameters, xp=xp)
    length = fiber["length"]
    boundaries = {0., length}
    for segment in (*attenuation, *rayleigh):
        boundaries.add(max(0., min(length, segment["start"])))
        boundaries.add(max(0., min(length, segment["end"])))
    boundaries = sorted(boundaries)
    sections = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        loss = fiber["attenuation_dB_per_km"]
        rate = fiber["rayleigh_coefficient_dB"]
        for segment in attenuation:
            if segment["start"] <= start < segment["end"]:
                loss = segment["attenuation_dB_per_km"]
        for segment in rayleigh:
            if segment["start"] <= start < segment["end"]:
                rate = segment["rayleigh_coefficient_dB"]
        sections.append(Section(start, end, guide, attenuation_from_db_per_km(loss, xp=xp),
                                db_to_linear(rate, xp=xp)))
    return tuple(sections)
