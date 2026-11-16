"""Build PyRayleigh objects from the validated PyOFDR configuration."""

from dataclasses import replace

import numpy as np

from pyrayleigh import (Fiber, FiberState, Field, FieldTerm, GuideOptics, MultiCoreFiber,
                        Section, SpatialProfile, WeakFBG,
                        index_segments_profile, lossless_reflector,
                        ornstein_uhlenbeck_index_profile)
from pyrayleigh.backends import real_arrays
from pyrayleigh.modes import (attenuation_from_loss_db, exponential_bend_loss_db,
                              marcuse_bend_loss_db, tabulated_bend_loss_db)
from pyrayleigh.units import (attenuation_from_db_per_km, db_to_linear,
                             dispersion_to_gvd_at_wavelength, wavelength_to_frequency)

from pyofdr.utils.seeding import derive_seed


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


def _bend_loss(bend, wavelength, *, xp):
    if bend["model"] == "exponential":
        return exponential_bend_loss_db(bend["radius"], bend["turns"],
                                        bend["A_dB_per_turn"], bend["R_c"], xp=xp)
    if bend["model"] == "tabulated":
        radii = [point["radius"] for point in bend["table"]]
        losses = xp.stack([xp.asarray(point["dB_per_turn"], dtype=xp.float64)
                           for point in bend["table"]])
        return tabulated_bend_loss_db(bend["radius"], bend["turns"], radii, losses, xp=xp)
    if bend["model"] == "marcuse":
        return marcuse_bend_loss_db(bend["radius"], bend["turns"], wavelength,
                                    **bend["marcuse"], xp=xp)
    raise ValueError(f"unknown bend model: {bend['model']!r}")


def build_sections(config, guide, *, xp=None):
    """Last loss/Rayleigh override wins; bends add at the source wavelength."""
    fiber = config["fiber"]
    attenuation = fiber["attenuation_segments"]
    rayleigh = fiber["rayleigh_segments"]
    bends = fiber["bends"]
    parameters = [guide.frequency_ref, fiber["attenuation_dB_per_km"], fiber["rayleigh_coefficient_dB"]]
    parameters.extend(segment["attenuation_dB_per_km"] for segment in attenuation)
    parameters.extend(segment["rayleigh_coefficient_dB"] for segment in rayleigh)
    parameters.append(config["source"]["center_wavelength"])
    for bend in bends:
        parameters.extend((bend["radius"], bend["turns"], bend["A_dB_per_turn"], bend["R_c"]))
        if bend["table"] is not None:
            parameters.extend(point["dB_per_turn"] for point in bend["table"])
        if bend["marcuse"] is not None:
            parameters.extend(bend["marcuse"].values())
    xp, _ = real_arrays(*parameters, xp=xp)
    bend_rates = [attenuation_from_loss_db(_bend_loss(bend, config["source"]["center_wavelength"], xp=xp),
                                          bend["end"] - bend["start"], xp=xp) for bend in bends]
    length = fiber["length"]
    boundaries = {0., length}
    for segment in (*attenuation, *rayleigh, *bends):
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
        alpha = attenuation_from_db_per_km(loss, xp=xp)
        for bend, coefficient in zip(bends, bend_rates):
            if bend["start"] <= start < bend["end"]:
                alpha = alpha + coefficient
        sections.append(Section(start, end, guide, alpha,
                                db_to_linear(rate, xp=xp)))
    return tuple(sections)


def build_reflectors(config, core=0, *, xp=None):
    """Point coefficients at configured positions, with input-side insertion loss."""
    reflectors = sorted(config["fiber"]["reflectors"], key=lambda item: item["z"])
    xp, _ = real_arrays(*(value for item in reflectors for value in (item["R"], item["loss_dB"])), xp=xp)
    seed = derive_seed(config["simulation"]["seed"], component="reflectors", core=core)
    rng = np.random.Generator(np.random.PCG64(seed))
    phases = xp.asarray(rng.uniform(0., 2.*np.pi, len(reflectors)), dtype=xp.float64)
    elements = []
    for index, item in enumerate(reflectors):
        point = lossless_reflector(item["z"], item["R"], phases[index], xp=xp)
        q = xp.sqrt(db_to_linear(-item["loss_dB"], xp=xp))
        elements.append(replace(point, r_left=point.r_left*q*q,
                                 t_left_to_right=point.t_left_to_right*q,
                                 t_right_to_left=point.t_right_to_left*q))
    return tuple(elements)


def build_fbgs(config, *, xp=None):
    """Weak gratings at configured positions, keeping the input order."""
    entries = config["fiber"]["fbg_arrays"]
    xp, values = real_arrays(*(value for item in entries for value in
                               (item["bragg_wavelength"], item["length"], item["peak_reflectivity"])), xp=xp)
    elements = []
    for index, item in enumerate(entries):
        wavelength, length, reflectivity = values[3*index:3*index + 3]
        elements.append(WeakFBG(item["z"], wavelength, length, reflectivity))
    return tuple(elements)


def build_index_profile(config, cell_length, *, xp=None):
    """Static delta n, with OU samples held constant over the fixed cells."""
    fiber = config["fiber"]
    segments = []
    for item in fiber["index_segments"]:
        start = max(0., min(fiber["length"], item["start"]))
        end = max(0., min(fiber["length"], item["end"]))
        if end > start:
            segments.append((start, end, item["delta_n"]))
    fluctuations = fiber["index_fluctuations"]
    parameters = [value for _, _, value in segments]
    if fluctuations is not None:
        parameters.extend((fluctuations["sigma"], fluctuations["correlation_length"]))
    xp, _ = real_arrays(*parameters, xp=xp)
    if fluctuations is not None:
        seed = derive_seed(config["simulation"]["seed"], component="index_fluctuations")
        sample = ornstein_uhlenbeck_index_profile(
            fiber["length"], fluctuations["sigma"], fluctuations["correlation_length"],
            cell_length, seed, xp=xp)
        # Keep left-cell values; the last OU knot only closes the fiber interval.
        segments.extend((sample.positions[index], sample.positions[index + 1], sample.values[index])
                        for index in range(len(sample.positions) - 1))
    return index_segments_profile(segments, xp=xp)


def build_core(config, core=0, *, cell_length, reference_temperature,
               frequency_min, frequency_max, xp):
    """One core on fixed cells, with reference temperature [K] and band [Hz].

    Pass one Array API namespace for all numerical components.
    """
    if not 0 <= core < config["fiber"]["n_cores"]:
        raise ValueError("core index is outside the configured fiber")
    guide = build_guide(config, xp=xp)
    sections = build_sections(config, guide, xp=xp)
    elements = build_reflectors(config, core, xp=xp) + build_fbgs(config, xp=xp)
    index = build_index_profile(config, cell_length, xp=xp)
    fiber = Fiber(f"core-{core}", config["fiber"]["length"], reference_temperature,
                  frequency_min, frequency_max, sections, elements, index_perturbation=index)
    seed = derive_seed(config["simulation"]["seed"], component="fiber", core=core)
    return fiber.with_rayleigh(seed, cell_length=cell_length, xp=xp)


def build_fiber(config, *, cell_length, reference_temperature, frequency_min, frequency_max, xp):
    """All configured cores, with stored crosstalk phases when enabled."""
    cores = [build_core(config, core, cell_length=cell_length,
                        reference_temperature=reference_temperature, frequency_min=frequency_min,
                        frequency_max=frequency_max, xp=xp)
             for core in range(config["fiber"]["n_cores"])]
    fiber = MultiCoreFiber(cores)
    coupling = config["fiber"]["crosstalk"]
    if coupling is None or fiber.n_cores == 1:
        return fiber
    seed = derive_seed(config["simulation"]["seed"], component="crosstalk")
    multiple = config["fiber"]["multiple_scattering"]
    order = 1 if multiple is None else multiple["max_order"]
    return fiber.with_crosstalk(seed, coupling["xt_dB_per_km"], topology=coupling["topology"],
                                max_order=order, xp=xp)


def build_state(config, cell_edges, mechanical_strain=None, temperature_change=None, *, xp):
    """Freeze sampled fields on fixed cells; inputs are scalars or one value per cell.

    Strain is mechanical and temperature is a change [K]. The last cell's
    value is kept at the end face unless an extra endpoint value is supplied.
    """
    edges = np.asarray(cell_edges)
    if edges.dtype.kind not in "iuf":
        raise TypeError("cell edges must contain fixed real numbers")
    edges = edges.astype(np.float64, copy=False)
    if (edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges))
            or np.any(np.diff(edges) <= 0.)):
        raise ValueError("cell edges must be a strictly increasing finite vector")
    if edges[0] != 0. or edges[-1] != config["fiber"]["length"]:
        raise ValueError("cell edges must cover the configured fiber")
    xp, (mechanical, temperature, alpha) = real_arrays(
        0. if mechanical_strain is None else mechanical_strain,
        0. if temperature_change is None else temperature_change,
        config["temperature"]["thermal_expansion"], xp=xp)
    count = len(edges) - 1
    if alpha.ndim != 0:
        raise ValueError("thermal expansion must be a scalar")
    endpoint = any(value.shape == (count + 1,) for value in (mechanical, temperature))
    values = []
    for value in (mechanical, temperature):
        if value.ndim == 0:
            value = xp.broadcast_to(value, (count + int(endpoint),))
        elif value.shape == (count,) and endpoint:
            value = xp.concat((value, value[-1:]))
        elif value.shape != (count + int(endpoint),):
            raise ValueError("fields need one value per cell and an optional endpoint")
        values.append(value)
    mechanical, temperature = values
    positions = np.repeat(edges, 2)[1:-1]
    if endpoint:
        positions = np.append(positions, edges[-1])
    fields = []
    for value in (mechanical + alpha*temperature, temperature):
        cells = value[:-1] if endpoint else value
        samples = xp.reshape(xp.stack((cells, cells), axis=1), (-1,))
        if endpoint:
            samples = xp.concat((samples, value[-1:]))
        fields.append(Field((FieldTerm(SpatialProfile(positions, samples)),)))
    return FiberState(axial_strain=fields[0], temperature_change=fields[1])
