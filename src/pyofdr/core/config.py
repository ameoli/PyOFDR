from __future__ import annotations
import math
from pathlib import Path
from typing import Any
import yaml
from pyofdr.core.config_models import InstrumentConfig, RootConfig
from pyofdr.utils.constants import C
from pyofdr.utils.sampling import sweep_sample_count


def _read_config(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return raw


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the complete legacy YAML recipe. Returns a plain dict."""
    return RootConfig(**_read_config(path)).model_dump()


def load_instrument_config(path: str | Path) -> dict[str, Any]:
    """Load instrument settings without constructing or validating a fiber."""
    return InstrumentConfig(**_read_config(path)).model_dump()


def compute_instrument_derived(cfg: dict | InstrumentConfig) -> dict:
    """Clock and frequency quantities independent of a fiber model."""
    config = cfg if isinstance(cfg, InstrumentConfig) else InstrumentConfig(**cfg)
    source, adc = config.source, config.adc
    delta_nu = C / source.center_wavelength**2 * source.sweep_range
    gamma = delta_nu / source.sweep_duration
    n_t = sweep_sample_count(source.sweep_duration, adc.sample_rate)
    frequency_step_hz = gamma / adc.sample_rate
    return {
        "delta_nu": delta_nu,
        "gamma": gamma,
        "n_t": n_t,
        "frequency_step_hz": frequency_step_hz,
        "fft_sweep_range_hz": n_t * frequency_step_hz,
        "f_nyquist": adc.sample_rate / 2.0,
        "linewidth": source.linewidth,
    }


def compute_derived(cfg: dict) -> dict:
    """Compute derived physical quantities from config."""
    # validate first so we can trust the fields and not duplicate defaults
    config = RootConfig(**cfg)
    derived = compute_instrument_derived(config.instrument())
    cfg = config.model_dump()
    fiber  = cfg["fiber"]

    n        = fiber["n_core"]
    n_cores  = fiber["n_cores"]
    lw       = derived["linewidth"]
    L        = fiber["length"]

    # frequency range from wavelenght range
    delta_nu = derived["delta_nu"]
    gamma = derived["gamma"]
    dz = C / (2 * n * delta_nu)      # spatial resolution [m]
    # match what fiber/profile.py and source/swept_laser.py actually do
    n_z = int(math.ceil(L / dz))
    fft_sweep_range_hz = derived["fft_sweep_range_hz"]
    fft_dz = C / (2 * n * fft_sweep_range_hz)

    # max beat frequency -- needs to be below Nyquist
    f_beat_max = 2 * n * gamma * L / C

    # coherence length L_coh = c / (pi * n * linewidth)
    L_coh = C / (math.pi * n * lw) if lw > 0 else float("inf")

    derived.update({
        "dz": dz,
        "n_z": n_z,
        "fft_dz": fft_dz,
        "f_beat_max": f_beat_max,
        "L_coh": L_coh,
        "length": L,
        "n_cores": n_cores,
    })
    return derived


def sweep_period(cfg: dict) -> float:
    """Time between consecutive sweeps -- the slow-time step.

    This is the cadence at which the dynamic perturbations (strain,
    temperature) are sampled, not the within-sweep optical chirp. For the
    swept laser it's just the sweep duration (sweeps are back-to-back).
    """
    return cfg["source"]["sweep_duration"]


def print_instrument_info(cfg: dict | InstrumentConfig) -> None:
    """Summarize the source and ADC without a spatial calibration."""
    config = cfg if isinstance(cfg, InstrumentConfig) else InstrumentConfig(**cfg)
    d = compute_instrument_derived(config)
    print("PyOFDR instrument configuration:")
    print(f"  Center wavelength:  {config.source.center_wavelength*1e9:.3f} nm")
    print(f"  Sweep duration:     {config.source.sweep_duration*1e6:.3f} us")
    print(f"  Sweep range:        {d['delta_nu']*1e-9:.1f} GHz")
    print(f"  Chirp rate:         {d['gamma']:.6g} Hz/s")
    print(f"  Time samples:       {d['n_t']}")
    print(f"  ADC sample rate:    {config.adc.sample_rate*1e-6:.1f} MS/s")
    print(f"  Frequency step:     {d['frequency_step_hz']*1e-6:.3f} MHz")
    print(f"  Nyquist:            {d['f_nyquist']*1e-6:.1f} MHz")
    print(f"  Linewidth:          {d['linewidth']*1e-3:.1f} kHz")


def print_info(cfg: dict) -> None:
    """Print a summary of the configuration."""
    d = compute_derived(cfg)
    print(f"PyOFDR configuration summary:")
    print(f"  Fiber length:       {d['length']:.1f} m")
    if d["n_cores"] > 1:
        print(f"  Cores:              {d['n_cores']}")
    print(f"  Spatial resolution: {d['dz']*1e3:.3f} mm")
    print(f"  Spatial points:     {d['n_z']}")
    print(f"  Time samples:       {d['n_t']}")
    print(f"  Sweep range:        {d['delta_nu']*1e-9:.1f} GHz")
    print(f"  Max beat freq:      {d['f_beat_max']*1e-6:.1f} MHz")
    print(f"  Nyquist:            {d['f_nyquist']*1e-6:.1f} MHz")
    headroom = (d['f_nyquist'] - d['f_beat_max']) * 1e-6
    print(f"  Nyquist headroom:   {headroom:.1f} MHz")
    if d["linewidth"] > 0:
        print(f"  Linewidth:          {d['linewidth']*1e-3:.1f} kHz")
        print(f"  Coherence length:   {d['L_coh']:.1f} m")
        if d["length"] > d["L_coh"]:
            print(f"  WARNING: fiber length exceeds coherence length, expect fringe washout")
