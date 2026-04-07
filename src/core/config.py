from __future__ import annotations
import math
from pathlib import Path
from typing import Any
import yaml
from core.config_models import RootConfig
from utils.constants import C


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate YAML config. Returns a plain dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return RootConfig(**raw).model_dump()


def compute_derived(cfg: dict) -> dict:
    """Compute derived physical quantities from config."""
    # validate first so we can trust the fields and not duplicate defaults
    cfg = RootConfig(**cfg).model_dump()
    fiber  = cfg["fiber"]
    source = cfg["source"]
    adc    = cfg["adc"]

    n        = fiber["n_core"]
    n_cores  = fiber["n_cores"]
    wl       = source["center_wavelength"]
    sweep_wl = source["sweep_range"]
    T_sweep  = source["sweep_duration"]
    lw       = source["linewidth"]
    fs       = adc["sample_rate"]
    L        = fiber["length"]

    # frequency range from wavelenght range
    delta_nu = C / wl**2 * sweep_wl
    gamma = delta_nu / T_sweep       # chirp rate [Hz/s]
    dz = C / (2 * n * delta_nu)      # spatial resolution [m]
    # match what fiber/profile.py and source/swept_laser.py actually do
    n_z = int(math.ceil(L / dz))
    n_t = int(math.ceil(T_sweep * fs))

    # max beat frequency -- needs to be below Nyquist
    f_beat_max = 2 * n * gamma * L / C

    # coherence length L_coh = c / (pi * n * linewidth)
    L_coh = C / (math.pi * n * lw) if lw > 0 else float("inf")

    return {
        "delta_nu": delta_nu,
        "gamma": gamma,
        "dz": dz,
        "n_z": n_z,
        "n_t": n_t,
        "f_beat_max": f_beat_max,
        "f_nyquist": fs / 2,
        "linewidth": lw,
        "L_coh": L_coh,
        "length": L,
        "n_cores": n_cores,
    }


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
