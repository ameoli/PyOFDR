from __future__ import annotations
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
    """Compute derived physical quantities from config.

    Returns a dict with stuff like spatial resolution, sweep rate, etc.
    Usefull for sanity checks and info printing.
    """
    fiber = cfg.get("fiber", {})
    source = cfg.get("source", {})
    adc = cfg.get("adc", {})

    n = fiber.get("n_core", 1.4682)
    wl = source.get("center_wavelength", 1550e-9)
    sweep_wl = source.get("sweep_range", 40e-9)
    T_sweep = source.get("sweep_duration", 0.01)
    fs = adc.get("sample_rate", 200e6)
    L = fiber.get("length", 10.0)

    # frequency range from wavelenght range
    delta_nu = C / wl**2 * sweep_wl
    gamma = delta_nu / T_sweep       # chirp rate [Hz/s]
    dz = C / (2 * n * delta_nu)      # spatial resolution [m]
    n_z = int(L / dz)
    n_t = int(T_sweep * fs)

    # max beat frequency -- needs to be below Nyquist
    f_beat_max = 2 * n * gamma * L / C

    return {
        "delta_nu": delta_nu,
        "gamma": gamma,
        "dz": dz,
        "n_z": n_z,
        "n_t": n_t,
        "f_beat_max": f_beat_max,
        "f_nyquist": fs / 2,
    }


def print_info(cfg: dict) -> None:
    """Print a summary of the configuration."""
    d = compute_derived(cfg)
    print(f"PyOFDR configuration summary:")
    print(f"  Fiber length:       {cfg.get('fiber', {}).get('length', 10.0):.1f} m")
    print(f"  Spatial resolution: {d['dz']*1e3:.3f} mm")
    print(f"  Spatial points:     {d['n_z']}")
    print(f"  Time samples:       {d['n_t']}")
    print(f"  Sweep range:        {d['delta_nu']*1e-9:.1f} GHz")
    print(f"  Max beat freq:      {d['f_beat_max']*1e-6:.1f} MHz")
    print(f"  Nyquist:            {d['f_nyquist']*1e-6:.1f} MHz")
    if d["f_beat_max"] > d["f_nyquist"]:
        print(f"  WARNING: beat frequency exeeds Nyquist!")
