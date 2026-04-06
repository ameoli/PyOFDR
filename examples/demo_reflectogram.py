#!/usr/bin/env python
"""Plot a basic OFDR reflectogram.

Runs the pipeline on a 10m fiber and shows:
  (a) the beat signal in time domain
  (b) the reflectogram (FFT magnitude vs position)

This is the simplest thing you can do with PyOFDR.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.config import load_config
from core.campaign import run_campaign


def main():
    cfg_path = Path(__file__).parent.parent / "configs" / "ofdr_basic.yaml"
    cfg = load_config(cfg_path)
    print("Running simulation...")
    acq = run_campaign(cfg)
    print(f"Done. {acq.n_samples} samples, dz = {acq.dz*1e3:.3f} mm")

    # -- reflectogram --
    beat = acq.digital_main.astype(np.float64)
    H = np.fft.fft(beat)
    n_half = len(H) // 2
    z = np.arange(n_half) * acq.dz
    amp_dB = 20 * np.log10(np.abs(H[:n_half]) + 1e-30)

    # -- plot --
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    # beat signal (first 5000 samples)
    n_show = min(5000, len(beat))
    ax1.plot(acq.t[:n_show] * 1e6, beat[:n_show], lw=0.3)
    ax1.set_xlabel("Time [us]")
    ax1.set_ylabel("ADC counts")
    ax1.set_title("Beat signal (time domain)")
    ax1.grid(True, alpha=0.3)

    # reflectogram
    L = cfg["fiber"]["length"]
    mask = z <= L * 1.3
    ax2.plot(z[mask], amp_dB[mask], lw=0.4, color="C1")
    ax2.axvline(L, color="r", ls="--", lw=1, label=f"Fiber end ({L} m)")
    ax2.set_xlabel("Position along fiber [m]")
    ax2.set_ylabel("Amplitude [dB]")
    ax2.set_title(f"OFDR Reflectogram | dz = {acq.dz*1e3:.2f} mm")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("PyOFDR v0.1 -- Basic Reflectogram", fontweight="bold")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
