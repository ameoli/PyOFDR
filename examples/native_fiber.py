"""Run with: python examples/native_fiber.py --output out/native.h5"""

import argparse
from pathlib import Path

import numpy as np
from pyrayleigh import (Fiber, FiberState, Field, FieldTerm, GuideOptics, Section,
                       SpatialProfile, lossless_reflector)
from pyrayleigh.units import attenuation_from_db_per_km, db_to_linear

from pyofdr.analysis.demodulation import reflectogram_from_acq
from pyofdr.core.campaign import run_campaign
from pyofdr.core.config import compute_instrument_derived, load_instrument_config
from pyofdr.utils.constants import C


def main(output_path):
    config_path = Path(__file__).resolve().parents[1]/"configs/instrument_native.yaml"
    cfg = load_instrument_config(config_path)

    guide = GuideOptics(C/1550e-9, effective_index=1.444, group_index=1.4682)
    section = Section(0., 1., guide, attenuation_from_db_per_km(.2), db_to_linear(-82.))
    fiber = Fiber("example-fiber", 1., 293.15, 190e12, 200e12, [section],
                  [lossless_reflector(.75, 1e-4, phase=.3)])
    fiber = fiber.with_rayleigh(17, cell_length=.01)

    strain = SpatialProfile([.2, .2, .6, .6], [0., 2e-6, 2e-6, 0.])
    state = FiberState(axial_strain=Field([FieldTerm(strain)]))

    # Nominal distance scale for this uniform guide.
    derived = compute_instrument_derived(cfg)
    spacing = C/(2*guide.group_index*derived["fft_sweep_range_hz"])

    output_path = Path(output_path)
    reference_path = output_path.with_name(output_path.stem+"_reference"+output_path.suffix)
    cfg["output"]["path"] = str(reference_path)
    reference = run_campaign(cfg, fiber=fiber, fft_dz=spacing)[0]
    cfg["output"]["path"] = str(output_path)
    strained = run_campaign(cfg, fiber=fiber, state=state, fft_dz=spacing)[0]

    H_ref, z = reflectogram_from_acq(reference)
    H_strained, _ = reflectogram_from_acq(strained)
    peak = np.argmax(np.abs(H_ref))
    phase_change = np.angle(H_strained[peak]*np.conj(H_ref[peak]))
    print(f"Peak position: {z[peak]:.4f} m (reflector at 0.7500 m)")
    print(f"Phase change at peak: {phase_change:.3f} rad")
    print(f"HDF5: {reference_path}, {output_path}")
    return reference, strained


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use a PyRayleigh fiber in PyOFDR")
    parser.add_argument("--output", default="out/native.h5")
    main(parser.parse_args().output)
