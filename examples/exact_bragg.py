"""Run with: python examples/exact_bragg.py --output out/bragg.h5

Scalar static fiber, zero laser linewidth and the exact PyRayleigh solver.
Read the coherent optical power terms back from HDF5.
"""

import argparse
from pathlib import Path

import numpy as np
from pyrayleigh import (BraggGrating, Fiber, FiberState, Field, FieldTerm, GuideOptics,
                       Section, SpatialProfile, lossless_reflector)
from pyrayleigh.units import attenuation_from_db_per_km, db_to_linear

from pyofdr.analysis.demodulation import reflectogram_from_acq
from pyofdr.core.campaign import run_campaign
from pyofdr.core.config import compute_instrument_derived, load_instrument_config
from pyofdr.output.hdf5_reader import load_sweep
from pyofdr.utils.constants import C


def main(output_path):
    config_path = Path(__file__).resolve().parents[1]/"configs/instrument_native.yaml"
    cfg = load_instrument_config(config_path)

    guide = GuideOptics(C/1550e-9, effective_index=1.444, group_index=1.4682)
    section = Section(0., 1., guide, attenuation_from_db_per_km(.2), db_to_linear(-82.))
    grating = BraggGrating(.4, .02, 1550e-9, 100., segment_length=.002)
    fiber = Fiber("bragg-fiber", 1., 293.15, 190e12, 200e12, [section],
                  elements=[lossless_reflector(.75, 1e-4, phase=.3)], gratings=[grating])
    fiber = fiber.with_rayleigh(17, cell_length=.01)

    strain = SpatialProfile([0., fiber.length], [1e-6, 1e-6])
    state = FiberState(axial_strain=Field([FieldTerm(strain)]))
    derived = compute_instrument_derived(cfg)
    # Nominal range scale; the grating also has its own spectral phase.
    spacing = C/(2*guide.group_index*derived["fft_sweep_range_hz"])
    options = dict(fiber=fiber, method="exact", loss_convention="background",
                   fft_dz=spacing, diagnostics=True)

    output_path = Path(output_path)
    reference_path = output_path.with_name(output_path.stem+"_reference"+output_path.suffix)
    cfg["output"]["path"] = str(reference_path)
    run_campaign(cfg, keep_acquisitions=False, **options)
    cfg["output"]["path"] = str(output_path)
    run_campaign(cfg, keep_acquisitions=False, state=state, **options)

    reference = load_sweep(reference_path)
    strained = load_sweep(output_path)
    peaks = []
    for name, acq in (("Reference", reference), ("Strained", strained)):
        response = acq.fiber_response
        reflectance = np.abs(response.reflection)**2
        peak = np.argmax(reflectance)
        wavelength = C/response.probe.frequency[peak]
        peaks.append(wavelength)
        spectrum, _ = reflectogram_from_acq(acq)
        print(f"{name}: sampled reflection peak at {wavelength*1e9:.6f} nm, "
              f"R = {reflectance[peak]:.4f}, FFT bins = {len(spectrum)}")
        diagnostic = acq.fiber_diagnostics
        # Total power includes the signed interference term.
        for port in ("reflection", "transmission"):
            total = np.mean(np.abs(getattr(diagnostic.total, port))**2)
            direct = np.mean(np.abs(getattr(diagnostic.direct, port))**2)
            residual = np.mean(np.abs(getattr(diagnostic.residual, port))**2)
            interference = np.mean(getattr(diagnostic, port+"_interference"))
            print(f"  Mean {port} terms (unit input): total={total:.6e}, direct={direct:.6e}, "
                  f"residual={residual:.6e}, interference={interference:+.6e}")
    print(f"Sampled peak shift for 1 microstrain: {(peaks[1]-peaks[0])*1e12:.3f} pm")
    print(f"HDF5: {reference_path}, {output_path}")
    return reference, strained


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interrogate a PyRayleigh Bragg fiber with PyOFDR")
    parser.add_argument("--output", default="out/bragg.h5")
    main(parser.parse_args().output)
