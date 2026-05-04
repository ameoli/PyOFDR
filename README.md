# PyOFDR

![tests](https://github.com/ameoli/PyOFDR/actions/workflows/tests.yml/badge.svg)
![license](https://img.shields.io/github/license/ameoli/PyOFDR)
![python](https://img.shields.io/badge/python-3.10%2B-blue)

**Python simulator for Optical Frequency Domain Reflectometry**
End-to-end OFDR simulation, from the laser sweep to digitized samples.
Think of it as a virtual optical bench.

## Key Feature
Simulates the full OFDR measurement chain:

```
Swept laser  -->  Mach-Zehnder interferometer  -->  Fiber (Rayleigh)  -->  Detector  -->  ADC
```

Outputs digitized beat signal identical to what you'd get from a real instrument (Luna OBR etc).

## What's simulated
- **Source**: linear sweep with nonlinearity and ripple, phase noise (Wiener + flicker + RW), RIN
- **Interferometer**: main MZ + auxiliary MZI for k-clock resampling, optical circulator
- **Fiber**: Rayleigh backscatter, attenuation, macrobend loss, discrete reflectors, weak FBG arrays,
  stochastic n(z), chromatic dispersion (GVD)
- **Strain**: static + dynamic (harmonic, impulsive, thermal, random vibration), Cox shear-lag coupling
- **Detection**: balanced photodetector, shot/thermal/dark-current noise, anti-alias + Butterworth LPF
- **ADC**: clip, quantize, jitter, DNL/INL, ENOB
- **Output**: HDF5 streaming, multi-sweep campaigns

## Install
```bash
pip install -e .              # editable install from a clone
pip install -e .[dev]         # + pytest (for the test suite)
```

## Quick start
```python
from pyofdr.core.config import load_config
from pyofdr.core.campaign import run_campaign

cfg = load_config("configs/ofdr_basic.yaml")
acqs = run_campaign(cfg)        # one Acquisition per sweep
acq  = acqs[-1]
# acq.digital_main contains the digitized beat signal (int16, shape (n_cores, n))
```

The `pyofdr` CLI is also installed:
```bash
pyofdr info     configs/ofdr_basic.yaml
pyofdr validate configs/ofdr_basic.yaml
pyofdr run -o out.h5 configs/ofdr_basic.yaml
```

## Run tests
```bash
pytest tests/ -v
```

## Status
Alpha. End-to-end pipeline working.
V1.0 in preparation -- API may still change.

## Licence
MIT

### Third-party code
- [`felixpatzelt/colorednoise`](https://github.com/felixpatzelt/colorednoise) (MIT),
  vendored as `src/pyofdr/utils/colorednoise.py` and used for the laser phase-noise
  PSD shaping. License in `LICENSES/colorednoise-MIT.txt`.

## Issues & Contributions
Report issues: https://github.com/ameoli/PyOFDR/issues
Pull requests welcome!

## References
Froggatt & Moore, "High-spatial-resolution distributed strain measurement in optical fiber with Rayleigh scatter", Appl Opt 1998
Soller et al, "High resolution OFDR for characterization of components and assemblies", Opt Express 2005
Hartog, "Intro to Distributed Optical Fibre Sensors", CRC 2017
