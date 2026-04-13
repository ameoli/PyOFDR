# PyOFDR

![tests](https://github.com/ameoli/PyOFDR/actions/workflows/tests.yml/badge.svg)
![license](https://img.shields.io/github/license/ameoli/PyOFDR)
![python](https://img.shields.io/badge/python-3.10%2B-blue)

**Python simulator for Optical Frequency Domain Reflectometry**
End-to-end OFDR simulation, from the laser sweep to digitized samples.
Think of it as a virtual optical bench.

## Key Feature
Simulates the full OFDR measureement chain:

```
Swept laser  -->  Mach-Zehnder interferometer  -->  Fiber (Rayleigh)  -->  Detector  -->  ADC
```

Outputs digitized beat signal identical to what you'd get from a real instrument (Luna OBR etc).

## Install
```bash
pip install -e .
```

## Quick start
```python
from pyofdr import load_config, run_campaign

config = load_config("configs/ofdr_basic.yaml")
acq = run_campaign(config)
# acq.digital contains the digitized beat signal
```

## Run tests
```bash
pytest tests/ -v
```

## Status
Very early prototype

## Licence
MIT

## Issues & Contributions
Report issues: https://github.com/ameoli/PyOFDR/issues
Pull requests welcome!

## References
Froggatt & Moore, "High-spatial-resolution distributed strain measurement in optical fiber with Rayleigh scatter", Appl Opt 1998
Soller et al, "High resolution OFDR for characterization of components and assemblies", Opt Express 2005
Hartog, "Intro to Distributed Optical Fibre Sensors", CRC 2017
