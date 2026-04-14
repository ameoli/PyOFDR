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
pip install -r requirements.txt   # or just: pip install numpy scipy pyyaml pydantic pint h5py
```

## Quick start
```python
import sys
sys.path.insert(0, "src")

from core.config import load_config
from core.campaign import run_campaign

cfg = load_config("configs/ofdr_basic.yaml")
acqs = run_campaign(cfg)        # one Acquisition per sweep
acq  = acqs[-1]
# acq.digital_main contains the digitized beat signal (int16, shape (n_cores, n))
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
