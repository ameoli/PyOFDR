# PyOFDR

**Python simulator for Optical Frequency Domain Reflectometry**

End-to-end OFDR simulation, from the laser sweep to digitized samples.
Think of it as a virtual optical bench.

## What it does

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

Very early prototype. Currently implmented:

- [x] Rayleigh backscatter generation
- [x] Swept laser (ideal, no noise)  
- [x] Mach-Zehnder interferomter
- [x] Photodetector (basic)
- [x] ADC quantization
- [ ] Phase noise, RIN
- [ ] Shot noise, thermal noise
- [ ] Fiber attenuation
- [ ] Strain/temperature perturbations
- [ ] HDF5 output
- [ ] Auxiliary MZI for k-linearization

## Licence

MIT
