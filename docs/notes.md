# PyOFDR -- Development Notes

## Physics model (what we simulate)

OFDR = Optical Frequency Domain Reflectometry

The laser sweeps linearly in frequency. Backscatter from each point
in the fiber produces a different beat frequency. FFT of the beat
gives position-resolved reflectivity.

Key equations:
- Beat frequency:  f_beat(z) = 2*n*gamma*z / c
- Spatial resolution: dz = c / (2*n*delta_nu)
- Backscatter: r(z) ~ circular Gaussian (Rayleigh model)

## What's implemented (v0.1)

1. Fiber: Rayleigh phasors, no attenuation
2. Source: ideal linear sweep, no noise
3. Optics: basic MZ, FFT-based backscatter
4. Detection: responsivity only
5. ADC: clip + quantize

## TODO for next versions

- [ ] fiber attenuation exp(-alpha*z)
- [ ] phase noise (this is critical for realistic simulations)
- [ ] shot noise, termal noise
- [ ] auxiliary MZI
- [ ] strain perturbations (the whole point really)
- [ ] HDF5 output
- [ ] proper config validation (pydantic?)
- [ ] multi-sweep campaigns
- [ ] CLI

## References

- Froggatt & Moore, "High-spatial-resolution distributed strain
  measurement in optical fiber with Rayleigh scatter", Appl Opt 1998
- Soller et al, "High resolution OFDR for characterization of
  components and assemblies", Opt Express 2005
- Hartog, "Intro to Distributed Optical Fibre Sensors", CRC 2017
