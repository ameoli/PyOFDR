# Architecture
For the physics see `theory.md`, for the YAML configuration see `config_reference.md`.

## What a run does
A single PyOFDR run is:
1. parse + validate a YAML config into pydantic models (`core/config.py`)
2. build the pieces (laser, fiber profile, reflectors, MZIs, detector, ADC)
3. for each sweep push samples through the signal chain
4. collect everything in an `Acquisition`
5. optionally stream the result to HDF5
The orchestrator lives in `core/`. Everything else is a building block it pulls in.

## Package layout
```
src/pyofdr/
  core/         orchestration -- config, pipeline, campaign, acquisition
  source/       swept laser + phase noise / RIN
  optics/       Mach-Zehnders + circulator + components
  fiber/        profile, Rayleigh, attenuation, bends, reflectors, strain, T
  detection/    photodetector + LPF + noise
  digitizer/    ADC chain (quantize, jitter, DNL/INL)
  utils/        constants, units, seeding, colored-noise helpers
  output/       HDF5 streaming writer
  analysis/     post-processing demod + quality metrics
  strain_transfer/  shear-lag coating coupling
  backends/     numpy / (planned) cupy
  cli.py        the `pyofdr` entry point
```

`__init__.py` only exposes `__version__`. The package is intentionally flat-ish:
import from the subpackage you need.

## Signal chain

```
                      +---------+
Swept laser --+-----> | aux MZI | --> aux reference (stored for post-processing)
              |       +---------+
              v
        +---------+     +-------+     +-----------+     +-----+
        | main MZ | --> | fiber | --> | photodet  | --> | ADC |
        +---------+     +-------+     +-----------+     +-----+
```

A sweep walks left to right. The aux MZI is tapped off the same laser to get a frequency reference used (in post-processing) to undo sweep nonlinearity.
The pipeline is straight-line: each stage takes an array, returns an array. `core/pipeline.py` only defines the `PipelineStep` base class -- the actual ordering of stages lives in `core/campaign.py`.

The aux MZI is *not* used inline: it just produces a reference signal that gets persisted in the `Acquisition` (and HDF5) alongside the main channel. The k-clock resampling that undoes sweep nonlinearity is done in post-processing under `analysis/`.

## Acquisition
`core.acquisition.Acquisition` is the dataclass that carries the output of one sweep -- digital main channel, aux MZI trace, timing, the source field, the strain / temperature fields actually applied, plus a log of which stages ran.
The HDF5 writer in `output/` is a thin layer on top.

## Campaign
Multi-sweep runs go through `core/campaign.py`. Each sweep gets its own `Acquisition`; the RNG seeding helpers in `utils/seeding.py` make the per-sweep streams reproducible without coupling the sweeps to each other.