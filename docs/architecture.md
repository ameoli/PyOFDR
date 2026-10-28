# Architecture
For the physics see `theory.md`, for the YAML configuration see `config_reference.md`.

## What a run does
A single PyOFDR run is:

1. parse + validate a YAML config into pydantic models (`core/config.py`)
2. construct the pipeline steps once
3. create an `Acquisition` for each sweep and pass it through those steps
4. optionally write the completed sweep to HDF5
5. keep the acquisition in the returned list, or release it in streaming mode

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
  backends/     numpy / (planned) cupy and jax
  cli.py        the `pyofdr` entry point
```

`__init__.py` only exposes `__version__`. The package is intentionally flat-ish:
import from the subpackage you need.

## Pipeline and data flow

`core/campaign.py` defines the execution order:

```
FiberGenerator -> StrainPerturbation -> TemperaturePerturbation
               -> SweptLaser -> MachZehnder -> AuxMZI
               -> Detector -> AntiAliasFilter -> ADC
```

Each stage subclasses `PipelineStep` from `core/pipeline.py` and implements
`process(acq)`. It reads fields populated by earlier stages, updates the
`Acquisition`, and returns it. The runner uses `acq = step.process(acq)`.
There is no separate pipeline registry; adding a class does not insert it
into the campaign automatically.

The base constructor validates the configuration and sets `self.config` and
`self.bk`. Step objects live for the whole campaign, so caches survive between
sweeps. For example, `FiberGenerator` reuses the same unperturbed fiber.
Do not retain completed acquisitions in a step: that would defeat streaming.

| Stage | Main fields it supplies or updates |
| --- | --- |
| `FiberGenerator` | `z`, `dz`, `fiber_profile`, `attenuation_envelope`, `ghost_profile` |
| Strain and temperature | `fiber_profile`, `ghost_profile`, applied perturbation fields |
| `SweptLaser` | `t`, `dt`, `n_samples`, `E_source`, frequency arrays |
| `MachZehnder` | `photocurrent_main` (optical beat power, despite the name) |
| `AuxMZI` | `aux_signal`, `aux_valid_start` when enabled |
| `Detector` | `analog_main`, voltage including detector noise |
| `AntiAliasFilter` | filtered `analog_main` |
| `ADC` | `digital_main` |

The aux MZI is *not* used inline: it just produces a reference signal that gets persisted in the `Acquisition` (and HDF5) alongside the main channel. The k-clock resampling that undoes sweep nonlinearity is done in post-processing under `analysis/`.

## Acquisition
`core.acquisition.Acquisition` is the dataclass that carries the output of one sweep -- digital main channel, aux MZI trace, timing, the source field, the strain / temperature fields actually applied, plus a log of which stages ran.
The HDF5 writer in `output/` is a thin layer on top.

Array fields start as `None` until the relevant stage supplies them. Main-channel
arrays have shape `(n_cores, n_samples)`; `fiber_profile` has shape
`(n_cores, n_z)`, as does the optional `ghost_profile`. Ghosts already include
their path losses and are added after weighting the direct profile by
`attenuation_envelope`. The laser and auxiliary signal are shared across cores and
use one-dimensional arrays. Call `acq.add_log(name, ...)` for per-sweep
metadata. Adding a dataclass field alone does not save it to HDF5: datasets
are written explicitly in `output/hdf5_writer.py`.

When strain is present, each sweep stores its applied field in
`/sweeps/0000/strain_field`, `/sweeps/0001/strain_field`, etc., using
`/fiber/z` as the spatial axis. This also works for static strain.
The older `/fiber/strain_field` dataset is kept and contains the first
sweep's field; use the per-sweep datasets for dynamic strain.

## Configuration and units

`load_config(path)` in `core/config.py` reads YAML, validates it with
`RootConfig`, then returns `model_dump()` as a plain dictionary. Models in
`core/config_models.py` supply defaults and reject unknown keys. Both
`compute_derived(cfg)` and the `PipelineStep` constructor also validate input,
which lets tests and scripts pass partial dictionaries directly.

Unit-aware fields accept numbers in their declared target units or strings
parsed with Pint. For example:

```python
from pyofdr.core.config_models import RootConfig

cfg = RootConfig(adc={"jitter_rms": "50 ps"}).model_dump()
assert abs(cfg["adc"]["jitter_rms"] - 50e-12) < 1e-24
```

Time, length and frequency fields become seconds, metres and hertz. Some
parameters use explicit conventions instead, such as attenuation in dB/km;
check `config_reference.md` before adding conversions.

Validation includes positive dimensions, ADC limits, maximum beat frequency
below Nyquist, fiber features inside the fiber, and auxiliary delay bounds.
`compute_derived` calculates quantities such as spatial bin size and sample
count separately. To add a parameter, extend its Pydantic section with a
default and appropriate bounds; add a model validator for constraints that
depend on several fields.

## Array backends

Only NumPy is implemented. Selecting `cupy` or `jax` is rejected during config
validation; calling `get_backend` with either name directly raises
`NotImplementedError`.

Pipeline stages access array operations through `self.bk`:

- `xp` for array creation and elementwise operations;
- `fft` for transforms;
- `random_generator(seed)` for a local random generator;
- `sosfilt_lowpass` for the filter implementation;
- `asarray` and `to_numpy` for array conversion.

The NumPy backend uses SciPy for filtering. Analysis functions and the HDF5
writer currently use NumPy directly, so implementing another backend also
requires checking those boundaries. The abstraction alone does not provide
GPU support.

## Campaign
Multi-sweep runs go through `core/campaign.py`. Each sweep gets its own `Acquisition`; the RNG seeding helpers in `utils/seeding.py` make the per-sweep streams reproducible without coupling the sweeps to each other.

By default, `run_campaign(cfg)` returns every acquisition in a list, even
when HDF5 output is enabled. For long campaigns, skip that collection:

```python
from pyofdr.core.config import load_config
from pyofdr.core.campaign import run_campaign

cfg = load_config("configs/ofdr_basic.yaml")
cfg["simulation"]["n_sweeps"] = 100
cfg["output"]["path"] = "data/campaign.h5"
run_campaign(cfg, keep_acquisitions=False)
```

This mode requires `output.path` and returns an empty list. Each acquisition
is released after writing its sweep, so completed sweeps do not accumulate
in RAM. The shared fiber and other pipeline caches remain available during
the campaign. The HDF5 layout is unchanged; intermediate arrays that the
writer does not save are discarded. `keep_acquisitions` is a Python argument,
not a YAML setting.

The CLI uses this mode automatically when `pyofdr run` has an HDF5 output
path, either from the YAML file or from `-o`. No additional flag is needed.
Without an output path, it keeps the default campaign behavior.

### Seeding

`derive_seed` hashes the master seed, component ID, core, sweep and
substream together with SHA-256. Each field is written as a decimal
integer separated by `:`, with a `sha256-v1` prefix. The full digest is
converted to a Python integer and passed to the existing NumPy RNG.
Component IDs are fixed; adding a stage must not renumber the old ones.
There is no shared counter, so calling stages in a different order does
not change their seeds.

This replaces the additive offsets (#83), where laser sweep 1000 and
detector sweep 0 used the same seed. The old core and substream strides
had similar overlaps. The new scheme removes those systematic collisions;
hashing and RNG initialization still have a finite state space, so this
is not a mathematical guarantee of independent streams.

The random realisations change from the old scheme, including the fiber
profile. Repeating the same config with this code remains deterministic.
Fiber profiles and ADC DNL curves still stay fixed across sweeps, while
the per-sweep noise changes. Random strain motion uses its own `(seed, t)`
scheme and is unaffected. HDF5 output records the scheme in the root
`seed_scheme` attribute; older files do not have that attribute.

## Adding a pipeline step

This fixed voltage gain is a small example, not an existing simulator stage.
It belongs after `Detector` and before `ADC`, where `analog_main` is available:

```python
from pyofdr.core.acquisition import Acquisition
from pyofdr.core.pipeline import PipelineStep


class VoltageGain(PipelineStep):
    name = "voltage_gain"

    def process(self, acq: Acquisition) -> Acquisition:
        if acq.analog_main is None:
            raise RuntimeError("VoltageGain: analog_main is None")

        gain = 2.0
        acq.analog_main = acq.analog_main * gain
        acq.add_log(self.name, gain=gain)
        return acq
```

To integrate a real stage:

1. Put the class in the relevant subpackage. If it needs an `__init__`, call
   `super().__init__(config)` before reading `self.config` or `self.bk`.
2. Add configurable parameters to `core/config_models.py`. Keep an optional
   effect disabled by default so existing configurations retain their behavior.
3. Import the class inside `run_campaign` and insert it in the `steps` list.
   For this example, use `VoltageGain(cfg)` between `Detector(cfg)` and
   `AntiAliasFilter(cfg)`. Ordering determines which signals the stage changes.
4. For stochastic effects, register a new component ID in `utils/seeding.py`
   without changing existing IDs. Use `derive_seed` with the core and sweep
   indices for changing noise; omit the sweep index for fixed imperfections.
   Create generators through `self.bk.random_generator`.
5. Add tests under `tests/test_<topic>.py`: check a known input/output,
   disabled behavior, missing inputs and array shapes. For random stages,
   check reproducibility and sweep behavior. Test the stage in a campaign too.
6. Update the configuration reference and theory guide for new physical
   parameters. Extend `Acquisition` and the HDF5 writer only if new outputs
   need to be carried or saved.

Run the relevant tests with `python -m pytest tests/test_<topic>.py`.
The Windows virtual environment can be used directly as
`.venv\Scripts\python.exe -m pytest tests/test_campaign.py`.
