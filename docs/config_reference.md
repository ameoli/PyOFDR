# Config reference

Every PyOFDR run starts from a YAML config validated against the pydantic
models in `core/config_models.py`. This page is the schema reference; for
*why* a field exists see `theory.md`, for the run pipeline see
`architecture.md`.

## Conventions

- **SI everywhere** inside the validated dict. The YAML accepts unit strings
  (`"40 nm"`, `"10 ms"`, `"17 ps/(nm km)"`) via pint, or bare numbers already
  in SI.
- Numeric inputs must be finite, including list entries and values converted
  from unit strings.
- **`extra = "forbid"`** on every block: typos in field names raise on load.
- All defaults below are taken straight from the model -- omit a field to
  get its default.

A minimal valid config is the empty dict `{}`: every block has defaults that
produce a 10 m fiber, 1550 nm source, single sweep, no perturbations.

## simulation

Global run controls: master seed, compute backend, campaign size.

| field | type | default | description |
| --- | --- | --- | --- |
| `seed` | int >= 0 | 42 | master RNG seed; per-stage streams are derived from this |
| `backend` | `"numpy"` \| `"cupy"` \| `"jax"` | `"numpy"` | only `numpy` is wired today |
| `n_sweeps` | int >= 1 | 1 | number of sweeps in the campaign |

```yaml
simulation:
  seed: 1234
  n_sweeps: 20
```

Seed derivation changed with #83. The same config still reproduces a run
with the current code, but the random realisation differs from older
versions, even for a single sweep. HDF5 files now record `seed_scheme`
as a root attribute (`sha256-v1`). Keep the code version together with
the config when saving results; the seed alone does not identify a run
across versions.

## fiber

Top-level fields:

| field | type | default | description |
| --- | --- | --- | --- |
| `length` | length (m) | 10.0 | total fiber length |
| `n_core` | float > 1 | 1.4682 | core group index |
| `n_cores` | int >= 1 | 1 | multi-core fibers use n_cores > 1 |
| `rayleigh_coefficient_dB` | float | -82.0 | baseline backscatter coefficient |
| `attenuation_dB_per_km` | float >= 0 | 0.0 | baseline attenuation outside any segment |
| `dispersion_D` | s/m^2 | 0.0 | GVD parameter; SMF-28 @ 1550 ~= `"17 ps/(nm km)"` |

Lists (all empty by default):

- `rayleigh_segments[]` -- override backscatter on `[start, end]`
- `attenuation_segments[]` -- override attenuation on `[start, end]`
- `bends[]` -- macrobending loss regions
- `reflectors[]` -- discrete reflectors (connectors, splices)
- `fbg_arrays[]` -- weak FBGs (Born approx)
- `index_segments[]` -- static piecewise delta_n perturbations

Nested blocks (all optional, default `None`):

- `index_fluctuations` -- stochastic delta_n as Ornstein-Uhlenbeck process
- `crosstalk` -- core-to-core MCF crosstalk (level A, phase-scrambled)
- `multiple_scattering` -- multi-bounce ghosts between discrete reflectors

### fiber.rayleigh_segments[]

| field | type | description |
| --- | --- | --- |
| `start`, `end` | length | segment bounds, `end > start` |
| `rayleigh_coefficient_dB` | float | local override |

### fiber.attenuation_segments[]

| field | type | description |
| --- | --- | --- |
| `start`, `end` | length | segment bounds |
| `attenuation_dB_per_km` | float >= 0 | local override |

### fiber.bends[]

Macrobending. Choose `exponential` for the empirical `A*exp(-R/R_c)` law,
`tabulated` for measured loss per turn, or `marcuse` for an analytic LP01
step-index model. Tabulated loss uses log-linear interpolation in radius
and clamps outside the table range with a warning.

| field | type | default | description |
| --- | --- | --- | --- |
| `start`, `end` | length | -- | segment bounds |
| `radius` | length > 0 | -- | bend radius |
| `turns` | float > 0 | -- | number of turns over the segment |
| `model` | `exponential` / `tabulated` / `marcuse` | `exponential` | loss model |
| `A_dB_per_turn` | float > 0 | 100.0 | empirical prefactor (exponential) |
| `R_c` | length > 0 | 5e-3 | critical radius (exponential) |
| `table[]` | `{radius, dB_per_turn}`, >= 2 | -- | loss points (tabulated) |
| `marcuse` | mapping below | -- | waveguide parameters, required only for `marcuse` |

The `marcuse` mapping contains:

| field | type | default | description |
| --- | --- | --- | --- |
| `core_radius` | length > 0 | -- | radius of the core, not the cladding |
| `core_index` | float > 1 | -- | core phase index at the source centre wavelength |
| `cladding_index` | float >= 1 | -- | cladding phase index, smaller than `core_index` |
| `radius_correction` | float > 0 | 1.28 | `R_eff/R`; use 1.0 to disable the silica stress correction |

These phase indices are independent of `fiber.n_core`, which sets the OFDR
group delay. The calculation uses `source.center_wavelength`; the loss stays
constant during a sweep. `pyofdr validate` rejects non-finite parameters,
`V >= 2.4048255577`, relative index difference `delta > 0.01`, and bend or
effective radii no larger than the core. Extremely weak confinement that the
mode solver cannot resolve is also rejected. The delta bound is a conservative
implementation limit for the weak-guidance approximation.

For example, using the waveguide parameters from Martins et al. (2009),
Table I (`NA=0.117`, so `core_index=sqrt(1.440^2+0.117^2)`):

```yaml
fiber:
  length: "1 m"
  bends:
    - start: "0.2 m"
      end: "0.263 m"
      radius: "10 mm"
      turns: 1
      model: marcuse
      marcuse:
        core_radius: "4.1 um"
        core_index: 1.444745306273739
        cladding_index: 1.440
        radius_correction: 1.28
source:
  center_wavelength: "1550 nm"
```

This gives about **3.3562 dB one-way** for the turn. The interval places that
loss along the simulated fiber; the arc length used in the formula is
`2*pi*radius*turns`. These example parameters are not universal SMF-28 defaults.
All three models integrate the traversed fraction of `[start, end)` at each
spatial sample. Bounds do not need to align with the grid: samples at or beyond
`end` see the full bend loss, and overlapping bends add in dB.
See [the theory guide](theory.md#macrobending) for equations, references and limits.

### fiber.reflectors[]

| field | type | default | description |
| --- | --- | --- | --- |
| `z` | length | -- | position along the fiber |
| `R` | float in [0, 1] | -- | power reflectivity |
| `loss_dB` | float >= 0 | 0.0 | extra one-way loss on the input side of the reflecting interface |

Each reflector gets a seeded phase per core, fixed across sweeps. Direct
returns beyond it include the round-trip field transmission `1-R` as well as
`loss_dB`. Reflectors must occupy distinct bins after spatial rounding.
`loss_dB` excludes the power removed by reflection; see the phase and loss
conventions in [the theory guide](theory.md#discrete-reflectors-and-connectors).

### fiber.fbg_arrays[]

Weak FBGs (Born approximation; safe for `peak_reflectivity <= ~0.5`).
Strong gratings via CMT / transfer matrix tracked in #78.

| field | type | description |
| --- | --- | --- |
| `z` | length | grating start position |
| `bragg_wavelength` | length | unperturbed Bragg wavelength |
| `length` | length > 0 | physical grating length |
| `peak_reflectivity` | float in [0, 1] | R_max on-Bragg |

The resonance shifts at each sweep with the applied strain and temperature
at the grating's z-bin. It uses `strain.photoelastic_coefficient`,
`temperature.thermal_expansion` and `temperature.thermo_optic`; no extra FBG
setting is needed. The local values represent the whole grating, so gradients
within its length do not broaden the spectrum. See the FBG section in
`theory.md` for the shift formula and approximations.

FBG propagation phase follows the same warped time as the main Rayleigh beat,
including sweep errors, ripple, dispersion and slow frequency noise. The
spectral envelope uses the instantaneous laser frequency. The main Nyquist
guard covers the grating carrier, but not the envelope's bandwidth.

### fiber.index_segments[]

Static delta_n perturbation. Small-signal: bin positions don't move, only
the round-trip phase does. For the full OPL-based n(z) remap see #33.

| field | type | description |
| --- | --- | --- |
| `start`, `end` | length | segment bounds |
| `delta_n` | float (signed) | deviation from `n_core` |

### fiber.index_fluctuations

Stochastic stationary delta_n as an Ornstein-Uhlenbeck process.

| field | type | description |
| --- | --- | --- |
| `sigma` | float >= 0 | stationary standard deviation of delta_n (default 0.0) |
| `correlation_length` | length > 0 | OU correlation length (required) |

### fiber.crosstalk

Multi-core fiber core-to-core leakage. Level A (phase-scrambled scalar).
Full coupled-mode equations in #69.

| field | type | description |
| --- | --- | --- |
| `xt_dB_per_km` | float | integrated per-pair XT after 1 km (negative) |
| `topology` | `"hex7"` \| `"linear"` | adjacency pattern (default `hex7`) |

### fiber.multiple_scattering

Cascading multi-bounce ghosts between discrete reflectors (#35). Order N
enumerates paths with 2N-1 reflection events; N=2 is the dominant case
(e.g. input connector + end face -> ghost at `2*z_b - z_a`).

| field | type | default | description |
| --- | --- | --- | --- |
| `max_order` | int in [2, 3] | 2 | maximum bounce order |

```yaml
fiber:
  length: "12 m"
  n_core: 1.4682
  reflectors:
    - {z: "0 m",     R: 1.0e-3, loss_dB: 0.5}
    - {z: "5.05 m",  R: 1.7e-3}
    - {z: "12 m",    R: 0.04}
  attenuation_dB_per_km: 0.2
  bends:
    - {start: "5 m", end: "10 m", radius: "37 mm", turns: 40}
  dispersion_D: "17 ps/(nm km)"
```

## source

Swept laser + noise + sweep imperfections.

| field | type | default | description |
| --- | --- | --- | --- |
| `center_wavelength` | length > 0 | 1550e-9 | nu_0 = c / lambda_0 |
| `sweep_range` | length > 0 | 40e-9 | optical bandwidth covered per sweep |
| `sweep_duration` | time > 0 | 0.01 | sweep period (lab time) |
| `power` | watt > 0 | 10e-3 | average optical power |
| `linewidth` | Hz >= 0 | 0.0 | FWHM Lorentzian (white FM term) |
| `flicker_noise_Hz` | float >= 0 | 0.0 | RMS 1/f frequency excursion |
| `random_walk_noise_Hz` | float >= 0 | 0.0 | RMS 1/f^2 frequency excursion |
| `rin_dB_per_Hz` | float \| None | None | RIN PSD; None disables |
| `power_envelope_edge_dB` | float >= 0 | 0.0 | parabolic edge droop |
| `sweep_nonlinearity_a2` | Hz/s^2 | 0.0 | quadratic chirp deviation |
| `sweep_nonlinearity_a3` | Hz/s^3 | 0.0 | cubic chirp deviation |
| `sweep_ripple_amplitude` | Hz | 0.0 | sinusoidal ripple on nu(t) |
| `sweep_ripple_period` | time | 0.0 | ripple period; 0 disables |

```yaml
source:
  center_wavelength: "1550 nm"
  sweep_range: "40 nm"
  sweep_duration: "10 ms"
  power: "5 mW"
  linewidth: "100 kHz"
  rin_dB_per_Hz: -140
  sweep_nonlinearity_a2: 1.0e14
```

## optics

| field | type | default | description |
| --- | --- | --- | --- |
| `splitting_ratio` | float in (0, 1) | 0.5 | main MZI coupler ratio |
| `circulator` | block | defaults | see below |
| `aux_mzi` | block | disabled | see below |

### optics.circulator

| field | type | default | description |
| --- | --- | --- | --- |
| `insertion_loss_dB` | float >= 0 | 0.7 | per-pass loss |
| `isolation_dB` | float >= 0 | 50.0 | port-to-port isolation |
| `return_loss_dB` | float >= 0 | 55.0 | back-reflection |

### optics.aux_mzi

Auxiliary k-clock MZI. Off by default. When enabled, requires a positive
`delay` (root validator). The aux signal is persisted in the
`Acquisition` and used in post-processing to undo sweep nonlinearity.

| field | type | default | description |
| --- | --- | --- | --- |
| `enabled` | bool | False | turn k-clock on |
| `delay` | time >= 0 | 0.0 | round-trip delay; typical 10-500 ns |

The delay is rounded to `n_tau = round(delay*sample_rate)` samples. It must
span at least two samples and leave at least two valid output samples. The
nominal auxiliary beat `gamma*n_tau/sample_rate` must be strictly below
`sample_rate/2`; the usable delay therefore depends on the sweep settings.
During simulation, the actual delayed frequency difference is checked too,
including polynomial sweep errors, ripple and laser frequency noise. A local
excursion reaching Nyquist raises an error before the auxiliary cosine is made.

```yaml
optics:
  splitting_ratio: 0.5
  circulator:
    insertion_loss_dB: 0.7
  aux_mzi:
    enabled: true
    delay: "100 ns"
```

## strain

| field | type | default | description |
| --- | --- | --- | --- |
| `segments` | list | `[]` | piecewise strain regions, see below |
| `photoelastic_coefficient` | float in [0, 1) | 0.22 | p_e in Froggatt-Moore |
| `transfer` | `"ideal"` \| `"cox"` | `"ideal"` | strain transfer model |
| `cox` | block \| None | None | required when `transfer="cox"` |

### strain.segments[]

| field | type | default | description |
| --- | --- | --- | --- |
| `start`, `end` | length | -- | segment bounds |
| `epsilon` | float | 0.0 | static strain offset |
| `motion` | union \| None | None | one of the motion kinds (next section) |

The motion adds on top of `epsilon`. With `motion=None` the segment is
purely static and benefits from the strain cache across sweeps.

### strain.cox

Required when `transfer="cox"`. Shear-lag coating coupling.

| field | type | description |
| --- | --- | --- |
| `beta` | float > 0 (1/m) | shear-lag parameter |

```yaml
strain:
  transfer: cox
  cox: {beta: 200}
  segments:
    - start: "4.5 m"
      end:   "5.5 m"
      epsilon: 0
      motion:
        kind: harmonic
        amplitude: 1.0e-6
        frequency: "120 Hz"
```

## temperature

Distributed temperature perturbation with strain-T cross-sensitivity
(Froggatt-Moore). Defaults are silica @ 1550 nm.

| field | type | default | description |
| --- | --- | --- | --- |
| `segments` | list | `[]` | piecewise temperature regions |
| `thermal_expansion` | 1/K >= 0 | 5.5e-7 | alpha_L |
| `thermo_optic` | 1/K >= 0 | 6.5e-6 | xi = (1/n) dn/dT |

### temperature.segments[]

Same motion union as strain. `delta_T` is the static offset in Kelvin
(signed). Motion modulates dT(t).

| field | type | default | description |
| --- | --- | --- | --- |
| `start`, `end` | length | -- | segment bounds |
| `delta_T` | float (K) | 0.0 | static offset |
| `motion` | union \| None | None | as for strain |

```yaml
temperature:
  segments:
    - start: "3 m"
      end:   "4 m"
      delta_T: 5.0
    - start: "7 m"
      end:   "8 m"
      delta_T: 0.0
      motion: {kind: thermal, amplitude: 2.0, tau: "2 s"}
```

## Motion kinds (strain + temperature)

Both `strain.segments[].motion` and `temperature.segments[].motion`
accept the same discriminated union, tagged by `kind`. For strain segments
the amplitudes are strain; for temperature segments they are Kelvin.

### kind: harmonic

| field | type | description |
| --- | --- | --- |
| `amplitude` | float >= 0 | sinusoidal amplitude |
| `frequency` | Hz > 0 | modulation frequency in lab time |
| `phase` | radians | default 0.0 |

Watch out for the sweep-rate Nyquist: `frequency >= 1/(2*sweep_duration)`
aliases and produces a UserWarning.

### kind: thermal

First-order relaxation `eps(t) = A * (1 - exp(-t/tau))`. `amplitude`
is signed (negative for cooling).

| field | type | description |
| --- | --- | --- |
| `amplitude` | float | asymptote (signed) |
| `tau` | time > 0 | time constant |

### kind: impulsive

Gaussian-in-time pulse `eps(t) = A * exp(-(t-t0)^2 / (2 w^2))`. Not
zero-mean: a residual tail persists after the pulse.

| field | type | description |
| --- | --- | --- |
| `amplitude` | float | peak (signed) |
| `center_time` | time >= 0 | peak time, lab frame |
| `width` | time > 0 | std dev (not FWHM) |

### kind: random

Broadband white-gaussian per-sweep noise. PSD flat up to the sweep-rate
Nyquist. Stateless: same `(seed, t)` -> same sample. Spectrally-shaped
variants tracked in #77.

| field | type | description |
| --- | --- | --- |
| `amplitude` | float >= 0 | RMS (sigma) |
| `seed` | int >= 0 | reproducibility seed (default 0) |

## detection

Photodiode + transimpedance chain: responsivity, analog bandwidth, the
noise terms (shot/thermal/dark), and optional saturation + nonlinearity.

| field | type | default | description |
| --- | --- | --- | --- |
| `responsivity` | A/W > 0 | 1.0 | photodiode responsivity |
| `bandwidth` | Hz > 0 | 1.0e8 | analog LPF cutoff |
| `filter_order` | int >= 1 | 4 | Butterworth order |
| `shot_noise` | bool | True | toggle shot noise |
| `thermal_nep` | W/sqrt(Hz) >= 0 | 1.0e-11 | thermal NEP |
| `dark_current` | A >= 0 | 1.0e-9 | photodiode dark current |
| `balanced` | bool | False | balanced detector mode |
| `saturation_current` | finite A > 0 \| None | None | total current limit per photodiode; None disables |
| `nonlinearity_coefficients` | list[float] | `[]` | a2, a3, ... polynomial coefficients |

`nonlinearity_coefficients` starts at order 2 (`a2`, `a3`, ...); the
linear term is implicit identity. Empty list = no polynomial distortion.
`saturation_current` clips each diode's polynomial response to `[0, I_sat]`,
including reference-arm DC, before balanced subtraction or single-ended
static-baseline removal. Shot noise uses the nonnegative current of each
diode after its polynomial response and any clipping. The limit applies to
the photodiode current, upstream of the TIA.
The reference DC uses the configured mean laser power, as detailed in the
[theory guide](theory.md#saturation).

```yaml
detection:
  responsivity: 0.9
  bandwidth: "100 MHz"
  balanced: true
  nonlinearity_coefficients: [10.0, 1.0e5]
```

## adc

Digitizer stage: quantization and sampling plus the non-ideal effects
(aperture jitter, DNL/INL, ENOB derating).

| field | type | default | description |
| --- | --- | --- | --- |
| `bits` | int, 1 to 32 | 16 | full bit depth; output is int16 up to 16 bits, int32 above |
| `enob` | float > 0 \| None | None | effective ENOB; must be <= `bits` |
| `sample_rate` | Hz > 0 | 2.0e8 | sampling rate |
| `voltage_range` | V > 0 | 2.0 | full-scale input range |
| `input_impedance` | ohm > 0 | 50.0 | termination |
| `jitter_rms` | time >= 0 | 0.0 | aperture jitter |
| `dnl_rms_lsb` | finite float >= 0 | 0.0 | target RMS of the random code-width error, in LSB; nonzero values require `bits <= 20` |
| `inl_peak_lsb` | finite float >= 0 | 0.0 | peak sinusoidal threshold displacement, in LSB |

DNL uses positive lognormal code widths, normalized to the full-scale range;
the realized RMS varies between finite chips. INL adds a sinusoidal displacement
to the thresholds. These are separate contributions: random DNL also creates
INL, and sinusoidal INL changes code widths. Thresholds are fixed by the seed
and shared across sweeps and channels. Configurations that make thresholds
cross or become numerically indistinguishable are rejected when the ADC is
created. See `theory.md` for the model and its limitations.

DNL uses a full threshold table, so it is supported up to 20 bits (about
8 MiB for the table, plus construction temporaries). Larger bit depths with
nonzero DNL are rejected during configuration validation, including by
`pyofdr validate`. For 21-32 bits, sinusoidal INL is evaluated on demand with
binary search and bounded temporary arrays. No random DNL approximation is
substituted. Ideal quantization, ENOB and jitter remain available up to 32 bits.

```yaml
adc:
  bits: 14
  enob: 12
  sample_rate: "250 MHz"
  voltage_range: "1 V"
  jitter_rms: "1 ps"
```

## output

Each sweep streams to a temporary file in the destination directory. Once all
sweeps finish and the HDF5 file closes successfully, it atomically replaces
`output.path`. A failed run leaves any previous output intact. When replacing
a file, allow disk space for both the previous and the new result.

`source.sweep_duration` and `adc.sample_rate` are preserved even when their
product is fractional. The sample count is its ceiling, except for roundoff
within four ULPs of an integer. The main beat and FBG phase use the actual
ADC times. Fractional counts use a chirp z-transform instead of the usual IFFT.

`acq.dz` remains the fiber construction spacing. Use `acq.fft_dz` for the
unpadded raw reflectogram and `acq.fft_sweep_range_hz` for its cross-correlation
frequency conversion. Both also appear in HDF5 `/derived` attributes and in
each sweep's attributes, together with `frequency_step_hz`. Each sweep records
`n_samples` and `dt` as well. See the sample-grid discussion in `theory.md`
before cropping, padding or k-clock resampling.

Applied strain and temperature are saved per sweep as
`/sweeps/0000/strain_field` and `/sweeps/0000/temperature_field`, and similarly
for subsequent sweeps. Both use `/fiber/z` as their spatial axis; temperature
values are changes in K, not absolute temperatures. The corresponding
`/fiber/` datasets contain only the first sweep's fields. Fields are omitted
when their perturbation is not configured.

| field | type | default | description |
| --- | --- | --- | --- |
| `path` | str \| None | None | HDF5 output path; None = no file output |

```yaml
output:
  path: "data/run.h5"
```

## Cross-block validation

A root-level validator enforces that the maximum beat frequency stays
below the ADC Nyquist:

```
f_beat_max = 2 * n_core * gamma * length / c     (gamma = sweep_hz / T)
f_beat_max < adc.sample_rate / 2
```

If this fails the config is rejected with a message naming both
quantities in MHz. Knobs that move `f_beat_max`: `fiber.length`,
`fiber.n_core`, `source.sweep_range`, `source.sweep_duration`,
`source.center_wavelength`.

For a nonlinear main beat, a runtime check also bounds the unwrapped phase
advance between ADC samples over the full spatial grid. Polynomial sweep
errors, ripple, dispersion and the realized slow frequency noise are included.
Reaching or exceeding Nyquist raises an error before generating the beat,
even if the auxiliary MZI is disabled. Increase `adc.sample_rate`, shorten
the fiber or reduce the excursions to bring the bound below Nyquist.
Configuration validation alone only checks the nominal linear chirp;
see `theory.md` for the sampled-phase guard and its limits.

For an enabled auxiliary MZI, validation also checks the nominal beat using
its rounded sample delay, independently of the fiber length. This catches
an undersampled k-clock even when the main beat is below Nyquist. The runtime
auxiliary check covers the realized sweep and noise; configuration validation
alone does not bound those excursions.
