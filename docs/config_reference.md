# Config reference

Every PyOFDR run starts from a YAML config validated against the pydantic
models in `core/config_models.py`. This page is the schema reference; for
*why* a field exists see `theory.md`, for the run pipeline see
`architecture.md`.

## Conventions

- **SI everywhere** inside the validated dict. The YAML accepts unit strings
  (`"40 nm"`, `"10 ms"`, `"17 ps/(nm km)"`) via pint, or bare numbers already
  in SI.
- **`extra = "forbid"`** on every block: typos in field names raise on load.
- All defaults below are taken straight from the model -- omit a field to
  get its default.

A minimal valid config is the empty dict `{}`: every block has defaults that
produce a 10 m fiber, 1550 nm source, single sweep, no perturbations.

## simulation

Global run controls: master seed, compute backend, campaign size.

| field | type | default | description |
| --- | --- | --- | --- |
| `seed` | int | 42 | master RNG seed; per-stage streams are derived from this |
| `backend` | `"numpy"` \| `"cupy"` \| `"jax"` | `"numpy"` | only `numpy` is wired today |
| `n_sweeps` | int >= 1 | 1 | number of sweeps in the campaign |

```yaml
simulation:
  seed: 1234
  n_sweeps: 20
```

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

Macrobending. The default `A_dB_per_turn`/`R_c` are SMF-28 ballparks
(see #36); override for other fibers.

| field | type | default | description |
| --- | --- | --- | --- |
| `start`, `end` | length | -- | segment bounds |
| `radius` | length > 0 | -- | bend radius |
| `turns` | float > 0 | -- | number of turns over the segment |
| `A_dB_per_turn` | float > 0 | 100.0 | empirical prefactor |
| `R_c` | length > 0 | 5e-3 | critical radius |

### fiber.reflectors[]

| field | type | default | description |
| --- | --- | --- | --- |
| `z` | length | -- | position along the fiber |
| `R` | float in [0, 1] | -- | power reflectivity |
| `loss_dB` | float >= 0 | 0.0 | one-way insertion loss at the reflector |

### fiber.fbg_arrays[]

Weak FBGs (Born approximation; safe for `peak_reflectivity <= ~0.5`).
Strong gratings via CMT / transfer matrix tracked in #78.

| field | type | description |
| --- | --- | --- |
| `z` | length | grating start position |
| `bragg_wavelength` | length | nominal Bragg wavelength |
| `length` | length > 0 | physical grating length |
| `peak_reflectivity` | float in [0, 1] | R_max on-Bragg |

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
enumerates paths with 2N+1 reflection events; N=2 is the dominant case
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
| `saturation_current` | A \| None | None | clamp; None disables |
| `nonlinearity_coefficients` | list[float] | `[]` | a2, a3, ... polynomial coefficients |

`nonlinearity_coefficients` starts at order 2 (`a2`, `a3`, ...); the
linear term is implicit identity. Empty list = ideal photodiode.

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
| `bits` | int >= 1 | 16 | full bit depth |
| `enob` | float > 0 \| None | None | effective ENOB; must be <= `bits` |
| `sample_rate` | Hz > 0 | 2.0e8 | sampling rate |
| `voltage_range` | V > 0 | 2.0 | full-scale input range |
| `input_impedance` | ohm > 0 | 50.0 | termination |
| `jitter_rms` | time >= 0 | 0.0 | aperture jitter |
| `dnl_rms_lsb` | float >= 0 | 0.0 | per-code DNL rms |
| `inl_peak_lsb` | float >= 0 | 0.0 | INL peak (sinusoidal model) |

```yaml
adc:
  bits: 14
  enob: 12
  sample_rate: "250 MHz"
  voltage_range: "1 V"
  jitter_rms: "1 ps"
```

## output

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
