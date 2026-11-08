# Theory

Physics behind each pipeline stage. Package layout in `architecture.md`, YAML
config in `config_reference.md`.

## Signal chain

```
SweptLaser -> FiberGenerator -> StrainPerturbation -> TemperaturePerturbation
           -> MachZehnder -> Detector -> AntiAliasFilter -> ADC
```

If the sweep is linear, a scatterer at distance `z` beats at
`f_beat(z) = 2 n gamma z / c`, linear in z. `n` is the group index,
`gamma = delta_nu / T_sweep`, factor 2 from the round trip. The time-domain
beat is then the inverse transform of the attenuation-weighted profile plus
ghosts:

    s(t) = IFFT[ profile(z) * attenuation(z) + ghosts(z) ] * n_samples

(see `optics/mach_zehnder.py`). O(N log N) instead of O(N^2) of a
per-scatterer sum, exact as long as the chirp stays linear. Mode hops and
sweep nonlinearity break this, hence the aux MZI k-clock or the time-warp
(optics section).

Two consequences for the grid: `dz = c / (2 n delta_nu)` (in
`core/config.py`, `compute_derived`), and `f_beat_max = 2 n gamma L / c`,
with `f_beat_max >= fs/2` rejected at validation (`core/config_models.py`,
`_check_nyquist`).

First OFDR: Eickhoff & Ulrich, Appl. Phys. Lett. 39, 693 (1981); see also
Soller et al., Opt. Express 13, 666 (2005).

## Source

`source/swept_laser.py`, one sweep at a time.

### Chirp

Instantaneous frequency is a linear ramp plus optional nonlinearity:

    nu(t) = nu_start + gamma t + a2 t^2 + a3 t^3 + A_r sin(2 pi t / T_r)

with `gamma = delta_nu / T_sweep` and `delta_nu = (c / lambda^2) delta_lambda`
(`utils/units.py`, `wavelength_range_to_freq_range`). Zero coefficients give
the ideal chirp. Phase is the rectangle-rule integral of nu, and the field
carries it on the power envelope:

    phi(t) = 2 pi cumsum(nu) dt
    E(t)   = sqrt(P(t)) exp(j phi(t))

### Phase noise

Three terms:

    S_nu(f) = h_0 + h_-1 / f + h_-2 / f^2

White FM gives a Gaussian phase step, zero mean, std `sqrt(2 pi Delta_nu dt)`
with `Delta_nu` the Lorentzian linewidth. Flicker (1/f) and random-walk (1/f²)
take RMS frequency deviations in Hz over the sampled band; both use the
FFT-shaped noise in `utils/colorednoise.py` and need at least two samples per
sweep. White FM is fine with one.

All three go into `nu_inst`, which sets the optical phase and drives the aux
MZI clock. In the main MZI the slow part changes the beat through the
time-warp, while the linewidth reduces the beat amplitude with distance via
the coherence roll-off.

Timmer & Koenig, Astron. Astrophys. 300, 707 (1995); Di Domenico et al.,
Appl. Opt. 49, 4801 (2010).

### RIN

White multiplicative noise on the power:

    P(t) = P0 (1 + m(t)),   m ~ N(0, sqrt(RIN_lin * BW))

with `RIN_lin = 10^(RIN_dB_per_Hz / 10)` and `BW = fs/2`.

### Power envelope

Tunable lasers droop toward the sweep edges. Parabola:

    P(u) = P0 (1 - 4 e u^2),   u = t/T_sweep - 0.5,   e = 1 - 10^(-edge_dB/10)

`edge_dB = 0` flat. Both the envelope and the RIN reach the detector because
the beat carries `P(t)` pointwise, not an average (optics section).

### Sweep nonlinearity

The `a2`, `a3`, `A_r`, `T_r` terms are the deviation from the linear ramp.
The time-warp models their effect on the beat; aux-MZI k-clock resampling can
compensate in post-processing.

Glombitza & Brinkmeyer, J. Lightwave Technol. 11, 1377 (1993).

## Fiber

`fiber/profile.py` draws the Rayleigh profile; the rest hangs off it as
submodules.

### Rayleigh backscatter

Each `dz` bin holds a circular-gaussian complex phasor:

    r(z) = sigma/sqrt(2) * (re + j im),   re, im ~ N(0,1)

with `sigma = sqrt(R_lin * cell_length)`, `R_lin = 10^(R_dB/10)` the
backscattered power per metre. Full cells use `cell_length = dz`, the last
cell only the remaining length. Magnitude squared is exponential, the usual
OFDR speckle. Per-segment `R(z)` is allowed, so sigma varies along z; each
core gets its own draw.

### Attenuation

Beer-Lambert. dB/km to Np/m:

    alpha_Np_m = alpha_dB_km * ln(10) / 10000

Round-trip amplitude envelope `env(z) = exp(-alpha_Np_m z)`
(`fiber/attenuation.py`). Worth stating the convention: the envelope
multiplies a field but represents round-trip power loss, because light walks
the fiber twice. With heterogeneous segments we build an `alpha(z)` vector
and integrate cumulatively.

### Macrobending

Three models in `fiber/bends.py`. Default `exponential` uses
`loss_per_turn = A exp(-R/R_c)` in dB/turn, empirical `A=100 dB/turn`,
`R_c=5 mm`. `tabulated` interpolates log(loss) linearly against radius,
clamps outside the table with a warning. Neither infers wavelength
dependence.

The `marcuse` model uses the LP01 mode of a weakly guiding step-index fibre
([Marcuse, 1976](https://doi.org/10.1364/JOSA.66.000216)).
The code computes the power decay coefficient `q` in m⁻¹ from the bend-loss
formula in [Martins et al. (2009), Eq. (4)](https://www.researchgate.net/publication/231179029_Modeling_of_Bend_Losses_in_Single-Mode_Optical_Fibers).
It uses the effective radius `R_eff = radius_correction * R` in that formula.
The default correction is 1.28 for silica. Set it to 1 for the uncorrected curve.
The arc length `L` uses the physical radius `R`.

    L = 2*pi*R*turns
    P_out / P_in = exp(-q*L)
    loss_dB = (10/ln(10))*q*L

The mode solver brackets the root in `log(W)`. It requires `0 < V < 2.4048255577`
and `delta <= 0.01`. Supply the core and cladding phase indices separately
from the OFDR group index.

For all three models the one-way bend loss is spread over `[start, end)`
and integrated cumulatively:

    covered = max(0, min(z, end) - max(0, start))
    bend_loss_to_z = total_loss_dB * covered / (end - start)

Zero at the start, full loss at or beyond the end, even if the whole bend
sits between two grid points. Overlapping bends add in dB. Same cumulative
loss for direct returns and ghosts; direct returns see the round-trip field
factor `10^(-loss_to_z/10)`, ghosts count the interval on every passage.
`tests/test_bend_grid.py` covers shifted, sub-cell, overlapping and partial
ghost cases. For a real commercial fiber you still need its optical
parameters.

### Discrete reflectors and connectors

A point reflector adds `r = sqrt(R) exp(j phi)` at its nearest bin
(`fiber/reflectors.py`). Per-core seeded phases, drawn once and reused
across sweeps; separate stream from the Rayleigh RNG. Reflectors are sorted
by position before drawing phases; two entries in the same bin are rejected.

Real field transmission `t = sqrt(1-R)`, right reflection `-conj(r)`, so the
lossless interface is the reciprocal unitary
`[[r, t], [t, -conj(r)]]`. This fixes the relative phases of direct returns
and ghosts; separate bounces are not independent. Two-port constraints in
Karlsson's electrodynamics notes, section 3.7.

`loss_dB` is an extra matched loss on the input side. With
`q = 10^(-loss_dB/20)`, left reflection `q^2 r`, right `-conj(r)`,
transmission `q t` either way. Direct reflection sees its own insertion loss
twice; `loss_dB` is additional to the transmitted-power loss `R`. Envelope:
`1-R` beyond each reflector, `10^(-loss_dB/10)` from its bin onwards. The
reflector's own return does not cross its interface. Rayleigh and weak-FBG
returns at the same bin use the upstream-side convention. `R=1` kills all
later direct returns.

### Multiple-scattering ghosts

Light can bounce between discrete reflectors and come back at a later delay
(`fiber/multiple_scattering.py`). Order-N path, `2N-1` reflections,
apparent depth

    z_app = z_i0 - z_i1 + z_i2 - ... + z_i(2N-2)

Every odd valley strictly below both neighbours; positions rounded to bins
before summing. Each path multiplies directional reflection coefficients and
the transmission of every intervening interface.

Distributed loss is integrated from the input up to each bin, with the value
at zero being zero and the loss at bin k applying to `[z[k], z[k+1])`. For
each leg, the difference of cumulative losses gives the one-way loss;
summing legs and converting with `10^(-path_loss_dB/20)` gives the field
attenuation. Uniform loss agrees with `2*z_app`, nonuniform depends on the
visited sections. First ghost of two lossless interfaces at `a < b`:
`-(1-R_a) conj(r_a) r_b^2` at `2*b-a`; `(b,a,c)` and `(c,a,b)` share a delay
and add coherently.

`Acquisition.ghost_profile` stores the attenuated returns separately; optics
adds them to `fiber_profile * attenuation_envelope` before coherence
roll-off and the beat transform. Dividing by the direct envelope would fail
behind an opaque reflector or an underflowed loss, even when a shorter path
still produces a ghost there. Finite-bounce, not a high-finesse solver:
expansion stops at `max_order` (2 or 3) and out-of-range ghosts are dropped.
Perturbations use the phase at the apparent bin, so a ghost path with
nonuniform strain is approximate. `tests/test_reflector_paths.py` covers
two-mirror amplitudes, repeated crossings, localized loss, opaque
interfaces, sweep reproducibility, and a frequency-domain recursion for
three weak mirrors to `1e-10` field amplitude.

### Strain

A scatterer at z picks up an extra round-trip phase from the integrated
strain (`fiber/strain.py`):

    d_phi(z) = 2 k0 n (1 - p_e) integral_0^z eps(z') dz'

`k0 = 2 pi / lambda`, `p_e ~ 0.22` at 1550 nm. The `(1 - p_e)` factor folds
in the index change, so the geometric stretch isn't double counted. Profile
multiplied by `exp(j phase)`. For strain, temperature and index changes the
added phase at the fiber input is zero. Each step from `z[k]` to `z[k+1]`
accumulates the phase from the field at `z[k]` over one `dz`.

Froggatt & Moore, Appl. Opt. 37, 1735 (1998); Bertholds & Dandliker,
J. Lightwave Technol. 6, 17 (1988).

### Strain transfer host to fiber

Two models in `strain_transfer/`:

- ideal: `eps_fiber = eps_host` inside the segment, zero outside.
- Cox shear-lag: `eps_fiber(s) = eps_host (1 - cosh(beta s)/cosh(beta L_h))`,
  `s` from segment centre, `L_h` half-length, `beta` a shape constant [1/m]
  from config. It lumps bonding shear modulus, geometry and Young's modulus;
  not derived from elastic constants here.

Cox, Br. J. Appl. Phys. 3, 72 (1952).

### Dynamic strain motions

`strain_transfer/motions.py`, sampled at lab time `t = sweep_index * T_sweep`
(one value per sweep), collapsed to a single `eps(t)` per segment and added
to the static offset:

- harmonic: `eps(t) = A sin(2 pi f t + phi)`, warns if
  `f >= 1/(2 T_sweep)`;
- thermal relaxation: `eps(t) = A (1 - exp(-t/tau))`;
- impulsive: `eps(t) = A exp(-(t-t0)^2 / 2 sigma^2)`, not zero-mean, leaves a
  residual;
- random vibration: `eps(t) ~ N(0, A^2)`, stateless via a SeedSequence on
  `(seed, t)`, reproducible without a buffer.

### Index perturbation (small-signal dn)

Deterministic segments rotate the round-trip phase without moving the bins
(`fiber/profile.py`):

    d_phi(z) = 2 k0 integral_0^z delta_n dz'

Same as strain without `(1 - p_e)`. On top, a stochastic slice: an
Ornstein-Uhlenbeck process discretised as AR(1),

    dn[k] = a dn[k-1] + sigma sqrt(1 - a^2) eta[k],   a = exp(-dz / L_corr)

started from the stationary distribution (`dn[0] = sigma eta[0]`), so the
variance is `sigma^2` and the lag-1 autocorrelation is `exp(-dz/L_corr)`.
The full OPL-based n(z) remap (splices, hollow-core, tapers, absolute
metrology) is the rest of #33, still V2.

### Distributed temperature

`fiber/temperature.py`, segments add

    d_phi(z) = 2 k0 n (alpha_L + xi) integral_0^z dT dz'

Silica defaults `alpha_L = 5.5e-7 /K`, `xi = (1/n) dn/dT = 6.5e-6 /K`.
Together with the strain phase this closes the Froggatt-Moore relation

    df/f = -(1 - p_e) eps - (alpha_L + xi) dT

so pure temperature looks like an apparent strain
`eps_app = (alpha_L + xi)/(1 - p_e) dT`. Segments can carry a `motion`
(above) read as `dT(t)`. The 1D thermal-diffusion PDE along z is V2 (#76).

### MCF core-to-core crosstalk

Scalar phase-scrambled model (`fiber/crosstalk.py`):

    c(z) = sqrt(XT_lin * z/1000)
    profile_target(z) += c(z) exp(j phi_rand(z)) profile_source(z)

`XT_lin = 10^(XT_dB_per_km/10)`, independent U(0, 2pi) phase per bin per
directed pair. Topologies `hex7` and `linear`. Weak-coupling limit: the power
leaking out isn't subtracted from the source cores. Full coupled-mode is
V2 (#69). Hayashi et al., Opt. Express 19, 16576 (2011).

### FBG arrays (Born approximation)

A weak Bragg grating is a reflector with a sinc spectral response
(`fiber/fbg.py`):

    r(t) = sqrt(R_max) sinc(2 n L_g (nu(t) - nu_B) / C)

`nu_B = C/lambda_B`, `L_g` the physical length, `sinc(x) = sin(pi x)/(pi x)`.
First null at `Delta_nu = C/(2 n L_g)`. Added separately to the beat with
carrier `cos(2*pi*k*s(t)/T)`, `k` the grating's rounded spatial bin, `s(t)`
the same warped time used for Rayleigh and ghosts. The envelope uses
`nu_inst`; the propagation phase includes nonlinearity, ripple, GVD and slow
FM. GVD changes the phase coordinate, not the frequency argument of the
envelope. White FM stays out of the warp, consistently with the main beat;
linewidth still gives the coherence roll-off and frequency noise can
modulate the envelope.

Linear chirp: `s(t)=t`; in z-space the FBG is a rect of `L_g/dz` bins, not a
delta. Born is fine for `R_max <~ 0.5`; coupled-mode / transfer-matrix for
strong gratings is V2 (#78).

Configured `bragg_wavelength` is the unperturbed resonance. At each sweep:

    lambda_B_eff = lambda_B * (1 + (1 - p_e) eps + (alpha_L + xi) dT)
    nu_B = C / lambda_B_eff

Coefficients from `strain.photoelastic_coefficient`,
`temperature.thermal_expansion`, `temperature.thermo_optic`. Strain is taken
after host-to-fiber transfer, including Cox when selected. Missing fields
mean zero perturbation. Temperature expansion is already in the thermal
term; don't add the free-fiber expansion to the strain field again.

Uniform field over each grating, sampled at the same rounded z-bin used for
its beat phase and attenuation. Strain and temperature shift the resonance
without changing grating length, bandwidth, peak reflectivity, or adding
their own propagation phase. The configured wavelength stays unchanged
between sweeps; nonpositive or nonfinite effective wavelengths are rejected.
`tests/test_fbg_nonlinearity.py` covers quadratic/cubic sweep errors, ripple,
both GVD signs, slow FM, fractional sample counts and repeated sweeps.

Erdogan, J. Lightwave Technol. 15, 1277 (1997).

## Optics

`optics/mach_zehnder.py` turns the profile into a photocurrent, plus
`optics/components.py` and the aux MZI.

### Beat amplitude and scaling

After the Fourier sum the main photocurrent is

    photocurrent(t) = prefactor * P(t) * Re[beat]
    prefactor = 2 sqrt(eta (1 - eta)) IL_circ^2

`P(t) = |E_source|^2` is instantaneous, so the envelope and RIN reach the
detector as multiplicative modulation on the AC beat. The slow-envelope
approximation `sqrt(P(t) P(t-tau)) ~ P(t)` is what we use throughout.

### Sweep duration and sample grids

For requested duration `T` and ADC rate `fs`, samples at `t[j] = j/fs`,
count `N = ceil(T*fs)`, with products within four ULPs of a positive integer
treated as that integer. This prevents roundoff from adding a sample;
genuine fractional counts are kept. Neither `T` nor `fs` is adjusted.

With span `B`, the fiber grid is `dz = c/(2*n*B)`. Bin `k` contributes
`h[k] * exp(2*pi*i*k*t/T)`. An ordinary IFFT evaluates this only when
`T*fs = N`; otherwise the main MZI uses ZoomFFT with the positive-exponent
convention on the actual sample times. Coefficients are reused across
sweeps. The FBG carrier uses `t/T` for a linear chirp and shares `s(t)/T`
with the warped fiber sum when nonlinearity, dispersion or slow FM is on.

The unpadded FFT grid differs from the fiber grid for fractional counts:

    frequency_step_hz = (B/T) / fs
    fft_sweep_range_hz = N * frequency_step_hz
    fft_dz = c / (2*n*fft_sweep_range_hz) = dz * T*fs/N

`Acquisition` and HDF5 output retain these. DFT span uses `N` increments,
the frequency difference between first and last samples uses `N-1`.
`reflectogram_from_acq` applies `fft_dz` automatically. For array calls:

```python
H, z = fft_reflectogram(acq.digital_main[0], acq.fft_dz)
# For windowed_xcorr_strain on these unpadded reflectograms:
# dz=acq.fft_dz, sweep_range_hz=acq.fft_sweep_range_hz
```

Zero-padding refines further; use the returned `z` spacing and scale the DFT
span by `n_pad/N` when demodulating. After k-clock resampling or cropping,
calibrate from the new frequency step and record length. Linear sweep
cropped to `M` samples: `fft_dz * N/M`.

The nonlinear time warp evaluates the fiber's Fourier sum at warped times,
using the physical period `T` even when `T*fs` is fractional, on a separate
oversampled grid.

### Circulator

The signal passes through the circulator twice (`optics/components.py`).
For a single-pass loss of `L_dB`, the field factor is `10^(-L_dB/20)` per
pass and `10^(-L_dB/10)` for the round trip. Squaring the round-trip field
factor gives the power ratio `10^(-2*L_dB/10)`. A 1 dB loss per pass gives
a 2 dB power loss over the round trip.
Isolation (port1->port3 leakage) lands at zero delay as a DC offset riding
`P(t)`; return loss is a reflector in the z=0 bin. Isolation is a scalar
number here, no polarization dependence.

### Laser coherence roll-off

Finite Lorentzian linewidth `Delta_nu` makes fringe visibility decay with
round-trip delay:

    V(z) = exp(-pi Delta_nu tau),   tau = 2 n z / C

Applied as a z-dependent multiplier on the profile before the IFFT (no
explicit reference arm, so the decay is folded in).

Coloured (flicker / random-walk) FM enters stochastically through the
time-warp: for noise slow compared to tau,
`phi(t) - phi(t-tau) ~ 2 pi tau nu_noise(t)`, which is the ripple mechanism,
so each sweep gets a real phase-noise pedestal. White FM is kept out of the
warp (it decorrelates sample to sample, and the small-tau approximation would
overshoot its pedestal by ~tau/dt), so the Lorentzian part appears only as
the ensemble-average roll-off above.

Coupland & Pickering, J. Opt. Soc. Am. A 9, 257 (1992).

### Sweep-nonlinearity time-warp

For small round-trip delay `tau`, the frequency deviation `delta_nu(t)` adds
beat phase `2*pi*tau*delta_nu(t)`. With nominal chirp rate `gamma` this is
equivalent to evaluating the ideal beat at

    s(t) = t + delta_nu(t)/gamma
    beat(t) = sum_k h[k] exp(2*pi*i*k*s(t)/T)

`h` is the fiber profile after attenuation, ghosts and coherence roll-off.
The deviation includes polynomial/ripple, slow colored FM and the equivalent
GVD term below; white FM keeps its separate coherence treatment. This is the
forward model; aux-MZI k-clock resampling is a later analysis step that tries
to compensate.

Linear interpolation on the ADC grid is a bad idea here: a half-sample shift
attenuates a tone at `0.4*fs` by 10.20 dB. The implementation centres the
known Fourier band around zero, evaluates by IFFT on at least `16*n_z`
points, and uses a periodic quintic spline before restoring the carrier. The
centred band has at least 32 samples per fastest cycle. Cores processed one
at a time, internal grid scales with the fiber size, not the ADC count.

The Fourier sum has period `T` in `s`, so evaluation beyond the nominal
sweep continues the sum rather than clamping to the first or last ADC
sample. Excursions beyond 5% of `T` still warn. Continuation follows the
discrete-fiber model; it is not a model of laser reset or flyback.

`tests/test_time_warp_accuracy.py` compares tones and complex multicore
profiles with direct Fourier sums, including fractional sample counts, sweep
boundaries, polynomial distortion, ripple, GVD and slow FM, to `2e-7` of
unit field amplitude. Before evaluating a warped beat, the main MZI checks
the unwrapped phase increments on the actual ADC clock. With
`k_max = n_z-1`,

    f_max = k_max/T * max(abs(1 + diff(delta_nu)/(gamma*dt)))

Values at or above `1/(2*dt)` raise `ValueError`, before reducing `s/T`
modulo one. Both signs and local excursions count. The bound covers the full
spatial grid, ghosts and all cores, even when the last bins are zero. This
guards the sampled phase increments; sidebands and excursions between source
samples are not bounded. `test_main_nyquist.py` covers both sides of
Nyquist, short fast ripple, dispersion, local slow-FM excursions and the
unchanged direct Fourier sum below the limit.

### Chromatic dispersion (GVD)

`beta2 = -lambda^2 D/(2 pi c)`. It adds
`4 pi^2 beta2 gamma^2 z (t - T/2)^2` to the beat phase, which factors as
`z K(t)`, so it rides the time-warp engine with

    delta_nu = pi beta2 gamma^2 (c/n) (t - T/2)^2

Warp is z-independent; the z-dependent broadening falls out of the FFT (a
chirp in time spreads in frequency). Only `D` / `beta2` is modelled.

### Aux MZI and k-clock

Aux MZI with known delay tau, fed by the same laser (`optics/aux_mzi.py`):

    I_aux(t) = cos(phi(t) - phi(t - tau)) = cos(2 pi integral_{t-tau}^t nu du)

Linear chirp: clean sine at `gamma tau`. When the sweep wobbles, I_aux
wobbles with it, so the unwrapped Hilbert phase is monotone in nu(t).
`analysis.demodulation.kclock_resample` uses it to resample the main beat
onto a uniform-nu grid before the FFT. First `round(tau/dt)` samples invalid,
non-monotone phase samples dropped before `np.interp` (#64).
Parasitic reflections can distort the clock in a real instrument. The simulated
auxiliary MZI has a single delay and no parasitic paths.

Integer lag `n_tau = round(tau/dt)`, effective delay `n_tau*dt`. Validation
requires `gamma*n_tau*dt < fs/2`. At runtime, before forming the cosine,
the code checks `max(abs(nu_inst[j] - nu_inst[j-n_tau])) < fs/2` for
`j >= n_tau`; this is the phase increment divided by `2*pi*dt` for the
rectangle integration. Checking the unwrapped frequency difference avoids
mistaking an already aliased tone for a valid low-frequency clock. Both
signs and local excursions are checked, including the realized laser
frequency noise. The delay must leave at least two valid samples.

Glombitza & Brinkmeyer, J. Lightwave Technol. 11, 1377 (1993).

## Detection
`detection/detector.py` and `detection/filter.py`.

### Photodetection
Current is responsivity times optical power, `I(t) = R P_opt(t)`, with R in A/W (typ. 1.0 for InGaAs at 1550 nm).

### Saturation
`saturation_current` is a positive, finite limit on the total current of each
photodiode, including the reference-arm DC. It is applied before balanced
subtraction or removal of the static operating-point current. With
`F(x) = x + sum(a_k*x^k)` for the optional polynomial and
`G(x) = clip(F(x), 0, I_sat)`, the noiseless outputs are

    single-ended: G(I_dc_pd + I_beat) - G(I_dc_pd)
    balanced:     G(I_dc_pd + I_beat/2) - G(I_dc_pd - I_beat/2)

Thus a small beat can be lost when the DC alone saturates both diodes. The
polynomial is applied before clipping, so it cannot exceed the per-diode
limit. Samples without clipping retain the existing small-signal calculation
to avoid cancellation when subtracting large DC terms. Static-baseline removal
is not a model of an AC-coupling filter's frequency response.

Shot, thermal and dark noise are added afterwards. The shot-noise variance
is `2*e*I_collected*B` per diode. The model takes the nonnegative part of
the total current after the polynomial and any clipping, before removing
the baseline or subtracting the two diode signals. Independent diode noise
contributions add in variance, even if their mean difference is zero. Thermal
and dark noise retain their existing models and are not clipped by this limit.

This is a static hard-clipping model with a Gaussian mean-current estimate for
shot noise, not a prediction of noise or recovery dynamics of a saturated
device. The DC estimate remains `R*eta*source.power/2`; reference-arm RIN and
envelope variations are not propagated into that operating point. A TIA
output-voltage limit is a separate effect. `tests/test_detector_saturation.py`
checks one/both saturated diodes, polynomial composition, noise variance,
configuration bounds and end-to-end output.

### Electronic noise
Independent gaussians:

- reference current per photodiode: `I_dc_pd = R eta P_laser / 2`, including the 50/50 recombiner,
- single-ended shot: use the diode response at `I_dc_pd + I_beat`,
- balanced shot: add the independent variances from the diode responses at
  `I_dc_pd + I_beat/2` and `I_dc_pd - I_beat/2`,
- thermal / NEP: `sigma = R NEP sqrt(B)`, NEP in W/sqrt(Hz),
- dark current: variance `2 e I_dark B` per photodiode, doubled in balanced mode.

`B = fs/2` (sampling Nyquist band) at noise-generation time;
the anti-alias filter then trims the useful band.

### Balanced detection
Two photodiodes on complementary arms:
    I_a = I_dc_pd + I_beat/2 + noise_a
    I_b = I_dc_pd - I_beat/2 + noise_b
    I_out = I_a - I_b = I_beat + noise_a - noise_b
DC cancels and the independent shot and dark variances add. One TIA after
the subtraction node supplies the thermal contribution; it is not doubled.
The current implementation feeds the same stored MZI beat amplitude to both
receiver modes. Toggling `balanced` therefore changes the noise and nonlinear
response, but does not itself implement a physical single-port/full-difference
signal-amplitude conversion or guarantee a 3 dB SNR improvement.

Source RIN multiplies that beat in both modes. The reference-arm DC intensity
fluctuations and finite common-mode rejection are not propagated. Consequently,
`I_dc_pd * sqrt(RIN * B)` is not the simulated RIN floor. In the linear, weak-RIN
regime it is `I_beat_rms * sqrt(RIN * B)`, including any optical leakage in
`I_beat_rms`. Use RMS before filtering, without subtracting the mean.

### Anti-alias filter
Butterworth low-pass (configurable order, typ. 4), implemented as second-order sections for stability, cutoff clipped to `0.99 fs/2`
(`detection/filter.py`).
The settled output variance of white noise with one-sided PSD `S` is
`S * B_noise`, where `B_noise = integral_0^(fs/2) |H(f)|^2 df`. This equivalent
noise bandwidth depends on order and sample rate as well as cutoff; it is not
generally equal to the configured cutoff. The budget integrates the digital
Butterworth response with the same cutoff clamp. Filter startup transients
are excluded from its estimates.

### Photodiode nonlinearity
A pre-TIA polynomial of order >= 2,
    I_out = I_in + sum_{k>=2} a_k I_in^k
In balanced mode it acts per arm (`I_A = I_dc_pd + I_beat/2`,
`I_B = I_dc_pd - I_beat/2`), then the outputs are differenced. In single-ended
mode the polynomial is evaluated at `I_dc_pd + I_beat` and its static value
at `I_dc_pd` is subtracted. Both modes retain the DC/beat mixing terms.
When a saturation limit is also configured, each polynomial output (including
the static single-ended baseline) is clipped as described above.

## ADC
`digitizer/adc.py`.

### Quantization
Uniform floor quantizer, `2^bits` levels, hard clip at `+/- V_range/2`, theoretical noise standard deviation
    sigma_q = V_range / (2^bits sqrt(12))

### ENOB
The effective number of bits lumps jitter + INL + analog noise into one number, injected as an extra gaussian before quantization:
    sigma_total = V_range / (2^enob sqrt(12))
    sigma_extra = sqrt(sigma_total^2 - sigma_q^2)

### Aperture jitter
A timing error `dt_err ~ N(0, sigma_j)` becomes a voltage error to first order, `dV = (dV/dt) dt_err`, with the slope from finite differences.
For a single sample the slope is unknown, so the ADC skips the jitter correction.

### DNL / INL
Nonlinearities act on the voltage thresholds before a code is assigned. With
`N = 2^bits` and `LSB = V_range/N`, the ideal boundaries are
`T[j] = (j-N/2)*LSB`, including the two full-scale endpoints. A sample exactly
on an internal threshold takes the upper code, as in the ideal floor quantizer.

For DNL, draw `N` positive lognormal widths with log standard deviation
`sqrt(log(1+dnl_rms_lsb^2))`, then normalize their sum to `N`. The cumulative
widths replace the ideal boundary positions `j`. Before finite-sample
normalization, the width distribution has coefficient of variation
`dnl_rms_lsb`; the measured DNL RMS of a particular chip has sampling scatter.
Positive widths avoid the negative code widths possible with Gaussian DNL.
This lognormal choice is a model assumption, not a measured device distribution.

INL then adds `inl_peak_lsb*sin(2*pi*j/N)` to these boundary positions, in LSB.
The endpoints stay fixed, so neither contribution adds a net full-scale gain
error. With both enabled, the random widths also contribute to INL, and the
sinusoidal displacement contributes to DNL: the two parameters describe
components, not bounds on the total error. Combinations that cross thresholds
or produce numerically indistinguishable boundaries raise `ValueError`.
Missing codes are not modeled.

The threshold realization is fixed by the simulation seed and shared across
sweeps and channels. ENOB noise and aperture jitter are applied before threshold
comparison. With both nonlinearities zero, the original ideal quantizer is used.
A tiny threshold displacement only changes samples near that threshold; it
does not shift a whole code bin by one LSB. Tests in `test_adc_thresholds.py`
recover transitions by bisection and verify code widths with a uniform ramp.

Threshold tables are limited to 20 bits: a float64 table then takes about
8 MiB, rather than 32 GiB at 32 bits. Random DNL above 20 bits is rejected
at configuration validation; its independent code widths are not replaced
with a coarser or periodic approximation.

For 21-32 bits with sinusoidal INL alone, integer binary search evaluates
the same thresholds directly. Each sample takes `bits` comparisons, processed
in blocks of at most 65536 samples, so temporary search storage does not grow
with the number of ADC codes. Input and output storage still scales with the
acquisition size. This trades computation time for bounded memory.
The minimum sinusoidal code width is `1-A*sin(2*pi/N)` LSB. The analytic path
requires this to exceed `8*ulp(N)` to leave a floating-point rounding margin;
otherwise configuration validation rejects the thresholds as crossed or too
close to resolve. The tabulated path checks its realized boundaries directly.
`test_adc_memory.py` checks agreement with an explicit 21-bit table, 32-bit
transition ties, reproducibility across blocks, and bounded memory use.

Ref: [Analog Devices MT-010](https://www.analog.com/media/en/training-seminars/tutorials/mt-010.pdf),
for the definitions of transition-level INL and code-width DNL.

## Analysis
Post-processing under `analysis/`; reads the simulated output back.

### Budget
`analysis/budget.py` estimates optical power, receiver and ADC noise, and
strain/temperature sensitivities for the configured receiver mode. Optical
backscatter estimates use homogeneous attenuation; segments and bends are
not folded in. Material sensitivities use `strain.photoelastic_coefficient`,
`temperature.thermal_expansion` and `temperature.thermo_optic` from the config.

All `sigma_*` values are current standard deviations in A, referred through
the configured transimpedance. Their measurement points are:

| Quantity | Meaning |
| --- | --- |
| `I_dc_ref` | Reference-arm equivalent current before the recombiner (retained convention) |
| `I_dc_pd` | Half of `I_dc_ref`, incident on each photodiode |
| `bandwidth`, `filter_cutoff` | Requested cutoff and actual clamped cutoff, Hz |
| `noise_bandwidth` | One-sided equivalent noise bandwidth of the receiver filter, Hz |
| `sigma_shot`, `sigma_thermal`, `sigma_dark` | Filtered receiver contributions; disabled shot noise is zero |
| `sigma_receiver` | RSS of those three additive contributions |
| `sigma_rin` | Filtered beat modulation, requiring the noiseless pre-filter beat RMS when RIN is enabled |
| `sigma_analog` | RSS of receiver and RIN contributions, compared with `analog_main / input_impedance` |
| `sigma_quant`, `sigma_adc_extra` | Full-Nyquist quantization estimate and additional ENOB noise, generated after the analog filter |
| `sigma_total` | RSS of analog, quantization and ENOB contributions, with no post-ADC filtering |

For a noiseless acquisition, the optional RIN input can be computed as:

```python
I_beat = noiseless_acq.photocurrent_main[0] * cfg["detection"]["responsivity"]
beat_rms = float(np.sqrt(np.mean(I_beat**2)))
b = compute_budget(cfg, beat_rms_current=beat_rms)
```

Use a linear, unsaturated reference run with source RIN and other noise
disabled, and match the measurement interval (excluding settling). If RIN is
enabled but `beat_rms_current` is omitted, `sigma_rin`, `sigma_analog`,
`sigma_total`, `nep_total` and `dynamic_range_dB` are `NaN`; the additive
receiver estimate remains available. `print_budget` explains the missing
input instead of presenting a DC-based floor as a pipeline prediction.

These are linear, reference-dominated estimates: strong signal-arm power,
photodiode distortion/clipping, large RIN clipped at zero optical power,
ADC clipping, jitter and DNL/INL are outside the noise estimate. Quantization
assumes a uniform error distribution; the ADC's half-LSB floor-quantizer
bias is not a noise standard deviation. `nep_total = sigma_total / R` is an
integrated equivalent optical noise, not a density in W/sqrt(Hz).
`dynamic_range_dB` uses the configured saturation current, or the reference
current summed over the active photodiodes, as a proxy; it is not a measured
ADC-limited dynamic range.

`sigma_phi_far` remains the theoretical Lorentzian phase-difference RMS.
It is separate from the current-noise RSS and does not imply that the main
MZI simulates a stochastic white-FM pedestal: that component currently enters
only through deterministic coherence roll-off.

Receiver-noise validation covers both detector modes, orders 1/4/8, and
cutoffs below and above the sampling Nyquist limit (exercising the clamp).
Paired noisy/noiseless traces are subtracted before measuring settled RMS;
the shot, thermal, dark, combined receiver, beat RIN and sampled-total checks
use a 3% tolerance. Material-coefficient checks compare recovered strain and
temperature spectral shifts against the budget within 5%.

### Demodulation

`windowed_xcorr_strain` compares spatial reflectogram windows of `W` samples.
For reference `a` and measurement `b`, define `q[j] = b[j] * conj(a[j])`.
The local spectral correlation, apart from normalization, is

\[
C(\ell) = \sum_{j=0}^{W-1} q_j \exp(2\pi i\ell j/W).
\]

The default `peak_method="continuous"` locates the largest sampled magnitude
with an IFFT padded to `8W`, then maximizes `|C|^2` over the neighboring
interval of plus or minus 1/8 original lag bin. The bounded scalar search
uses an offset tolerance of `1e-10` bins. This is a solver stopping tolerance,
not a physical accuracy guarantee. Scaling each window before multiplication
avoids losing faint signals through underflow.

The recovered shift is `dnu = lag * sweep_range_hz / W`, and strain is
`-dnu / (center_freq * (1 - p_e))`. The lag is periodic and returned in
`[-W/2, W/2)`. Larger shifts alias; noise or decorrelation can also make
another peak win. Continuous refinement removes the coarse parabolic
interpolation bias, but cannot resolve those ambiguities.

Use `peak_method="parabolic"` to reproduce the previous, faster estimator.
It fits three samples of the unpadded correlation magnitude and can bias
fractional shifts substantially. Existing calls now use continuous refinement
and therefore have a higher processing cost. Fiber generation is unchanged.

Both methods return `NaN` for zero or nonfinite input windows. The continuous
method also returns `NaN` when fewer than two nonzero spatial samples overlap,
or when optimization fails. This is a minimal validity check, not a confidence
score. Tests use known phase ramps, both shift signs, periodic boundaries,
and the fractional shift corresponding to a 5 K temperature change.

#### Fractional-lag validation (#105)

The implementation uses fixed 8x coarse localization followed by continuous
refinement; it does not expose the configurable padding factor proposed in
the original issue. Validation therefore checks the shipped estimator against
known shifts, rather than treating a padding-factor sweep as an implemented
feature.

`tests/test_fractional_lag_validation.py` checks 39 lags from -0.95 to +0.95
bins and four cases straddling the periodic boundaries, for three Rayleigh
seeds and window sizes 64 and 488. The circular lag error must be below
`1e-7` bins. The legacy parabolic method is retained as a negative comparison:
its maximum error over the 39 interior lags exceeds 0.05 bins for each profile.
These are noiseless numerical checks, not physical resolution claims.

The same file tests full-pipeline recovery of +/-17 microstrain and +/-5 K
on a 0.2--0.8 m segment of a 1 m fiber, for two seeds and both detector modes.
Reference sweep 0 and measurement sweep 1 share the fiber seed. Undoing the
known perturbation must recover the same complex fiber fingerprint. Subtracting
each matching noiseless receiver output gives nonzero receiver-noise residuals
whose absolute Pearson correlation must be below 0.03. Changing the global seed
would change the fiber too, so it is not a substitute for independent sweeps.

This test uses a linear 40 nm sweep at 1550 nm, 131072 samples at 20 MS/s,
10 mW laser power, an 8 MHz fourth-order receiver filter, shot noise,
`1e-12 W/sqrt(Hz)` thermal NEP and `1e-9 A` dark current. The ADC has 18 bits,
a 0.2 V span and 1000 ohm transimpedance; clipping is checked explicitly.
For non-overlapping 1 cm gauges wholly inside the segment, the absolute median
bias must be below 5% of the imposed magnitude and the window RMSE below 10%.
The median offset before the segment must also be below 5% of that magnitude.
The existing strain robustness tests additionally use independent sweeps at
2 mm, 1 cm and 5 cm gauges, with shot noise on and off.

The new tests record numerical errors and recovery/noise metrics as pytest
JUnit properties. For example:

```bash
pytest tests/test_fractional_lag_validation.py -o junit_family=xunit1 --junitxml=validation.xml
```

These finite, reproducible cases cover the regression's receiver-noise regime.
They do not establish accuracy under fiber decorrelation, phase noise, RIN,
clipping, arbitrary SNR, or experimental instrument conditions. The original
issue's configurable-factor proposal is superseded by validation of the fixed
coarse search plus continuous refinement described above.

### Quality metrics
Spatial (`analysis/spatial_metrics.py`): peak SNR, integrated reflected power per region, baseline noise floor, dead-zone.
Strain (`analysis/strain_quality.py`): RMS error vs ground truth, white and 1/f noise slopes, non-overlapping Allan variance.

### Spectral diagnostics
`analysis/spectral.py`, thin wrappers over scipy.signal (Welch, periodogram) with OFDR-sensible defaults, for checking noise slopes and sweep-nonlinearity spurs.

### Pipeline-level validation
_Work in progress._

## Conventions
- attenuation envelope is a field, but `|.|^2` is round-trip power,  hence `10^(-loss_dB/10)` (not /20) on field steps.
- Rayleigh coefficient in dB is backscattered power per metre, `10^(x/10)`
- seeding via `utils.seeding.derive_seed(seed, component=, core=, sweep=, sub=)`, one RNG per component x noise type, so toggling one source doesn't reseed the others
- every step works through `self.bk.xp` (numpy today, cupy will be implemented in V2).
