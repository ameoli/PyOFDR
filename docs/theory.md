# Theory
This document covers the physics behind each pipeline stage.
For the package layout see `architecture.md`. YAML config in `config_reference.md`.

## Signal chain
```
SweptLaser -> FiberGenerator -> StrainPerturbation -> TemperaturePerturbation
           -> MachZehnder -> Detector -> AntiAliasFilter -> ADC
```
For a perfectly linear sweep the beat frequency of a scatterer at distance
`z` is `f_beat(z) = 2 n gamma z / c`, linear in z. Here `n` is the group
index, `gamma = delta_nu / T_sweep` the chirp rate, factor 2 for the round
trip. The whole time-domain beat is then the inverse transform of the
attenuation weighted fiber profile:
    s(t) = IFFT[ profile(z) * attenuation(z) + ghosts(z) ] * n_samples

(`optics/mach_zehnder.py`). O(N log N) instead of the O(N^2) of a
per-scatterer sum, and exact while the chirp stays linear. Mode hops and
sweep nonlinearity break that assumption so we need the
auxiliary MZI k-clock or the MZI time-warp (see optics section).

Two consequences for the sampling grid:
- spatial resolution `dz = c / (2 n delta_nu)` (`core/config.py`, `compute_derived`),
- highest beat `f_beat_max = 2 n gamma L / c`. A config with `f_beat_max >= fs/2` is rejected at validation (`core/config_models.py`, `_check_nyquist`).

Ref: Eickhoff & Ulrich, Appl. Phys. Lett. 39, 693 (1981) (first OFDR);
Soller et al., Opt. Express 13, 666 (2005).

## Source
`source/swept_laser.py` builds the optical field one sweep at a time.

### Chirp
Instantaneous frequency is a linear ramp plus optional nonlinearity:
    nu(t) = nu_start + gamma t + a2 t^2 + a3 t^3 + A_r sin(2 pi t / T_r)
with `gamma = delta_nu / T_sweep` and the wavelength-to-frequency conversion `delta_nu = (c / lambda^2) delta_lambda` (`utils/units.py`,
`wavelength_range_to_freq_range`). Zero coefficients give the ideal chirp. Phase is the rectangle-rule integral of nu, and the field carries it on the power envelope:
    phi(t) = 2 pi cumsum(nu) dt
    E(t)   = sqrt(P(t)) exp(j phi(t))

### Phase noise
Frequency-noise PSD is a power-law sum,
    S_nu(f) = h_0 + h_-1 / f + h_-2 / f^2
generated in `source/phase_noise.py` and added to `nu_inst` itself, so it propagates to the field phase, the aux MZI clock and the main-MZI time-warp (not just to `E_source`, whose phase the beat model never sees directly -- #82):

- white FM, a Lorentzian linewidth `Delta_nu`: per-step `d_phi ~ N(0, sqrt(2 pi Delta_nu dt))`, cumulative sum is a Wiener process
- flicker FM (1/f) and random-walk FM (1/f^2): FFT-shaped Gaussian noise (Timmer & Koenig), integrated into phase. Each parametrized by its  RMS in Hz over the sampled band.

The 1/f shaping is from `utils/colorednoise.py`.

Ref: Timmer & Koenig, Astron. Astrophys. 300, 707 (1995); Di Domenico et
al., Appl. Opt. 49, 4801 (2010).

### RIN
Relative intensity noise = white multiplicative noise on the power:
    P(t) = P0 (1 + m(t)),   m ~ N(0, sqrt(RIN_lin * BW))
with `RIN_lin = 10^(RIN_dB_per_Hz / 10)` and `BW = fs/2`.

### Power envelope
Tunable lasers droop toward the sweep edges. Parabolic model:
    P(u) = P0 (1 - 4 e u^2),   u = t/T_sweep - 0.5,   e = 1 - 10^(-edge_dB/10)
`edge_dB = 0` is flat. The envelope and the RIN both reach the detector because the beat carries `P(t)` pointwise, not an average (see optics section).

### Sweep nonlinearity
The `a2`, `a3`, `A_r`, `T_r` terms above are the deviation from the linear ramp.
The MZI time-warp models their effect on the beat; aux-MZI k-clock resampling
can compensate it in post-processing.

Ref: Glombitza & Brinkmeyer, J. Lightwave Technol. 11, 1377 (1993)

## Fiber
`fiber/profile.py` draws the Rayleigh profile, the rest of the fiber physics hangs off it as submodules.

### Rayleigh backscatter
Every spatial bin `dz` holds a circular-gaussian complex phasor:
    r(z) = sigma/sqrt(2) * (re + j im),   re, im ~ N(0,1)
with `sigma = sqrt(R_lin * dz)` and `R_lin = 10^(R_dB/10)` the backscattered power per metre. Magnitude squared is exponential — the usual OFDR speckle. Per-segment `R(z)` is allowed (so sigma varies along z), and each core gets its own independent draw.

Ref: Gifford et al., distributed temperature sensing using Rayleigh backscatter, ECOC (2005).

### Attenuation
Beer-Lambert. dB/km to Np/m is
    alpha_Np_m = alpha_dB_km * ln(10) / 10000
and the round-trip amplitude envelope is `env(z) = exp(-alpha_Np_m z)` (`fiber/attenuation.py`). Worth stating the convention: the envelope multiplies a field but represents round-trip power loss, because light walks the fiber twice (so field amplitude round-trip equals one-way power in dB). With heterogeneous segments we build an `alpha(z)` vector and integrate it cumulatively.

### Macrobending
`fiber/bends.py` provides three alternatives. The default `exponential` model
uses `loss_per_turn = A exp(-R/R_c)` in dB/turn, with empirical starting values
`A=100 dB/turn` and `R_c=5 mm`. The `tabulated` model interpolates log(loss)
linearly against radius; it clamps to the nearest endpoint outside the table
and issues a warning. Neither model infers wavelength dependence.

The `marcuse` model uses a circular step-index core of radius `a` and infinite
cladding, with phase indices `n1 > n2`. Its straight-fiber LP01 mode is found
from

    k = 2*pi/lambda
    V = k*a*sqrt(n1^2 - n2^2)
    U^2 + W^2 = V^2
    U*J1(U)/J0(U) = W*K1(W)/K0(W)
    kappa = U/a, gamma = W/a
    beta = sqrt((n2*k)^2 + gamma^2)

The solver brackets the root in `log(W)` so small cladding decay constants
remain resolved. `J` and `K` are the ordinary and modified Bessel functions.
This scalar weak-guidance mode is not a full vector HE11 solution. The
implementation accepts `0 < V < 2.4048255577` and
`delta=(n1^2-n2^2)/(2*n1^2) <= 0.01`; the latter is a conservative numerical
scope restriction, not an exact physical boundary for weak guidance.
The phase indices must be supplied separately from the OFDR group index.

Writing `q = 2*alpha` for the **power** decay coefficient in inverse metres,
Marcuse's LP01 expression is

    R_eff = radius_correction * R
    q = sqrt(pi)*kappa^2 / (2*gamma^(3/2)*V^2*sqrt(R_eff)*K1(W)^2)
        * exp(-2*gamma^3*R_eff/(3*beta^2))
    P_out/P_in = exp(-q*L)
    loss_dB = (10/ln(10))*q*L,  L = 2*pi*R*turns

The usual silica stress correction is `R_eff/R = 1.28`; setting it to 1
recovers the uncorrected model. It changes the radiation coefficient while
the arc length still uses the physical radius. The implementation evaluates
the expression in log form and lets negligible losses underflow to zero.
It does not exponentiate a large positive optical depth to convert to dB.

For all three models, the total one-way bend loss is spread uniformly over
`[start, end)`. Its cumulative contribution at a scatterer position `z` is
integrated over the physical interval, including fractional cells:

    covered = max(0, min(z, end) - max(0, start))
    bend_loss_to_z = total_loss_dB * covered / (end - start)

For a bend inside the fiber this gives zero at its start and the full loss
at or beyond its end, even if the entire bend lies between two grid points.
Overlapping bends add in dB. At the last sampled position, a bend that extends
further towards the fiber end contributes only the portion already traversed.
The same cumulative loss is used for direct returns and ghost paths. Direct
returns see the round-trip field factor `10^(-loss_to_z/10)`; ghosts count
the full or partial bent interval on every passage.

`tests/test_bend_grid.py` checks shifted and sub-cell bends, different spatial
grids, overlap with other attenuation, and partial ghost crossings, with
`1e-12` relative tolerances on nonzero losses or field factors.

The analytic calculation uses the source centre wavelength and fixed phase
indices. During an OFDR sweep the resulting envelope is held constant; this
does not model spectral bend filtering or material dispersion across a wide
sweep. The approximation assumes weak guidance, gently curved fiber
(`R_eff` much larger than `a`) and a nearly unchanged mode shape. Coating
reflections, transition loss, whispering-gallery oscillations and bend-induced
birefringence remain outside the model (#57). Predicting a particular
commercial fiber still requires its optical parameters and measurements.

References: [Marcuse, JOSA 66, 216-220 (1976)](https://doi.org/10.1364/JOSA.66.000216);
[Martins et al., Conftele 2009, Eq. (4) and Table I](https://www.researchgate.net/publication/231179029_Modeling_of_Bend_Losses_in_Single-Mode_Optical_Fibers).
The effective-radius treatment is also discussed by
[Schermer and Cole, IEEE JQE 43, 899-909 (2007)](https://doi.org/10.1109/JQE.2007.903364).

For the Martins waveguide (`a=4.1 um`, `n2=1.440`, `NA=0.117`), numerical
reference values at `R=10 mm`, one turn and correction 1.28 are:

| wavelength | W | one-way loss [dB] |
| --- | --- | --- |
| 1310 nm | 1.63441932659613 | 0.0592091599471278 |
| 1550 nm | 1.22651814034357 | 3.35620476859821 |
| 1625 nm | 1.12338156463537 | 8.21001752202825 |

These values were computed independently with 70-digit arithmetic, bisecting
`b=W^2/V^2` for 240 iterations with the cross-multiplied continuity equation.
They are evaluations of the stated model using published waveguide parameters,
not measured values or a reproduction of the paper's experimental curves.
`tests/test_marcuse_bends.py` checks nine loss cases at relative tolerance
`2e-10`, mode parameters at `2e-12`, continuity at the core boundary, physical
arc length, unit scaling, invalid inputs, and integration with direct and ghost
attenuation. No extra dependency is needed to run these tests.

### Discrete reflectors and connectors
A point reflector adds `r = sqrt(R) exp(j phi)` at its nearest spatial bin
(`fiber/reflectors.py`). Each core has its own seeded reflector phases, drawn
once and reused across sweeps. This stream is separate from the Rayleigh RNG.
Reflectors are sorted by position before drawing phases. Two entries that
round to the same bin are rejected: their internal ordering is not resolved.

We choose real field transmission `t = sqrt(1-R)`. Reflection from the right
is then `-conj(r)`, so the lossless interface has the reciprocal, unitary
matrix `[[r, t], [t, -conj(r)]]`. This fixes the relative phases of direct
returns and ghosts; the phases of separate bounces are not drawn independently.
For the underlying two-port constraints, see the scattering-matrix discussion
in [Karlsson's electrodynamics notes, section 3.7](https://www.eit.lth.se/fileadmin/eit/courses/eitn80/root.pdf).

`loss_dB` is an extra matched loss just before the interface, on its input
(smaller-z) side. Writing `q = 10^(-loss_dB/20)`, the left reflection is
`q^2 r`, right reflection is `-conj(r)`, and transmission is `q t` in either
direction. This preserves the earlier convention that a direct reflection
sees its own insertion loss twice. It also avoids counting the reflected
fraction twice: `loss_dB` is additional to the loss of transmitted power `R`.

The direct-return envelope includes `1-R` beyond each reflector and
`10^(-loss_dB/10)` from its bin onwards. The reflector's own return does not
cross its interface. Rayleigh and weak-FBG returns at that same bin use the
upstream-side convention. With `R=1`, all later direct returns vanish.

### Multiple-scattering ghosts
Light can bounce between discrete reflectors and return at a later delay
(`fiber/multiple_scattering.py`). An order-N path has `2N-1` reflections and
apparent depth

    z_app = z_i0 - z_i1 + z_i2 - ... + z_i(2N-2)

Every odd "valley" must sit strictly below both neighbours. Positions are
rounded to bins before forming this sum. Each path multiplies the directional
reflection coefficients and the transmission of every intervening interface,
including the initial trip from the input and the final trip back.

Distributed loss is integrated from the input to each bin. Its value at zero
is zero; the loss assigned to bin k applies to `[z[k], z[k+1])`. For each path
leg, the difference of these cumulative losses gives the one-way loss of that
leg. Summing the legs and converting with `10^(-path_loss_dB/20)` gives the
field attenuation. Segments and bends therefore count on every crossing.
For uniform loss this agrees with a total distance `2*z_app`; for nonuniform
loss, the visited sections matter.

For example, the first ghost of two lossless interfaces at `a < b` is
`-(1-R_a) conj(r_a) r_b^2` at `2*b-a`, before distributed attenuation.
The two paths `(b,a,c)` and `(c,a,b)` share a delay and add coherently.

`Acquisition.ghost_profile` stores these already attenuated returns separately.
The optics step adds them to `fiber_profile * attenuation_envelope`, then
applies the usual coherence roll-off and beat transform. Dividing by the
direct envelope would fail behind an opaque reflector or an underflowed loss,
even when a shorter physical path still produces a ghost at that delay.

The expansion still stops at `max_order` (2 or 3), and out-of-range ghosts
are dropped. It is a finite-bounce model, not a converged high-finesse cavity
solver. Index, strain and temperature perturbations still use the phase at
the apparent bin, so nonuniform perturbations along a ghost path remain an
approximation. Rayleigh-mediated multiple scattering is outside this model.

`tests/test_reflector_paths.py` checks complex two-mirror amplitudes, repeated
crossings, localized loss, opaque interfaces, sweep reproducibility and the
optics sum. A separate frequency-domain recursion sums the infinite series
for three weak mirrors; its impulse response must agree within `1e-10` field
amplitude, where omitted orders are below that tolerance. These are numerical
checks of the scalar model, not experimental validation.

### Strain
A scatterer at z picks up an extra round-trip phase from the integrated strain (`fiber/strain.py`):

    d_phi(z) = 2 k0 n (1 - p_e) integral_0^z eps(z') dz'
with `k0 = 2 pi / lambda` and `p_e ~ 0.22` the photoelastic coefficient at 1550 nm. The `(1 - p_e)` factor folds in the index change from the
strain itself, so the geometric stretch isn't double counted. Code multiplies the profile by `exp(j phase)` with a left-Riemann cumsum.

Ref: Froggatt & Moore, Appl. Opt. 37, 1735 (1998); Bertholds &
Dandliker, J. Lightwave Technol. 6, 17 (1988).

### Strain transfer host to fiber
Two models in `strain_transfer/`:

- ideal: `eps_fiber = eps_host` inside the segment, zero outside.
- Cox shear-lag: `eps_fiber(s) = eps_host (1 - cosh(beta s)/cosh(beta L_h))`   with `s` from segment centre, `L_h` the half-length and `beta` a shape   constant [1/m] passed in config — lumps bonding shear modulus, geometry   and Young's modulus, not derived from elastic constants here.

Ref: Cox, Br. J. Appl. Phys. 3, 72 (1952).

### Dynamic strain motions
Motions in `strain_transfer/motions.py` are sampled at lab time `t = sweep_index * T_sweep` (one value per sweep) and collapsed to a single `eps(t)` per segment, added to the static offset:

- harmonic: `eps(t) = A sin(2 pi f t + phi)`, warns if
  `f >= 1/(2 T_sweep)` (sweep-rate Nyquist),
- thermal relaxation: `eps(t) = A (1 - exp(-t/tau))`,
- impulsive: `eps(t) = A exp(-(t-t0)^2 / 2 sigma^2)` (not zero-mean,  leaves a residual),
- random vibration: `eps(t) ~ N(0, A^2)`, stateless via a SeedSequence   on `(seed, t)` so it's reproducible without a buffer.

### Index perturbation (small-signal dn)
Deterministic segments rotate the round-trip phase without moving the bins (`fiber/profile.py`):
    d_phi(z) = 2 k0 integral_0^z delta_n dz'
same as strain but without the `(1 - p_e)` factor. On top of the segments there's a stochastic slice: an Ornstein-Uhlenbeck process discretised as AR(1)
    dn[k] = a dn[k-1] + sigma sqrt(1 - a^2) eta[k],   a = exp(-dz / L_corr)
started from the stationary distribution (`dn[0] = sigma eta[0]`), so the variance is `sigma^2` and the lag-1 autocorrelation is `exp(-dz/L_corr)`.
The full OPL-based n(z) remap (splices SMF/DCF, hollow-core, tapers, absolute metrology) is the rest of #33, still V2.

Ref: Froggatt & Moore, Appl. Opt. 37, 1735 (1998).

### Distributed temperature
Temperature segments add (`fiber/temperature.py`)
    d_phi(z) = 2 k0 n (alpha_L + xi) integral_0^z dT dz'
with silica defaults `alpha_L = 5.5e-7 /K` (thermal expansion) and `xi = (1/n) dn/dT = 6.5e-6 /K` (thermo-optic). Added to the strain phase this closes the Froggatt-Moore relation
    df/f = -(1 - p_e) eps - (alpha_L + xi) dT
so pure temperature looks like an apparent strain `eps_app = (alpha_L + xi)/(1 - p_e) dT`, about 9.04 ue/K for silica at 1550 nm. Segments can carry a `motion` (see above) read as `dT(t)`.
The 1D thermal-diffusion PDE along z is V2 (#76).

### MCF core-to-core crosstalk
Scalar phase-scrambled model (level A) for multicore fibre (`fiber/crosstalk.py`):
    c(z) = sqrt(XT_lin * z/1000)
    profile_target(z) += c(z) exp(j phi_rand(z)) profile_source(z)
with `XT_lin = 10^(XT_dB_per_km/10)` and an independent U(0, 2pi) phase per bin per directed pair. Topologies `hex7` and `linear`.
Weak-coupling limit — the power leaking out isn't subtracted from the source cores (fine for XT well below -20 dB/km). Full coupled-mode is V2 (#69).

Ref: Hayashi et al., Opt. Express 19, 16576 (2011).

### FBG arrays (Born approximation)
A weak Bragg grating acts as a reflector with a sinc spectral response (`fiber/fbg.py`):

    r(t) = sqrt(R_max) sinc(2 n L_g (nu(t) - nu_B) / C)

with `nu_B = C/lambda_B`, `L_g` the physical grating length and
`sinc(x) = sin(pi x)/(pi x)`. First null at `Delta_nu = C/(2 n L_g)`.
Injected into the beat after the time-warp; in z-space the FBG is not a
delta but a rect of width `L_g/dz` bins. Born is fine for `R_max <~ 0.5`;
coupled-mode / transfer-matrix for strong gratings is V2 (#78).

The configured `bragg_wavelength` is the unperturbed resonance. At each sweep,
the FBG reads the applied strain and temperature change at its spatial bin:

    lambda_B_eff = lambda_B * (1 + (1 - p_e) eps + (alpha_L + xi) dT)
    nu_B = C / lambda_B_eff

Coefficients come from `strain.photoelastic_coefficient`,
`temperature.thermal_expansion` and `temperature.thermo_optic`. Strain is
taken after host-to-fiber transfer, including Cox when selected. Missing
fields mean zero perturbation. Temperature expansion is already included
in the thermal term; do not add that same free-fiber expansion to the strain
field again.

This is a small-perturbation model with a uniform field over each grating,
sampled at the same rounded z-bin used for its beat phase and attenuation.
It shifts the resonance without changing grating length, bandwidth, peak
reflectivity or propagation phase. Gradients along a grating and the resulting
spectral distortion are not represented. The configured wavelength stays
unchanged between sweeps; an effective wavelength that is nonpositive or
nonfinite is rejected.

Ref: Erdogan, J. Lightwave Technol. 15, 1277 (1997).

## Optics
`optics/mach_zehnder.py` turns the profile into a photocurrent, plus the
passive components in `optics/components.py` and the aux MZI.

### Beat amplitude and scaling
After the Fourier sum the main photocurrent comes out as

    photocurrent(t) = prefactor * P(t) * Re[beat]
    prefactor = 2 sqrt(eta (1 - eta)) IL_circ^2

Here `P(t) = |E_source|^2` is instantaneous, not averaged, so the power
envelope and RIN reach the detector as a multiplicative modulation on the
AC beat. The slow-envelope approximation `sqrt(P(t) P(t-tau)) ~ P(t)` applies
while the round-trip delay is much shorter than the envelope timescale.

### Sweep duration and sample grids

For requested duration `T` and ADC rate `fs`, samples occur at `t[j] = j/fs`.
Their count is `N = ceil(T*fs)`, with products within four floating-point ULPs
of a positive integer treated as that integer. This prevents roundoff from
adding a sample; genuine fractional counts are retained. Neither `T` nor `fs`
is adjusted.

With optical span `B`, the fiber grid remains `dz = c/(2*n*B)`. Its coefficient
at bin `k` contributes `h[k] * exp(2*pi*i*k*t/T)`. An ordinary IFFT evaluates
this expression only when `T*fs = N`. Otherwise, the main MZI uses ZoomFFT
with the positive-exponent convention to evaluate the same sum on the actual
sample times. Transform coefficients are reused across sweeps. The FBG carrier
also uses `t/T`; its separate ideal-chirp phase approximation is unchanged.

The unpadded FFT grid is different from the fiber grid for fractional counts:

    frequency_step_hz = (B/T) / fs
    fft_sweep_range_hz = N * frequency_step_hz
    fft_dz = c / (2*n*fft_sweep_range_hz) = dz * T*fs/N

`Acquisition` and HDF5 output retain these quantities. The DFT span uses `N`
increments; the frequency difference between first and last samples uses
`N-1`. These are distinct conventions, not a change to the configured sweep.
`reflectogram_from_acq` applies `fft_dz` automatically. For array-based calls:

```python
H, z = fft_reflectogram(acq.digital_main[0], acq.fft_dz)
# For windowed_xcorr_strain on these unpadded reflectograms:
# dz=acq.fft_dz, sweep_range_hz=acq.fft_sweep_range_hz
```

Zero-padding refines the FFT grid further; use the returned `z` spacing and
scale the DFT span by `n_pad/N` when demodulating that padded reflectogram.
After k-clock resampling or cropping, calibrate from the new frequency step
and record length instead of reusing the raw-acquisition scale. For a linear
sweep, cropping to `M` samples gives `fft_dz * N/M`.

The nonlinear time warp evaluates the fiber's Fourier sum at warped times,
using the physical sweep period `T` even when `T*fs` is fractional. It uses
a separate oversampled grid; see below for accuracy and boundary behavior.

### Circulator
3-port scalar model (`optics/components.py`). Signal passes the circulator twice, so `round_trip_transmission = IL^2` in field = `10^(-IL_dB/10)` in power. Isolation (port1->port3 leakage) lands at zero delay as a DC offset riding P(t); return loss is a reflector sitting in the z=0 bin.

### Laser coherence roll-off
A finite Lorentzian linewidth `Delta_nu` makes the fringe visibility decay with the round-trip delay:
    V(z) = exp(-pi Delta_nu tau),   tau = 2 n z / C
applied as a z-dependent multiplier on the profile before the IFFT, the IFFT trick carries no explicit reference arm, so the decay is folded in.

The coloured (flicker / random-walk) FM noise instead enters stochastically through the time-warp: for noise slow compared to tau, `phi(t) - phi(t-tau) ~ 2 pi tau nu_noise(t)`, which is exactly the ripple mechanism, so each sweep gets a real phase-noise pedestal. The white FM term is deliberately kept out of the warp -- it decorrelates sample to sample and the small-tau approximation would overshoot its pedestal by ~tau/dt -- so the Lorentzian part appears only as the ensemble-average roll-off above (no stochastic pedestal).

Ref: Coupland & Pickering, J. Opt. Soc. Am. A 9, 257 (1992).

### Sweep-nonlinearity time-warp
For small round-trip delay `tau`, the frequency deviation `delta_nu(t)` adds
the beat phase `2*pi*tau*delta_nu(t)`. With nominal chirp rate `gamma`, this
is equivalent to evaluating the ideal beat at

    s(t) = t + delta_nu(t)/gamma
    beat(t) = sum_k h[k] exp(2*pi*i*k*s(t)/T)

Here `h` is the fiber profile after attenuation, ghosts and coherence roll-off.
The deviation includes the polynomial/ripple terms, slow colored FM and the
equivalent GVD term below. White FM retains its separate coherence treatment.
This is the forward model; auxiliary-MZI k-clock resampling is a subsequent
analysis operation that attempts to compensate the nonlinearity.

Linear interpolation on the ADC grid is unsuitable here: a half-sample shift
attenuates a tone at `0.4*fs` by 10.20 dB. Instead, the implementation centres
the known Fourier band around zero, evaluates it by IFFT on at least `16*n_z`
points, and uses a periodic quintic spline before restoring the carrier.
The centred band has at least 32 samples per fastest cycle. Cores are processed
one at a time to limit temporary memory; the internal grid scales with the
fiber size, independently of the ADC sample count.

The Fourier sum has period `T` in `s`, so evaluation beyond the nominal sweep
continues that sum rather than clamping to the first or last ADC sample.
Excursions greater than 5% of `T` still emit a warning. This continuation follows
the discrete-fiber model; it is not a model of laser reset or flyback.

`tests/test_time_warp_accuracy.py` compares tones and complex multicore profiles
with direct Fourier sums, including fractional sample counts, sweep boundaries,
polynomial distortion, ripple, GVD and slow FM. Tone errors are checked against
`2e-7` of unit field amplitude and profile relative RMS errors against `2e-7`.
The small-delay approximation remains, and sufficiently fast warps can still
alias at the ADC: the nominal linear-sweep Nyquist check does not bound all
instantaneous beat frequencies. FBG contributions retain their separate
ideal-chirp propagation-phase approximation.

### Chromatic dispersion (GVD)
Group-velocity dispersion enters through `beta2 = -lambda^2 D/(2 pi c)`.
It adds `4 pi^2 beta2 gamma^2 z (t - T/2)^2` to the beat phase, which factors as `z K(t)`, so it rides the same time-warp engine with an extra 
    delta_nu = pi beta2 gamma^2 (c/n) (t - T/2)^2
The warp itself is z-independent; the z-dependent peak broadening then falls out of the FFT (a chirp in time spreads in frequency). Only D / beta2 is modelled, third-order dispersion isn't (#34).

### Aux MZI and k-clock
An auxiliary MZI with a known delay tau, fed by the same swept laser (`optics/aux_mzi.py`):
    I_aux(t) = cos(phi(t) - phi(t - tau)) = cos(2 pi integral_{t-tau}^t nu du)
For a linear chirp it's a clean sine at `gamma tau`. When the sweep wobbles I_aux wobbles with it, so its unwrapped Hilbert phase is a monotone function of nu(t). `analysis.demodulation.kclock_resample` uses it as a clock to resample the main beat onto a uniform-nu grid, undoing the sweep nonlinearity before the FFT. The first `round(tau/dt)` samples are invalid, and non-monotone phase samples are dropped before `np.interp` (#64).

The implementation uses the integer lag `n_tau = round(tau/dt)`, so its
effective delay is `n_tau*dt`. Configuration validation requires the nominal
beat `gamma*n_tau*dt < fs/2`. At runtime, before forming the cosine, it checks
`max(abs(nu_inst[j] - nu_inst[j-n_tau])) < fs/2` for `j >= n_tau`.
This difference is the phase increment divided by `2*pi*dt` for the rectangle
integration used here. Checking the unwrapped frequency difference avoids
mistaking an already aliased tone for a valid low-frequency clock. Both signs
and local excursions are checked, including the realized laser frequency noise.
The delay must leave at least two valid samples to define the mean beat.

This is a sampling guard on the modeled frequency trace, not a guarantee of
strict spectral band limitation: modulation sidebands and any variation
between source samples are not bounded by this check. It also does not ensure
monotone k-clock phase; the resampling checks still apply.

Ref: Glombitza & Brinkmeyer, J. Lightwave Technol. 11, 1377 (1993).

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

Shot, thermal and dark noise are added afterwards. With saturation enabled,
the shot-noise variance is `2*e*I_collected*B` per diode, using its nonnegative
limited current before baseline removal/subtraction. Independent diode noise
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
- without saturation, single-ended shot: variance `2 e max(I_dc_pd + I_beat, 0) B`; the reference dominates for weak returns,
- without saturation, balanced shot: two independent contributions, each with variance `2 e I_dc_pd B`,
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
