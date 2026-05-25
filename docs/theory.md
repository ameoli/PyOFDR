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
    s(t) = IFFT[ profile(z) * attenuation(z) ] * n_samples

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
generated in `source/phase_noise.py` and added to `phi`:

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
The `a2`, `a3`, `A_r`, `T_r` terms above are the deviation from the linear ramp. Compensated either by the MZI time-warp or the aux-MZI k-clock in post-processing

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
Empirical Marcuse-style form,
    alpha_per_turn(R) = A exp(-R / R_c)    [dB/turn]
defaults `A = 100 dB/turn`, `R_c = 5 mm` for SMF-28 (`fiber/bends.py`).
Total loss `A turns exp(-R/R_c)` is spread uniformly over the bent segment as extra dB/km on top of the base alpha.
Transition loss, whispering-gallery ripples and bend birefringence are out of scope (V2).

Ref: Marcuse, "Curvature loss formula for optical fibers", JOSA 66, 216
(1976).

### Discrete reflectors and connectors
A point reflector just adds `sqrt(R)` into its bin of the profile (`fiber/reflectors.py`). A connector insertion loss is a multiplicative step on the envelope, `envelope[idx:] *= 10^(-loss_dB/10)`. Exponent /10 and not /20 because of the round-trip convention above.

### Multiple-scattering ghosts
Light reflected by one strong reflector can bounce off another and come back via a longer path, showing up as a ghost peak
(`fiber/multiple_scattering.py`). For an order-N path (2N-1 reflection events) the apparent depth and amplitude are
    z_app = z_i0 - z_i1 + z_i2 - ... + z_i(2N-2)
    a     = prod sqrt(R_im)
with the constraint that every odd "valley" reflector sits strictly below both neighbours. Ghosts whose apparent bin falls outside `[0, n_z)`
are dropped, extending the array to 2L would touch too much downstream code. The continuum Rayleigh-Rayleigh double-scatter floor is much weaker in standard SMF and tracked separately in #79.

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

Ref: Erdogan, J. Lightwave Technol. 15, 1277 (1997).

## Optics
`optics/mach_zehnder.py` turns the profile into a photocurrent, plus the
passive components in `optics/components.py` and the aux MZI.

### Beat amplitude and scaling
After the IFFT the main photocurrent comes out as

    photocurrent(t) = prefactor * P(t) * Re[beat]
    prefactor = 2 sqrt(eta (1 - eta)) IL_circ^2

with `P(t) = |E_source|^2` instantaneous, not averaged, so the power envelope and RIN reach the detector as a multiplicative modulation on the AC beat. Uses the slow-envelope approximation `sqrt(P(t) P(t-tau)) ~ P(t)`, good while the round-trip delay is much shorter than the envelope timescale.

### Circulator
3-port scalar model (`optics/components.py`). Signal passes the circulator twice, so `round_trip_transmission = IL^2` in field = `10^(-IL_dB/10)` in power. Isolation (port1->port3 leakage) lands at zero delay as a DC offset riding P(t); return loss is a reflector sitting in the z=0 bin.

### Laser coherence roll-off
A finite Lorentzian linewidth `Delta_nu` makes the fringe visibility decay with the round-trip delay:
    V(z) = exp(-pi Delta_nu tau),   tau = 2 n z / C
applied as a z-dependent multiplier on the profile before the IFFT, the IFFT trick carries no explicit reference arm, so the decay is folded in.
Only the Lorentzian term is modelled; flicker / random-walk would need the full phase structure function.

Ref: Coupland & Pickering, J. Opt. Soc. Am. A 9, 257 (1992).

### Sweep-nonlinearity time-warp
_Work in progress._

### Chromatic dispersion (GVD)
Group-velocity dispersion enters through `beta2 = -lambda^2 D/(2 pi c)`.
It adds `4 pi^2 beta2 gamma^2 z (t - T/2)^2` to the beat phase, which factors as `z K(t)`, so it rides the same time-warp engine with an extra 
    delta_nu = pi beta2 gamma^2 (c/n) (t - T/2)^2
The warp itself is z-independent; the z-dependent peak broadening then falls out of the FFT (a chirp in time spreads in frequency). Only D / beta2 is modelled, third-order dispersion isn't (#34).

### Aux MZI and k-clock
An auxiliary MZI with a known delay tau, fed by the same swept laser (`optics/aux_mzi.py`):
    I_aux(t) = cos(phi(t) - phi(t - tau)) = cos(2 pi integral_{t-tau}^t nu du)
For a linear chirp it's a clean sine at `gamma tau`. When the sweep wobbles I_aux wobbles with it, so its unwrapped Hilbert phase is a monotone function of nu(t). `analysis.demodulation.kclock_resample` uses it as a clock to resample the main beat onto a uniform-nu grid, undoing the sweep nonlinearity before the FFT. The first `round(tau/dt)` samples are invalid, and non-monotone phase samples are dropped before `np.interp` (#64).

Ref: Glombitza & Brinkmeyer, J. Lightwave Technol. 11, 1377 (1993).

## Detection
`detection/detector.py` and `detection/filter.py`.

### Photodetection
Current is responsivity times optical power, `I(t) = R P_opt(t)`, with R in A/W (typ. 1.0 for InGaAs at 1550 nm).

### Saturation
Symmetric clip at `+/- I_sat` before noise is added — the TIA noise goes on the already-clipped current.

### Electronic noise
Independent gaussians:

- shot: variance `2 e |I| B`. In balanced the signal term is negligible and `I_dc = R eta P_laser` is used,
- thermal / NEP: `sigma = R NEP sqrt(B)`, NEP in W/sqrt(Hz),
- dark current: `sqrt(2 e I_dark B)`.

`B = fs/2` (sampling Nyquist band) at noise-generation time;
the anti-alias filter then trims the useful band.

### Balanced detection
Two photodiodes on complementary arms:
    I_a = I_dc + I_beat + noise_a
    I_b = I_dc - I_beat + noise_b
    I_out = (I_a - I_b)/2 = I_beat + (noise_a - noise_b)/2
DC cancels, beat adds, independent noises combine as sqrt(2), so the SNR gain is 3 dB over single-ended.

### Anti-alias filter
Butterworth low-pass (configurable order, typ. 4), implemented as second-order sections for stability, cutoff clipped to `0.99 fs/2`
(`detection/filter.py`).

### Photodiode nonlinearity
A pre-TIA polynomial of order >= 2,
    I_out = I_in + sum_{k>=2} a_k I_in^k
In balanced mode it acts per arm (`I_A = I_dc + I_beat`, `I_B = I_dc - I_beat` reconstructed from the DC current, each through the polynomial, then differenced), which captures the DC*beat mixing (even orders) that the single-ended path misses

## ADC
`digitizer/adc.py`.

### Quantization
Uniform mid-tread, `2^bits` levels, hard clip at `+/- V_range/2`, theoretical RMS
    sigma_q = V_range / (2^bits sqrt(12))

### ENOB
The effective number of bits lumps jitter + INL + analog noise into one number, injected as an extra gaussian before quantization:
    sigma_total = V_range / (2^enob sqrt(12))
    sigma_extra = sqrt(sigma_total^2 - sigma_q^2)

### Aperture jitter
A timing error `dt_err ~ N(0, sigma_j)` becomes a voltage error to first order, `dV = (dV/dt) dt_err`, with the slope from finite differences.

### DNL / INL
Fixed once "fabricated": one realization `curve[k]` (LSB offset per code) built at init. INL is a single-cycle sine over the code range with amplitude `inl_peak_lsb`; DNL is a cumulative random walk of step `N(0, dnl_rms_lsb)`, detrended linearly (otherwise it's just a gain error).

Ref: Kester, The Data Conversion Handbook

## Analysis
Post-processing under `analysis/`; reads the simulated output back.

### Budget
`analysis/budget.py` is algebra over the config: optical power chain (laser, splitter, circulator, fiber), Rayleigh backscatter at z=0 and z=L with round-trip attenuation, noise floor as the RSS of shot / thermal / dark / quantization, plus RIN, phase-noise and strain/temp sensitivity terms. Reports total NEP and a dynamic range. Uses only the homogeneous attenuation — segments and bends aren't folded in.

### Demodulation
_Work in progress._

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
