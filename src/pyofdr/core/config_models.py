"""Pydantic models for the YAML config. See #10."""

from __future__ import annotations

from typing import Annotated, Literal

from pint import UnitRegistry
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from pyofdr.utils.constants import C as _C

_ureg = UnitRegistry()


def _parse(unit: str):
    def parser(v):
        if isinstance(v, (int, float)):
            return float(v)
        return float(_ureg.Quantity(v).to(unit).magnitude)
    return parser


# unit-aware float aliases. accept either a bare number (already in
# SI) or a string like "40 nm", "10 ms", ...
Length    = Annotated[float, BeforeValidator(_parse("meter"))]
Time      = Annotated[float, BeforeValidator(_parse("second"))]
Frequency = Annotated[float, BeforeValidator(_parse("hertz"))]
Power     = Annotated[float, BeforeValidator(_parse("watt"))]
Voltage   = Annotated[float, BeforeValidator(_parse("volt"))]
Current   = Annotated[float, BeforeValidator(_parse("ampere"))]
Resistance = Annotated[float, BeforeValidator(_parse("ohm"))]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SimulationConfig(_Strict):
    seed: int = 42
    backend: Literal["numpy", "cupy", "jax"] = "numpy"
    n_sweeps: int = Field(1, ge=1)


class ReflectorEntry(_Strict):
    z: Length
    R: float = Field(..., ge=0, le=1)
    loss_dB: float = Field(0.0, ge=0)   # one-way insertion loss [dB]


class RayleighSegment(_Strict):
    start: Length
    end:   Length
    rayleigh_coefficient_dB: float

    @model_validator(mode="after")
    def _check_order(self):
        if self.end <= self.start:
            raise ValueError(f"rayleigh segment: end ({self.end}) must be > start ({self.start})")
        return self


class AttenuationSegment(_Strict):
    start: Length
    end:   Length
    attenuation_dB_per_km: float = Field(..., ge=0)

    @model_validator(mode="after")
    def _check_order(self):
        if self.end <= self.start:
            raise ValueError(f"attenuation segment: end ({self.end}) must be > start ({self.start})")
        return self


class IndexSegment(_Strict):
    """Static delta_n perturbation on a fiber section. Small-signal: the
    scatterer bin positions don't move, only their round-trip phase does.
    See #68; full z-dependent n(z) (with OPL-based dz) is in #33.
    """
    start:   Length
    end:     Length
    delta_n: float    # deviation from n_core, signed

    @model_validator(mode="after")
    def _check_order(self):
        if self.end <= self.start:
            raise ValueError(f"index segment: end ({self.end}) must be > start ({self.start})")
        return self


class BendSegment(_Strict):
    start:  Length
    end:    Length
    radius: Length = Field(..., gt=0)
    turns:  float  = Field(..., gt=0)
    # SMF-28 ballparks; override for other fibers
    A_dB_per_turn: float = Field(100.0, gt=0)
    R_c:           Length = Field(5e-3,  gt=0)

    @model_validator(mode="after")
    def _check_order(self):
        if self.end <= self.start:
            raise ValueError(f"bend segment: end ({self.end}) must be > start ({self.start})")
        return self


class CrosstalkConfig(_Strict):
    """Core-to-core crosstalk for MCF. Phase-scrambled scalar model
    (level A), see #47. Full coupled-mode equations tracked in #69.
    """
    xt_dB_per_km: float    # integrated per-pair XT after 1 km (negative)
    topology: Literal["hex7", "linear"] = "hex7"


class FiberConfig(_Strict):
    length: Length = Field(10.0, gt=0)
    n_core: float = Field(1.4682, gt=1.0)
    n_cores: int = Field(1, ge=1)
    rayleigh_coefficient_dB: float = -82.0
    rayleigh_segments: list[RayleighSegment] = Field(default_factory=list)
    attenuation_dB_per_km: float = Field(0.0, ge=0)
    attenuation_segments: list[AttenuationSegment] = Field(default_factory=list)
    bends: list[BendSegment] = Field(default_factory=list)
    reflectors: list[ReflectorEntry] = Field(default_factory=list)
    index_segments: list[IndexSegment] = Field(default_factory=list)
    crosstalk: CrosstalkConfig | None = None


class SourceConfig(_Strict):
    center_wavelength: Length    = Field(1550e-9, gt=0)
    sweep_range:       Length    = Field(40e-9,   gt=0)
    sweep_duration:    Time      = Field(0.01,    gt=0)
    power:             Power     = Field(10e-3,   gt=0)
    linewidth:         Frequency = Field(0.0, ge=0)   # FWHM Lorentzian
    # coloured FM contributions beyond the Lorentzian white term.
    # Both are RMS frequency excursions [Hz] over the sampled bandwidth.
    flicker_noise_Hz:     float = Field(0.0, ge=0)   # 1/f
    random_walk_noise_Hz: float = Field(0.0, ge=0)   # 1/f^2
    rin_dB_per_Hz:     float | None = None   # RIN power spectral density [dB/Hz]
    # power envelope across the sweep: P(nu) drops by this many dB at the
    # sweep edges, parabolic shape (0.0 = flat / backward compat)
    power_envelope_edge_dB: float = Field(0.0, ge=0)
    # sweep nonlinearity: polynomial deviation from the ideal linear chirp.
    # nu(t) = nu_start + gamma*t + a2*t^2 + a3*t^3 + ripple
    # a2 [Hz/s^2], a3 [Hz/s^3], all default 0 (perfectly linear).
    sweep_nonlinearity_a2: float = 0.0
    sweep_nonlinearity_a3: float = 0.0
    # sinusoidal ripple on the instantaneous frequency [Hz amplitude]
    sweep_ripple_amplitude: Frequency = Field(0.0, ge=0)
    # ripple period [s] -- zero disables the ripple
    sweep_ripple_period: Time = Field(0.0, ge=0)


class HarmonicMotion(_Strict):
    """Sinusoidal strain modulation in lab time (across sweeps)."""
    kind:      Literal["harmonic"] = "harmonic"
    amplitude: float     = Field(..., ge=0)      # strain amplitude
    frequency: Frequency = Field(..., gt=0)      # Hz
    phase:     float     = 0.0                   # radians


class ThermalRelaxation(_Strict):
    """First-order exponential approach: eps(t) = A * (1 - exp(-t/tau)).

    amplitude can be negative (cooling) -- the sign matters here, unlike
    harmonic where phase absorbs it.
    """
    kind:      Literal["thermal"] = "thermal"
    amplitude: float = Field(...)           # strain asymptote (signed)
    tau:       Time  = Field(..., gt=0)     # relaxation time constant [s]


class ImpulsivePulse(_Strict):
    """Gaussian-in-time strain pulse -- impact / crack / PZT click.

    eps(t) = A * exp(-(t - center_time)^2 / (2 * width^2))

    width is the std dev (not FWHM). gaussian is not zero-mean so the
    fibre sees a residual 'tail' after the pulse; good enough for impulse-
    like events. zero-mean variants (ricker, damped sine) would go as
    separate motion kinds.
    """
    kind:        Literal["impulsive"] = "impulsive"
    amplitude:   float = Field(...)             # peak strain (signed)
    center_time: Time  = Field(..., ge=0)       # peak time, lab frame [s]
    width:       Time  = Field(..., gt=0)       # gaussian std dev [s]


class RandomVibration(_Strict):
    """Broadband white-gaussian strain noise sampled per sweep.

    eps(t) ~ N(0, amplitude^2). PSD is flat up to the sweep-rate
    Nyquist. Stateless and reproducible: same (seed, t) -> same
    sample, no buffer carried in the dict. Spectrally-shaped variants
    (pink, brown, band-limited) are tracked separately in #77.
    """
    kind:      Literal["random"] = "random"
    amplitude: float = Field(..., ge=0)         # RMS strain (sigma)
    seed:      int   = Field(0, ge=0)           # for reproducibility


class StrainSegment(_Strict):
    start:   Length
    end:     Length
    # static offset -- the motion adds on top of this
    epsilon: float = 0.0
    motion:  HarmonicMotion | ThermalRelaxation | ImpulsivePulse | RandomVibration | None = None

    @model_validator(mode="after")
    def _check_order(self):
        if self.end <= self.start:
            raise ValueError(f"strain segment: end ({self.end}) must be > start ({self.start})")
        return self


class CoxParams(_Strict):
    beta: float = Field(..., gt=0)   # 1/m


class StrainConfig(_Strict):
    segments: list[StrainSegment] = Field(default_factory=list)
    photoelastic_coefficient: float = Field(0.22, ge=0, lt=1)
    transfer: Literal["ideal", "cox"] = "ideal"
    cox: CoxParams | None = None

    @model_validator(mode="after")
    def _cox_needs_params(self):
        if self.transfer == "cox" and self.cox is None:
            raise ValueError("strain.transfer=cox requires a strain.cox block with beta")
        return self


class TemperatureSegment(_Strict):
    """Piecewise-constant temperature change along the fiber. Motion (if
    given) modulates dT(t) -- same registry as strain segments, just
    interpreted as Kelvin instead of strain."""
    start:   Length
    end:     Length
    delta_T: float = 0.0   # static offset [K], signed
    motion:  HarmonicMotion | ThermalRelaxation | ImpulsivePulse | RandomVibration | None = None

    @model_validator(mode="after")
    def _check_order(self):
        if self.end <= self.start:
            raise ValueError(f"temperature segment: end ({self.end}) must be > start ({self.start})")
        return self


class TemperatureConfig(_Strict):
    """Distributed temperature perturbation. Defaults are silica @ 1550 nm.
    Combined with StrainConfig this gives the Froggatt-Moore strain-T
    cross-sensitivity (#75)."""
    segments: list[TemperatureSegment] = Field(default_factory=list)
    thermal_expansion: float = Field(5.5e-7, ge=0)   # alpha_L [1/K]
    # xi = (1/n) dn/dT [1/K]; user with raw dn/dT specifies dn_dT/n_core
    thermo_optic:      float = Field(6.5e-6, ge=0)


class CirculatorConfig(_Strict):
    insertion_loss_dB: float = Field(0.7, ge=0)
    isolation_dB:      float = Field(50.0, ge=0)
    return_loss_dB:    float = Field(55.0, ge=0)


class AuxMZIConfig(_Strict):
    # auxiliary (k-clock) Mach-Zehnder -- a short interferometer with a known
    # delay that tracks the instantaneous optical frequency. Its phase is used
    # to resample the main beat onto a uniform-nu grid (see analysis/demodulation).
    # off by default so existing configs keep working.
    enabled: bool = False
    delay:   Time = Field(0.0, ge=0)   # round-trip delay [s], typical 10-500 ns


class OpticsConfig(_Strict):
    splitting_ratio: float = Field(0.5, gt=0, lt=1)
    circulator: CirculatorConfig = Field(default_factory=CirculatorConfig)
    aux_mzi: AuxMZIConfig = Field(default_factory=AuxMZIConfig)

    @model_validator(mode="after")
    def _aux_needs_delay(self):
        if self.aux_mzi.enabled and self.aux_mzi.delay <= 0:
            raise ValueError("optics.aux_mzi.enabled=true requires a positive delay")
        return self


class DetectionConfig(_Strict):
    responsivity: float     = Field(1.0, gt=0)
    bandwidth:    Frequency = Field(1.0e8, gt=0)
    filter_order: int       = Field(4, ge=1)
    shot_noise:   bool      = True
    thermal_nep:  float     = Field(1.0e-11, ge=0)   # W/sqrt(Hz), bare float
    dark_current: Current   = Field(1.0e-9, ge=0)
    balanced:     bool      = False
    saturation_current: Current | None = None   # photodiode clamp [A]
    # polynomial distortion applied pre-TIA: I_out = I_in + a2*I_in^2 + a3*I_in^3 + ...
    # coefficients start at order 2 (linear term is implicit identity). Empty = ideal.
    nonlinearity_coefficients: list[float] = Field(default_factory=list)


class ADCConfig(_Strict):
    bits:            int          = Field(16, ge=1)
    enob:            float | None = Field(None, gt=0)
    sample_rate:     Frequency    = Field(2.0e8, gt=0)
    voltage_range:   Voltage      = Field(2.0,   gt=0)
    input_impedance: Resistance   = Field(50.0,  gt=0)
    jitter_rms:      Time         = Field(0.0, ge=0)   # aperture jitter [s]
    dnl_rms_lsb:     float        = Field(0.0, ge=0)   # DNL per-code rms [LSB]
    inl_peak_lsb:    float        = Field(0.0, ge=0)   # INL peak (sinusoidal) [LSB]

    @model_validator(mode="after")
    def _enob_within_bits(self):
        if self.enob is not None and self.enob > self.bits:
            raise ValueError(f"adc.enob ({self.enob}) must be <= adc.bits ({self.bits})")
        return self


class OutputConfig(_Strict):
    path: str | None = None   # None = no file output


class RootConfig(_Strict):
    simulation:  SimulationConfig  = Field(default_factory=SimulationConfig)
    fiber:       FiberConfig       = Field(default_factory=FiberConfig)
    source:      SourceConfig      = Field(default_factory=SourceConfig)
    optics:      OpticsConfig      = Field(default_factory=OpticsConfig)
    strain:      StrainConfig      = Field(default_factory=StrainConfig)
    temperature: TemperatureConfig = Field(default_factory=TemperatureConfig)
    detection:   DetectionConfig   = Field(default_factory=DetectionConfig)
    adc:         ADCConfig         = Field(default_factory=ADCConfig)
    output:      OutputConfig      = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def _check_nyquist(self):
        wl   = self.source.center_wavelength
        dwl  = self.source.sweep_range
        T    = self.source.sweep_duration
        L    = self.fiber.length
        n    = self.fiber.n_core
        fs   = self.adc.sample_rate

        sweep_hz   = _C / wl**2 * dwl
        gamma      = sweep_hz / T
        f_beat_max = 2.0 * n * gamma * L / _C
        f_nyq      = fs / 2.0
        if f_beat_max >= f_nyq:
            raise ValueError(
                f"max beat freq {f_beat_max*1e-6:.1f} MHz "
                f"exceeds Nyquist {f_nyq*1e-6:.1f} MHz"
            )
        return self
