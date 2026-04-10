"""Pydantic models for the YAML config. See #10."""

from __future__ import annotations

from typing import Annotated, Literal

from pint import UnitRegistry
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from utils.constants import C as _C

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


class FiberConfig(_Strict):
    length: Length = Field(10.0, gt=0)
    n_core: float = Field(1.4682, gt=1.0)
    n_cores: int = Field(1, ge=1)
    rayleigh_coefficient_dB: float = -82.0
    attenuation_dB_per_km: float = Field(0.0, ge=0)


class SourceConfig(_Strict):
    center_wavelength: Length    = Field(1550e-9, gt=0)
    sweep_range:       Length    = Field(40e-9,   gt=0)
    sweep_duration:    Time      = Field(0.01,    gt=0)
    power:             Power     = Field(10e-3,   gt=0)
    linewidth:         Frequency = Field(0.0, ge=0)   # FWHM Lorentzian


class StrainSegment(_Strict):
    start:   Length
    end:     Length
    epsilon: float

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


class OpticsConfig(_Strict):
    splitting_ratio: float = Field(0.5, gt=0, lt=1)


class DetectionConfig(_Strict):
    responsivity: float     = Field(1.0, gt=0)
    bandwidth:    Frequency = Field(1.0e8, gt=0)
    filter_order: int       = Field(4, ge=1)
    shot_noise:   bool      = True
    thermal_nep:  float     = Field(1.0e-11, ge=0)   # W/sqrt(Hz), bare float
    dark_current: Current   = Field(1.0e-9, ge=0)
    balanced:     bool      = False


class ADCConfig(_Strict):
    bits:            int          = Field(16, ge=1)
    enob:            float | None = Field(None, gt=0)
    sample_rate:     Frequency    = Field(2.0e8, gt=0)
    voltage_range:   Voltage      = Field(2.0,   gt=0)
    input_impedance: Resistance   = Field(50.0,  gt=0)
    jitter_rms:      Time         = Field(0.0, ge=0)   # aperture jitter [s]

    @model_validator(mode="after")
    def _enob_within_bits(self):
        if self.enob is not None and self.enob > self.bits:
            raise ValueError(f"adc.enob ({self.enob}) must be <= adc.bits ({self.bits})")
        return self


class RootConfig(_Strict):
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    fiber:      FiberConfig      = Field(default_factory=FiberConfig)
    source:     SourceConfig     = Field(default_factory=SourceConfig)
    optics:     OpticsConfig     = Field(default_factory=OpticsConfig)
    strain:     StrainConfig     = Field(default_factory=StrainConfig)
    detection:  DetectionConfig  = Field(default_factory=DetectionConfig)
    adc:        ADCConfig        = Field(default_factory=ADCConfig)

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
