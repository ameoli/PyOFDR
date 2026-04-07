"""Pydantic models for the YAML config. See #10."""

from __future__ import annotations

from typing import Annotated, Literal

from pint import UnitRegistry
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

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
    rayleigh_coefficient_dB: float = -82.0
    attenuation_dB_per_km: float = Field(0.0, ge=0)


class SourceConfig(_Strict):
    center_wavelength: Length    = Field(1550e-9, gt=0)
    sweep_range:       Length    = Field(40e-9,   gt=0)
    sweep_duration:    Time      = Field(0.01,    gt=0)
    power:             Power     = Field(10e-3,   gt=0)


class OpticsConfig(_Strict):
    splitting_ratio: float = Field(0.5, gt=0, lt=1)


class DetectionConfig(_Strict):
    responsivity: float     = Field(1.0, gt=0)
    bandwidth:    Frequency = Field(1.0e8, gt=0)
    filter_order: int       = Field(4, ge=1)
    shot_noise:   bool      = True
    thermal_nep:  float     = Field(1.0e-11, ge=0)   # W/sqrt(Hz), bare float
    dark_current: Current   = Field(1.0e-9, ge=0)


class ADCConfig(_Strict):
    bits:            int        = Field(16, ge=1)
    sample_rate:     Frequency  = Field(2.0e8, gt=0)
    voltage_range:   Voltage    = Field(2.0,   gt=0)
    input_impedance: Resistance = Field(50.0,  gt=0)


class RootConfig(_Strict):
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    fiber:      FiberConfig      = Field(default_factory=FiberConfig)
    source:     SourceConfig     = Field(default_factory=SourceConfig)
    optics:     OpticsConfig     = Field(default_factory=OpticsConfig)
    detection:  DetectionConfig  = Field(default_factory=DetectionConfig)
    adc:        ADCConfig        = Field(default_factory=ADCConfig)
