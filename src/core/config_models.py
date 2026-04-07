"""Pydantic models for the YAML config. See #10."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SimulationConfig(_Strict):
    seed: int = 42
    backend: Literal["numpy", "cupy", "jax"] = "numpy"


class FiberConfig(_Strict):
    length: float = Field(10.0, gt=0)
    n_core: float = Field(1.4682, gt=1.0)
    rayleigh_coefficient_dB: float = -82.0
    attenuation_dB_per_km: float = Field(0.0, ge=0)


class SourceConfig(_Strict):
    center_wavelength: float = Field(1550e-9, gt=0)
    sweep_range:       float = Field(40e-9,   gt=0)
    sweep_duration:    float = Field(0.01,    gt=0)
    power:             float = Field(10e-3,   gt=0)


class OpticsConfig(_Strict):
    splitting_ratio: float = Field(0.5, gt=0, lt=1)


class DetectionConfig(_Strict):
    responsivity: float = Field(1.0, gt=0)
    bandwidth:    float = Field(1.0e8, gt=0)
    filter_order: int   = Field(4, ge=1)
    shot_noise:   bool  = True
    thermal_nep:  float = Field(1.0e-11, ge=0)
    dark_current: float = Field(1.0e-9, ge=0)


class ADCConfig(_Strict):
    bits:            int   = Field(16, ge=1)
    sample_rate:     float = Field(2.0e8, gt=0)
    voltage_range:   float = Field(2.0,   gt=0)
    input_impedance: float = Field(50.0,  gt=0)


class RootConfig(_Strict):
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    fiber:      FiberConfig      = Field(default_factory=FiberConfig)
    source:     SourceConfig     = Field(default_factory=SourceConfig)
    optics:     OpticsConfig     = Field(default_factory=OpticsConfig)
    detection:  DetectionConfig  = Field(default_factory=DetectionConfig)
    adc:        ADCConfig        = Field(default_factory=ADCConfig)
