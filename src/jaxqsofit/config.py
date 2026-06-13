from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass
class Observation:
    """Observation-level metadata for one quasar spectrum."""

    object_id: str = "result"
    redshift: float = 0.0
    ra: float | None = None
    dec: float | None = None
    apply_mw_deredden: bool = True


@dataclass
class SpectroscopyData:
    """Observed spectral measurements on an observed-frame wavelength grid."""

    wave_obs: Sequence[float]
    fluxes: Sequence[float]
    errors: Sequence[float] | float | None = None
    wavelength_dispersion: Sequence[float] | None = None
    mask: Sequence[bool] | None = None

    def validate(self) -> None:
        n = len(self.wave_obs)
        if len(self.fluxes) != n:
            raise ValueError("Spectroscopy fluxes must have the same length as wave_obs.")
        if self.errors is not None and not np.isscalar(self.errors) and len(self.errors) != n:
            raise ValueError("Spectroscopy errors must be scalar, None, or match wave_obs length.")
        if self.wavelength_dispersion is not None and len(self.wavelength_dispersion) != n:
            raise ValueError("wavelength_dispersion must match wave_obs length.")
        if self.mask is not None and len(self.mask) != n:
            raise ValueError("spectroscopy mask must match wave_obs length.")


@dataclass
class PSFPhotometryData:
    """Optional PSF-aperture photometry used for spectral recalibration.

    JAXQSOFit is a spectral fitter, so these data are only used as an extra
    calibration constraint on the fitted spectrum. Use bands whose transmission
    curves overlap the observed spectral wavelength coverage. For full joint
    spectrum + broadband SED modeling, use ``jaxsedfit`` instead.
    """

    magnitudes: Sequence[float]
    magnitude_errors: Sequence[float]
    filter_names: Sequence[str] = ("u", "g", "r", "i", "z")

    def validate(self) -> None:
        n = len(self.magnitudes)
        if len(self.magnitude_errors) != n or len(self.filter_names) != n:
            raise ValueError("PSF magnitudes, errors, and filter_names must have the same length.")


@dataclass
class PreprocessingConfig:
    """Spectrum preprocessing options applied before fitting."""

    wave_range: tuple[float, float] | None = None
    wave_mask: Sequence[Sequence[float]] | None = None
    mask_lya_forest: bool = True


@dataclass
class ContinuumConfig:
    """Continuum and spectral component switches."""

    fit_power_law: bool = True
    fit_feii: bool = True
    fit_balmer_continuum: bool = False
    fit_bal_absorption: bool = False
    fit_polynomial_tilt: bool = True
    fit_reddening: bool = True
    polynomial_order: int = 2


@dataclass
class HostConfig:
    """Host-galaxy spectral decomposition configuration."""

    enabled: bool = True
    sfh_model: str | None = None
    dsps_ssp_fn: str = "tempdata.h5"
    age_grid_gyr: Sequence[float] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    logzsol_grid: Sequence[float] = (-1.0, -0.5, 0.0, 0.2)


@dataclass
class LineConfig:
    """Emission-line model configuration."""

    enabled: bool = True
    custom_components: Sequence[Any] | None = None
    custom_line_components: Sequence[Any] | None = None


@dataclass
class InferenceConfig:
    """Inference defaults for Optax and NUTS."""

    method: str = "optax+nuts"
    map_steps: int = 600
    learning_rate: float = 1.0e-2
    num_warmup: int = 50
    num_samples: int = 50
    num_chains: int = 1
    target_accept_prob: float = 0.9
    plot_init: bool = False


@dataclass
class OutputConfig:
    """Plotting and persistence defaults."""

    output_path: str | None = None
    save_name: str | None = None
    save_result: bool = True
    plot_fig: bool = True
    save_fig: bool = True
    show_plot: bool = False


@dataclass
class FitConfig:
    """Top-level configuration bundle for one JAXQSOFit spectral fit."""

    observation: Observation
    spectroscopy: SpectroscopyData
    psf_photometry: PSFPhotometryData | None = None
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    continuum: ContinuumConfig = field(default_factory=ContinuumConfig)
    host: HostConfig = field(default_factory=HostConfig)
    lines: LineConfig = field(default_factory=LineConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    prior_config: dict[str, Any] | None = None

    def validate(self) -> None:
        self.spectroscopy.validate()
        if self.psf_photometry is not None:
            self.psf_photometry.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_dataclass(cls, value: Any):
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in value:
                kwargs[field_name] = value[field_name]
        return cls(**kwargs)
    raise TypeError(f"Cannot coerce {type(value)!r} to {cls.__name__}")


def fit_config_from_mapping(data: Mapping[str, Any]) -> FitConfig:
    """Build a validated FitConfig from a nested mapping."""

    psf_raw = data.get("psf_photometry")
    psf_obj = None if psf_raw is None else _coerce_dataclass(PSFPhotometryData, psf_raw)
    cfg = FitConfig(
        observation=_coerce_dataclass(Observation, data.get("observation", {})),
        spectroscopy=_coerce_dataclass(SpectroscopyData, data["spectroscopy"]),
        psf_photometry=psf_obj,
        preprocessing=_coerce_dataclass(PreprocessingConfig, data.get("preprocessing", {})),
        continuum=_coerce_dataclass(ContinuumConfig, data.get("continuum", {})),
        host=_coerce_dataclass(HostConfig, data.get("host", {})),
        lines=_coerce_dataclass(LineConfig, data.get("lines", {})),
        inference=_coerce_dataclass(InferenceConfig, data.get("inference", {})),
        output=_coerce_dataclass(OutputConfig, data.get("output", {})),
        prior_config=None if data.get("prior_config") is None else dict(data.get("prior_config", {})),
    )
    cfg.validate()
    return cfg


def serialize_config(value: Any) -> Any:
    """Convert config-like objects into JSON-serializable Python values."""

    if is_dataclass(value):
        return {k: serialize_config(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: serialize_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_config(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
