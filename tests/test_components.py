import numpy as np
import jax
from numpyro.handlers import seed, trace

from jaxqsofit.components import SpectralComponentConfig, evaluate_joint_spectral_components


def test_evaluate_joint_spectral_components_uses_external_continuum():
    wave_obs = np.linspace(4500.0, 7500.0, 64)
    continuum = np.full_like(wave_obs, 2.0)

    tr = trace(seed(evaluate_joint_spectral_components, jax.random.PRNGKey(3))).get_trace(
        wave_obs=wave_obs,
        redshift=0.1,
        continuum_mjy=continuum,
        config=SpectralComponentConfig(
            use_lines=False,
            use_feii=False,
            use_balmer_continuum=False,
            use_multiplicative_tilt=False,
        ),
    )

    assert np.allclose(np.asarray(tr["jqf_total_model"]["value"]), continuum)
    assert np.allclose(np.asarray(tr["jqf_line_model"]["value"]), 0.0)


def test_evaluate_joint_spectral_components_adds_line_sites():
    wave_obs = np.linspace(4500.0, 7500.0, 64)
    continuum = np.full_like(wave_obs, 2.0)

    tr = trace(seed(evaluate_joint_spectral_components, jax.random.PRNGKey(4))).get_trace(
        wave_obs=wave_obs,
        redshift=0.1,
        continuum_mjy=continuum,
        config=SpectralComponentConfig(
            use_lines=True,
            use_feii=False,
            use_balmer_continuum=False,
            line_centers_rest=(4861.33,),
            line_names=("Hbeta",),
            broad_line_names=("Hbeta",),
        ),
    )

    assert "jqf_line_amp_Hbeta" in tr
    assert "jqf_line_fwhm_Hbeta" in tr
    assert "jqf_line_velocity_Hbeta" in tr
    assert np.asarray(tr["jqf_total_model"]["value"]).shape == wave_obs.shape


def test_evaluate_joint_spectral_components_uses_default_tied_lines():
    wave_obs = np.linspace(4700.0, 5100.0, 96)
    continuum = np.full_like(wave_obs, 2.0)

    tr = trace(seed(evaluate_joint_spectral_components, jax.random.PRNGKey(5))).get_trace(
        wave_obs=wave_obs,
        redshift=0.0,
        continuum_mjy=continuum,
        config=SpectralComponentConfig(
            use_lines=True,
            use_tied_lines=True,
            use_feii=False,
            use_balmer_continuum=False,
            line_flux_scale_mjy=2.0,
        ),
    )

    assert "jqf_line_dmu_group" in tr
    assert "jqf_line_sig_group" in tr
    assert "jqf_line_amp_group" in tr
    assert "jqf_line_amp_per_component" in tr
    assert "jqf_line_model_broad" in tr
    assert "jqf_line_model_narrow" in tr
    assert np.asarray(tr["jqf_total_model"]["value"]).shape == wave_obs.shape


def test_evaluate_joint_spectral_components_reports_fixed_narrow_line_controls():
    wave_obs = np.linspace(4990.0, 5010.0, 96)
    continuum = np.full_like(wave_obs, 1.0)

    tr = trace(seed(evaluate_joint_spectral_components, jax.random.PRNGKey(6))).get_trace(
        wave_obs=wave_obs,
        redshift=0.0,
        continuum_mjy=continuum,
        config=SpectralComponentConfig(
            use_lines=True,
            use_tied_lines=False,
            use_feii=False,
            use_balmer_continuum=False,
            line_centers_rest=(5000.0,),
            line_names=("OIII5007c",),
            fixed_narrow_fwhm_kms=321.0,
            fixed_narrow_amp_scale=2.5,
        ),
    )

    assert tr["jqf_line_narrow_fwhm_kms"]["value"] == 321.0
    assert tr["jqf_line_narrow_amp_scale"]["value"] == 2.5
    assert np.nanmax(np.asarray(tr["jqf_line_model_narrow"]["value"])) > 0.0
