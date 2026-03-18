import numpy as np

from jaxqsofit.defaults import build_default_prior_config


def test_build_default_prior_config_has_expected_keys():
    flux = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    cfg = build_default_prior_config(flux)

    required = [
        'log_cont_norm',
        'PL_slope',
        'PL_pivot',
        'log_frac_host',
        'tau_host',
        'raw_w',
        'line_dmu_scale_mult',
        'line_sig_scale_mult',
        'line_amp_scale_mult',
        'line',
        'student_t_df',
    ]
    for k in required:
        assert k in cfg


def test_build_default_prior_config_scales_with_flux_median():
    flux = np.array([10.0, 20.0, 30.0], dtype=float)
    cfg = build_default_prior_config(flux)

    expected = np.log(np.median(np.abs(flux)))
    got = float(cfg['log_cont_norm']['loc'])
    assert np.isfinite(got)
    assert np.isclose(got, expected)


def test_build_default_prior_config_accepts_manual_pl_pivot():
    flux = np.array([1.0, 2.0, 3.0], dtype=float)
    cfg = build_default_prior_config(flux, pl_pivot=3000.0)
    assert cfg["PL_pivot"] == 3000.0
