import numpy as np
import pytest

from jaxqsofit import QSOFit, build_default_prior_config


def _make_simple_spectrum(n=64):
    lam = np.linspace(3800.0, 9200.0, n)
    flux = 50.0 + 0.002 * (lam - 6000.0)
    err = np.full_like(flux, 0.5)
    return lam, flux, err


def test_init_err_optional_defaults_to_small_value():
    lam, flux, _ = _make_simple_spectrum()
    q = QSOFit(lam=lam, flux=flux, z=0.1)

    assert q.err_in.shape == flux.shape
    assert np.allclose(q.err_in, 1e-6)


def test_fit_dispatch_nuts(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = QSOFit(lam=lam, flux=flux, err=err, z=0.1)

    called = {'nuts': 0}

    def _stub_nuts(**kwargs):
        called['nuts'] += 1

    monkeypatch.setattr(q, 'run_fsps_numpyro_fit', _stub_nuts)

    q.Fit(
        deredden=False,
        fit_method='nuts',
        plot_fig=False,
        save_result=False,
        prior_config=build_default_prior_config(flux),
    )

    assert called['nuts'] == 1


def test_fit_dispatch_optax(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = QSOFit(lam=lam, flux=flux, err=err, z=0.1)

    called = {'optax': 0}

    def _stub_optax(**kwargs):
        called['optax'] += 1

    monkeypatch.setattr(q, 'run_fsps_optax_fit', _stub_optax)

    q.Fit(
        deredden=False,
        fit_method='optax',
        plot_fig=False,
        save_result=False,
        prior_config=build_default_prior_config(flux),
    )

    assert called['optax'] == 1


def test_fit_dispatch_optax_nuts(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = QSOFit(lam=lam, flux=flux, err=err, z=0.1)

    called = {'optax_nuts': 0}

    def _stub_optax_nuts(**kwargs):
        called['optax_nuts'] += 1

    monkeypatch.setattr(q, 'run_fsps_optax_nuts_fit', _stub_optax_nuts)

    q.Fit(
        deredden=False,
        fit_method='optax+nuts',
        plot_fig=False,
        save_result=False,
        prior_config=build_default_prior_config(flux),
    )

    assert called['optax_nuts'] == 1


def test_fit_method_unknown_raises():
    lam, flux, err = _make_simple_spectrum()
    q = QSOFit(lam=lam, flux=flux, err=err, z=0.1)

    with pytest.raises(ValueError, match='Unknown fit_method'):
        q.Fit(
            deredden=False,
            fit_method='not-a-method',
            plot_fig=False,
            save_result=False,
            prior_config=build_default_prior_config(flux),
        )
