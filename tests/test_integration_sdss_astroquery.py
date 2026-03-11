import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_sdss_fetch_and_qsofit_init():
    astroquery = pytest.importorskip('astroquery.sdss')
    coordinates = pytest.importorskip('astropy.coordinates')
    units = pytest.importorskip('astropy.units')

    SDSS = astroquery.SDSS
    SkyCoord = coordinates.SkyCoord
    u = units

    ra, dec = 184.0307, -2.2383
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')

    try:
        xid = SDSS.query_region(coord, radius=5 * u.arcsec, spectro=True)
    except Exception as exc:
        pytest.skip(f'SDSS query unavailable: {exc}')

    if xid is None or len(xid) == 0:
        pytest.skip('No SDSS spectrum found near target coordinates')

    try:
        spectra = SDSS.get_spectra(matches=xid[:1])
    except Exception as exc:
        pytest.skip(f'SDSS spectrum download unavailable: {exc}')

    if not spectra:
        pytest.skip('No SDSS spectra returned')

    from jaxqsofit import QSOFit

    hdu = spectra[0]
    data = hdu[1].data
    lam = np.asarray(10 ** data['loglam'], dtype=float)
    flux = np.asarray(data['flux'], dtype=float)
    ivar = np.asarray(data['ivar'], dtype=float)

    err = np.full_like(flux, np.inf)
    m = ivar > 0
    err[m] = 1.0 / np.sqrt(ivar[m])
    err[~np.isfinite(err)] = 1e-6
    err[err <= 0] = 1e-6

    z = float(xid[0]['z']) if 'z' in xid.colnames else 0.1

    q = QSOFit(lam=lam, flux=flux, err=err, z=z, ra=ra, dec=dec)

    assert q.lam_in.size > 100
    assert np.isfinite(q.flux_in).any()
    assert np.isfinite(q.err_in).any()
