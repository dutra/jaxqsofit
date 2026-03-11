import numpy as np

from jaxqsofit.model import _extract_line_table_from_prior_config, build_tied_line_meta_from_linelist


def test_extract_line_table_from_prior_config_layouts():
    table = [{'lambda': 5008.24, 'linename': 'OIII5007', 'compname': 'Hb', 'ngauss': 1, 'inisca': 1.0, 'minsca': 0.0, 'maxsca': 1e3, 'inisig': 1e-3, 'minsig': 1e-4, 'maxsig': 1e-2, 'voff': 0.01, 'vindex': 1, 'windex': 1, 'findex': 1, 'fvalue': 1.0}]

    cfg1 = {'line_priors': table}
    cfg2 = {'line_table': table}
    cfg3 = {'line': {'table': table}}
    cfg4 = {'line': {'priors': table}}

    assert _extract_line_table_from_prior_config(cfg1) is table
    assert _extract_line_table_from_prior_config(cfg2) is table
    assert _extract_line_table_from_prior_config(cfg3) is table
    assert _extract_line_table_from_prior_config(cfg4) is table


def test_build_tied_line_meta_from_linelist_minimal():
    line_table = [
        {
            'lambda': 5008.24,
            'linename': 'OIII5007',
            'compname': 'Hb',
            'ngauss': 1,
            'inisca': 1.0,
            'minsca': 0.0,
            'maxsca': 1e3,
            'inisig': 1e-3,
            'minsig': 1e-4,
            'maxsig': 1e-2,
            'voff': 0.01,
            'vindex': 1,
            'windex': 1,
            'findex': 1,
            'fvalue': 1.0,
        },
        {
            'lambda': 4960.30,
            'linename': 'OIII4959',
            'compname': 'Hb',
            'ngauss': 1,
            'inisca': 0.3,
            'minsca': 0.0,
            'maxsca': 1e3,
            'inisig': 1e-3,
            'minsig': 1e-4,
            'maxsig': 1e-2,
            'voff': 0.01,
            'vindex': 1,
            'windex': 1,
            'findex': 1,
            'fvalue': 0.33,
        },
    ]
    wave = np.linspace(4800.0, 5100.0, 200)

    meta = build_tied_line_meta_from_linelist(line_table, wave)

    assert meta['n_lines'] == 2
    assert meta['n_vgroups'] >= 1
    assert meta['n_wgroups'] >= 1
    assert meta['n_fgroups'] >= 1
    assert len(meta['names']) == 2
    assert np.all(np.isfinite(meta['line_lambda']))
