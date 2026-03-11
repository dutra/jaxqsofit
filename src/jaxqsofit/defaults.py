from __future__ import annotations

import copy
from typing import Any, Dict, List

import numpy as np

# Default line table in plain dict rows (same schema as notebook line config).
DEFAULT_LINE_PRIOR_ROWS: List[Dict[str, Any]] = [
    # Halpha complex
    {'lambda': 6564.61, 'compname': 'Ha', 'minwav': 6400, 'maxwav': 6800, 'linename': 'Ha_br', 'ngauss': 2, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 5e-3, 'minsig': 0.004, 'maxsig': 0.05, 'voff': 0.015, 'vindex': 0, 'windex': 0, 'findex': 0, 'fvalue': 0.05, 'vary': 1},
    {'lambda': 6564.61, 'compname': 'Ha', 'minwav': 6400, 'maxwav': 6800, 'linename': 'Ha_na', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 5e-4, 'maxsig': 0.00169, 'voff': 0.01, 'vindex': 1, 'windex': 1, 'findex': 0, 'fvalue': 0.002, 'vary': 1},
    {'lambda': 6549.85, 'compname': 'Ha', 'minwav': 6400, 'maxwav': 6800, 'linename': 'NII6549', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 5e-3, 'vindex': 1, 'windex': 1, 'findex': 1, 'fvalue': 0.001, 'vary': 1},
    {'lambda': 6585.28, 'compname': 'Ha', 'minwav': 6400, 'maxwav': 6800, 'linename': 'NII6585', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 5e-3, 'vindex': 1, 'windex': 1, 'findex': 1, 'fvalue': 0.003, 'vary': 1},
    {'lambda': 6718.29, 'compname': 'Ha', 'minwav': 6400, 'maxwav': 6800, 'linename': 'SII6718', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 5e-3, 'vindex': 1, 'windex': 1, 'findex': 2, 'fvalue': 0.001, 'vary': 1},
    {'lambda': 6732.67, 'compname': 'Ha', 'minwav': 6400, 'maxwav': 6800, 'linename': 'SII6732', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 5e-3, 'vindex': 1, 'windex': 1, 'findex': 2, 'fvalue': 0.001, 'vary': 1},
    # Hbeta / [OIII]
    {'lambda': 4862.68, 'compname': 'Hb', 'minwav': 4640, 'maxwav': 5100, 'linename': 'Hb_br', 'ngauss': 2, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 5e-3, 'minsig': 0.004, 'maxsig': 0.05, 'voff': 0.01, 'vindex': 0, 'windex': 0, 'findex': 0, 'fvalue': 0.01, 'vary': 1},
    {'lambda': 4862.68, 'compname': 'Hb', 'minwav': 4640, 'maxwav': 5100, 'linename': 'Hb_na', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.01, 'vindex': 1, 'windex': 1, 'findex': 0, 'fvalue': 0.002, 'vary': 1},
    {'lambda': 4960.30, 'compname': 'Hb', 'minwav': 4640, 'maxwav': 5100, 'linename': 'OIII4959c', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.01, 'vindex': 1, 'windex': 1, 'findex': 0, 'fvalue': 0.002, 'vary': 1},
    {'lambda': 5008.24, 'compname': 'Hb', 'minwav': 4640, 'maxwav': 5100, 'linename': 'OIII5007c', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.01, 'vindex': 1, 'windex': 1, 'findex': 0, 'fvalue': 0.004, 'vary': 1},
    {'lambda': 4960.30, 'compname': 'Hb', 'minwav': 4640, 'maxwav': 5100, 'linename': 'OIII4959w', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 3e-3, 'minsig': 2.3e-4, 'maxsig': 0.004, 'voff': 0.01, 'vindex': 2, 'windex': 2, 'findex': 0, 'fvalue': 0.001, 'vary': 1},
    {'lambda': 5008.24, 'compname': 'Hb', 'minwav': 4640, 'maxwav': 5100, 'linename': 'OIII5007w', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 3e-3, 'minsig': 2.3e-4, 'maxsig': 0.004, 'voff': 0.01, 'vindex': 2, 'windex': 2, 'findex': 0, 'fvalue': 0.002, 'vary': 1},
    # Higher-order Balmer
    {'lambda': 4341.68, 'compname': 'Hg', 'minwav': 4200, 'maxwav': 4400, 'linename': 'Hg_br', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 5e-3, 'minsig': 0.004, 'maxsig': 0.05, 'voff': 0.01, 'vindex': 0, 'windex': 0, 'findex': 0, 'fvalue': 0.01, 'vary': 1},
    {'lambda': 4341.68, 'compname': 'Hg', 'minwav': 4200, 'maxwav': 4400, 'linename': 'Hg_na', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.01, 'vindex': 1, 'windex': 1, 'findex': 0, 'fvalue': 0.002, 'vary': 1},
    {'lambda': 4102.89, 'compname': 'Hd', 'minwav': 4000, 'maxwav': 4150, 'linename': 'Hd_br', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 5e-3, 'minsig': 0.004, 'maxsig': 0.05, 'voff': 0.01, 'vindex': 0, 'windex': 0, 'findex': 0, 'fvalue': 0.01, 'vary': 1},
    {'lambda': 4102.89, 'compname': 'Hd', 'minwav': 4000, 'maxwav': 4150, 'linename': 'Hd_na', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.01, 'vindex': 1, 'windex': 1, 'findex': 0, 'fvalue': 0.002, 'vary': 1},
    # Other optical/UV
    {'lambda': 3728.48, 'compname': 'OII', 'minwav': 3650, 'maxwav': 3800, 'linename': 'OII3728', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 3.333e-4, 'maxsig': 0.00169, 'voff': 0.01, 'vindex': 1, 'windex': 1, 'findex': 0, 'fvalue': 0.001, 'vary': 1},
    {'lambda': 3426.84, 'compname': 'NeV', 'minwav': 3380, 'maxwav': 3480, 'linename': 'NeV3426', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 3.333e-4, 'maxsig': 0.00169, 'voff': 0.01, 'vindex': 0, 'windex': 0, 'findex': 0, 'fvalue': 0.001, 'vary': 1},
    {'lambda': 2798.75, 'compname': 'MgII', 'minwav': 2700, 'maxwav': 2900, 'linename': 'MgII_br', 'ngauss': 2, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 5e-3, 'minsig': 0.004, 'maxsig': 0.05, 'voff': 0.015, 'vindex': 0, 'windex': 0, 'findex': 0, 'fvalue': 0.05, 'vary': 1},
    {'lambda': 2798.75, 'compname': 'MgII', 'minwav': 2700, 'maxwav': 2900, 'linename': 'MgII_na', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 5e-4, 'maxsig': 0.00169, 'voff': 0.01, 'vindex': 1, 'windex': 1, 'findex': 0, 'fvalue': 0.002, 'vary': 1},
    {'lambda': 1908.73, 'compname': 'CIII', 'minwav': 1700, 'maxwav': 1970, 'linename': 'CIII_br', 'ngauss': 2, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 5e-3, 'minsig': 0.004, 'maxsig': 0.05, 'voff': 0.015, 'vindex': 99, 'windex': 0, 'findex': 0, 'fvalue': 0.01, 'vary': 1},
    {'lambda': 1549.06, 'compname': 'CIV', 'minwav': 1500, 'maxwav': 1700, 'linename': 'CIV_br', 'ngauss': 2, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 5e-3, 'minsig': 0.004, 'maxsig': 0.05, 'voff': 0.015, 'vindex': 0, 'windex': 0, 'findex': 0, 'fvalue': 0.05, 'vary': 1},
    {'lambda': 1402.06, 'compname': 'SiIV', 'minwav': 1290, 'maxwav': 1450, 'linename': 'SiIV_OIV1', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 5e-3, 'minsig': 0.002, 'maxsig': 0.05, 'voff': 0.015, 'vindex': 1, 'windex': 1, 'findex': 0, 'fvalue': 0.05, 'vary': 1},
    {'lambda': 1215.67, 'compname': 'Lya', 'minwav': 1150, 'maxwav': 1290, 'linename': 'Lya_br', 'ngauss': 3, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 5e-3, 'minsig': 0.002, 'maxsig': 0.05, 'voff': 0.02, 'vindex': 0, 'windex': 0, 'findex': 0, 'fvalue': 0.05, 'vary': 1},
    {'lambda': 1240.14, 'compname': 'Lya', 'minwav': 1150, 'maxwav': 1290, 'linename': 'NV1240', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 2e-3, 'minsig': 0.001, 'maxsig': 0.01, 'voff': 0.005, 'vindex': 0, 'windex': 0, 'findex': 0, 'fvalue': 0.002, 'vary': 1},
]

DEFAULT_LINE_CONFIG: Dict[str, Any] = {
    "line_dmu_scale_mult": 0.25,
    "line_sig_scale_mult": 0.25,
    "line_amp_scale_mult": 0.25,
    "line": {"table": DEFAULT_LINE_PRIOR_ROWS},
}


def build_default_prior_config(flux: np.ndarray, line_config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Build a full prior_config with sane defaults from data flux scale."""
    f = np.asarray(flux, dtype=float)
    finite = np.isfinite(f)
    fscale = float(np.nanmedian(np.abs(f[finite]))) if np.any(finite) else 1.0
    if not np.isfinite(fscale) or fscale <= 0:
        fscale = 1.0

    cfg: Dict[str, Any] = {
        "log_cont_norm": {"loc": np.log(max(fscale, 1e-8)), "scale": 0.3},
        "PL_slope": {"loc": -1.5, "scale": 0.4, "low": -3.5, "high": 0.3},
        "log_frac_host": {"loc": 0.0, "scale": 1.0},
        "tau_host": {"scale": 1.0},
        "raw_w": {"loc": -0.5, "scale": 1.0},
        "gal_v_kms": {"loc": 0.0, "scale": 120.0},
        "gal_sigma_kms": {"scale": 200.0},
        "log_Fe_uv_norm": {"loc": np.log(max(1e-3 * fscale, 1e-10)), "scale": 0.5},
        "log_Fe_op_norm": {"loc": np.log(max(1e-3 * fscale, 1e-10)), "scale": 0.5},
        "log_Fe_uv_FWHM": {"loc": np.log(3000.0), "scale": 0.3},
        "log_Fe_op_FWHM": {"loc": np.log(3000.0), "scale": 0.3},
        "Fe_uv_shift": {"loc": 0.0, "scale": 1e-3},
        "Fe_op_shift": {"loc": 0.0, "scale": 1e-3},
        "log_Balmer_norm": {"loc": np.log(max(1e-3 * fscale, 1e-10)), "scale": 0.5},
        "log_Balmer_Tau": {"loc": np.log(0.5), "scale": 0.25},
        "log_Balmer_vel": {"loc": np.log(3000.0), "scale": 0.25},
        "poly_c1": {"loc": 0.0, "scale": 0.1},
        "poly_c2": {"loc": 0.0, "scale": 0.1},
        "frac_jitter": {"scale": 0.02},
        "add_jitter": {"scale_mult_err": 0.3},
        "student_t_df": 3.0,
    }

    lc = copy.deepcopy(DEFAULT_LINE_CONFIG if line_config is None else line_config)
    cfg.update(lc)
    return cfg

