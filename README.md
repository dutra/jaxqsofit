# JaxQSOFit

<p align="center">
  <img src="logo.png" alt="JaxQSOFit logo" width="220">
</p>

Bayesian quasar spectral fitting with JAX + NumPyro, including:

- AGN continuum (power law)
- Host galaxy decomposition from DSPS SSP templates
- FeII UV/optical templates
- Balmer continuum
- Tied Gaussian emission-line model
- Student-t likelihood (robust to outliers/absorption)

The main implementation is in [`src/jaxqsofit/core.py`](./src/jaxqsofit/core.py).

## Key Differences vs PyQSOFit

JaxQSOFit is designed around a **joint Bayesian model** of AGN and host components, rather than fitting them in separate stages.

- **Joint host+AGN inference**:
  Host galaxy SPS (DSPS SSP mixture + LOSVD) is fit simultaneously with AGN continuum and lines.
- **More robust for complex spectra**:
  This joint treatment reduces AGN/host degeneracies and yields more stable parameters when spectra have strong blending, absorption, or mixed host contamination.
- **Probabilistic uncertainties**:
  NumPyro/NUTS produces posterior distributions for continuum, host, Fe, Balmer, and line parameters in one consistent model.

## Requirements

- Python 3.10+
- `jax`, `jaxlib`
- `numpyro`
- `numpy`, `scipy`, `matplotlib`, `pandas`
- `astropy`
- `extinction`
- `dsps`
- `dustmaps`

## Install

### 1. Create/activate environment

```bash
conda create -n jaxqsofit python=3.12 -y
conda activate jaxqsofit
```

### 2. Install dependencies

CPU example:

```bash
pip install -U pip
pip install numpy scipy matplotlib pandas astropy extinction dustmaps dsps numpyro
pip install "jax[cpu]"
```

If you want GPU JAX, install `jax`/`jaxlib` following official JAX instructions for your CUDA setup.

### 3. Provide DSPS SSP templates

This code expects an HDF5 SSP file path passed as `dsps_ssp_fn`, e.g. `tempdata.h5`.

Current default in code/notebook examples:

```python
dsps_ssp_fn="tempdata.h5"
```

### 4. Configure dustmaps SFD (one-time)

This repo assumes `dustmaps` is already configured and SFD maps are available.

Typical one-time setup:

```python setup.py fetch --map-name=sfd
```

After this, `SFDQuery()` should work without further setup.

## Repo layout

- `src/jaxqsofit/core.py` – fitting code
- `src/jaxqsofit/defaults.py` – default prior/line configs
- `src/jaxqsofit/__init__.py` – package exports
- `tests/` – test directory
- `test.ipynb` – development/example notebook
- `fe_uv.txt`, `fe_optical.txt` – Fe templates
- `tempdata.h5` – DSPS SSP template file (example)
- `data/spec-0332-52367-0639.csv` – example spectrum (`loglam`, `flux`, `ivar`)
- `data/spec-0332-52367-0639-meta.csv` – example metadata (`z`, `ra`, `dec`, etc.)

## Minimal usage

```python
import numpy as np
import pandas as pd
import jaxqsofit

# Load example spectrum
spec = pd.read_csv("data/spec-0332-52367-0639.csv")
meta = pd.read_csv("data/spec-0332-52367-0639-meta.csv").iloc[0]

lam = 10 ** spec["loglam"].to_numpy()
flux = spec["flux"].to_numpy()
err = 1.0 / np.sqrt(spec["ivar"].to_numpy())

q = jaxqsofit.QSOFit(
    lam, flux, err, z=float(meta["z"]),
    ra=float(meta["ra"]), dec=float(meta["dec"]),
    plateid=int(meta["plateid"]), mjd=int(meta["mjd"]), fiberid=int(meta["fiberid"]),
    path="."
)

# You must provide prior_config keys used by enabled model components.
prior_config = {
    "log_cont_norm": {"loc": np.log(np.nanmedian(np.abs(flux))), "scale": 0.3},
    "PL_slope": {"loc": -1.5, "scale": 0.4, "low": -3.5, "high": 0.3},

    # host
    "log_frac_host": {"loc": 0.0, "scale": 1.0},
    "tau_host": {"scale": 1.0},
    "raw_w": {"loc": -0.5, "scale": 1.0},
    "gal_v_kms": {"loc": 0.0, "scale": 120.0},
    "gal_sigma_kms": {"scale": 200.0},

    # Fe (required if fit_fe=True)
    "log_Fe_uv_norm": {"loc": np.log(1e-2), "scale": 0.5},
    "log_Fe_op_norm": {"loc": np.log(1e-2), "scale": 0.5},
    "log_Fe_uv_FWHM": {"loc": np.log(3000.0), "scale": 0.3},
    "log_Fe_op_FWHM": {"loc": np.log(3000.0), "scale": 0.3},
    "Fe_uv_shift": {"loc": 0.0, "scale": 1e-3},
    "Fe_op_shift": {"loc": 0.0, "scale": 1e-3},

    # Balmer continuum (required if fit_bc=True)
    "log_Balmer_norm": {"loc": np.log(1e-2), "scale": 0.5},
    "log_Balmer_Tau": {"loc": np.log(0.5), "scale": 0.25},
    "log_Balmer_vel": {"loc": np.log(3000.0), "scale": 0.25},

    # optional polynomial (required if fit_poly=True)
    "poly_c1": {"loc": 0.0, "scale": 0.1},
    "poly_c2": {"loc": 0.0, "scale": 0.1},

    # line model (required if fit_lines=True)
    "line_dmu_scale_mult": 0.25,
    "line_sig_scale_mult": 0.25,
    "line_amp_scale_mult": 0.25,
    "line": {"table": ...},  # structured line-prior array, see test.ipynb

    # noise
    "frac_jitter": {"scale": 0.02},
    "add_jitter": {"scale_mult_err": 0.3},

    # robust likelihood (optional; default 3.0)
    "student_t_df": 3.0,
}

q.Fit(
    deredden=True,
    fit_lines=True,
    decompose_host=True,
    fit_fe=False,
    fit_bc=True,
    fit_poly=False,
    prior_config=prior_config,
    dsps_ssp_fn="tempdata.h5",
    nuts_warmup=300,
    nuts_samples=600,
    nuts_chains=1,
    plot_fig=True,
    save_fig=False,
)
```

## Important API notes

- There are no hidden/default prior checks for required parameters. Missing keys raise normal Python/JAX errors.
- `fit_lines=True` requires a line prior table in:
  - `prior_config["line"]["table"]` (preferred), or
  - `prior_config["line_priors"]`, or
  - `prior_config["line_table"]`.
- `fit_fe=False`, `fit_bc=False`, `fit_poly=False`, `decompose_host=False` disable those model blocks.
- If `prior_config=None`, defaults are auto-built from `src/jaxqsofit/defaults.py` using the input flux scale.
- Likelihood is Student-t:
  - `prior_config["student_t_df"]` controls tail heaviness.
  - Lower `df` is more robust to outliers.

## Outputs on `QSOFit` object

Common fitted arrays:

- `wave`, `flux`, `err`
- `model_total`
- `f_conti_model`, `f_line_model`
- `f_pl_model`, `f_fe_mgii_model`, `f_fe_balmer_model`, `f_bc_model`
- `host`, `qso`

Posterior artifacts:

- `numpyro_mcmc`, `numpyro_samples`
- `pred_out`
- `_pred_host_draws`, `_pred_bc_draws`, `_pred_cont_draws`

Derived fractions:

- `frac_host_4200`, `frac_host_5100`, `frac_host_2500`
- `frac_bc_2500`

## Troubleshooting

- `SFDQuery` errors:
  - Ensure dustmaps data are downloaded and `config['data_dir']` is set.
- DSPS load errors:
  - Confirm `dsps_ssp_fn` points to a valid SSP HDF5 file.
- Line amplitudes explode:
  - Tighten line prior ranges (`maxsca`, `maxsig`) and scale multipliers.
- Fe fit degrades continuum:
  - Use stronger shrinkage priors on Fe norms and narrower Fe shift/FWHM priors.
