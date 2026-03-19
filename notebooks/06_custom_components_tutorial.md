# Custom Components Tutorial

`jaxqsofit` custom components are fully generic additive terms.

A component is defined by:

- a name
- a set of parameter priors
- a function `evaluate(wave, params, metadata)` that returns the component flux

The evaluator owns all physics: shifts, broadenings, interpolations, polynomials, or anything else.

```python
from jaxqsofit import (
    QSOFit,
    build_default_prior_config,
    make_custom_component,
    make_template_component,
)
```

## 1. Generic function component

This is the most flexible path and should be the default recommendation.

```python
import jax.numpy as jnp
import numpy as np


def gaussian_bump_component(wave, params, metadata):
    mu = metadata["mu"]
    sigma = metadata["sigma"]
    x = (wave - mu) / sigma
    return params["amp"] * jnp.exp(-0.5 * x**2)


custom_components = [
    make_custom_component(
        name="gaussian_bump",
        parameter_priors={
            "amp": {"dist": "LogNormal", "loc": np.log(0.02), "scale": 0.5},
        },
        evaluate=gaussian_bump_component,
        metadata={"mu": 2800.0, "sigma": 120.0},
    ),
]

prior_config = build_default_prior_config(flux)

q = QSOFit(lam=lam_obs, flux=flux, err=err, z=z)
q.fit(
    prior_config=prior_config,
    custom_components=custom_components,
)
```

After fitting:

```python
q.custom_components["gaussian_bump"]
```

and in posterior reconstruction:

```python
recon = q.reconstruct_posterior_spectrum(wave_min=2000.0)
bump_draws = recon["draws"]["gaussian_bump"]
```

## 2. User-controlled broadened template

If the user wants custom shift/broadening behavior, put it directly in the evaluator.

```python
import jax.numpy as jnp
import numpy as np

from jaxqsofit.model import _fe_template_component


template = np.loadtxt("my_feii_template.txt")
template_wave = template[:, 0]
template_flux = template[:, 1]


def my_feii_component(wave, params, metadata):
    return _fe_template_component(
        wave=wave,
        wave_template=metadata["wave_template"],
        flux_template=metadata["flux_template"],
        norm=params["norm"],
        fwhm_kms=params["fwhm_kms"],
        shift_frac=params["shift_frac"],
        base_fwhm_kms=metadata["base_fwhm_kms"],
    )


custom_components = [
    make_custom_component(
        name="alt_feii",
        parameter_priors={
            "norm": {"dist": "LogNormal", "loc": np.log(0.02), "scale": 0.6},
            "fwhm_kms": {"dist": "LogNormal", "loc": np.log(2500.0), "scale": 0.3},
            "shift_frac": {"dist": "Normal", "loc": 0.0, "scale": 1e-3},
        },
        evaluate=my_feii_component,
        metadata={
            "wave_template": template_wave,
            "flux_template": template_flux,
            "base_fwhm_kms": 900.0,
        },
    ),
]

q = QSOFit(lam=lam_obs, flux=flux, err=err, z=z)
q.fit(
    prior_config=build_default_prior_config(flux),
    custom_components=custom_components,
    fit_fe=False,
)
```

This disables the bundled Fe model and replaces it with a user-defined additive component.

## 3. Additive polynomial example

If you want an additive polynomial, just write it as a normal custom component:

```python
import jax.numpy as jnp


def additive_poly_component(wave, params, metadata):
    pivot = metadata["pivot"]
    x = (wave - pivot) / pivot
    return params["c0"] + params["c1"] * x + params["c2"] * x**2


custom_components = [
    make_custom_component(
        name="blue_excess",
        parameter_priors={
            "c0": {"dist": "Normal", "loc": 0.0, "scale": 0.2 * np.nanmedian(np.abs(flux))},
            "c1": {"dist": "Normal", "loc": 0.0, "scale": 0.05 * np.nanmedian(np.abs(flux))},
            "c2": {"dist": "Normal", "loc": 0.0, "scale": 0.05 * np.nanmedian(np.abs(flux))},
        },
        evaluate=additive_poly_component,
        metadata={"pivot": 3000.0},
    ),
]
```

## 4. Template convenience wrapper

The template helper remains as sugar for a common case:

```python
custom_components = [
    make_template_component(
        "alt_feii",
        wave=template_wave,
        flux=template_flux,
        fit_fwhm=True,
        fit_shift=True,
    ),
]
```

This is just sugar around `make_custom_component(...)`.

## 5. Prior naming convention

Each component parameter becomes a global prior/sample key:

- component name `alt_feii`
- local parameter `norm`
- global key `custom_alt_feii_norm`

Examples:

- `custom_alt_feii_norm`
- `custom_alt_feii_fwhm`
- `custom_alt_feii_shift`
- `custom_blue_excess_c0`
- `custom_blue_excess_c1`

## 6. Save/load note

Custom component evaluators should be importable top-level Python functions if you want posterior bundles to reload cleanly with `QSOFit.load_from_samples(...)`.

Avoid:

- lambdas
- nested functions
- local closures

Use:

- functions defined at module scope
