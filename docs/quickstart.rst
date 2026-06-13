Quickstart
==========

Minimal fitting example
-----------------------

.. code-block:: python

   import numpy as np
   from jaxqsofit import (
       ContinuumConfig,
       HostConfig,
       InferenceConfig,
       JAXQSOFit,
       Observation,
       OutputConfig,
       FitConfig,
       SpectroscopyData,
   )

   # Example arrays
   lam = np.linspace(3800.0, 9200.0, 2000)
   flux = 50.0 + 0.002 * (lam - 6000.0)
   err = np.full_like(flux, 0.5)
   z = 0.1

   cfg = FitConfig(
       observation=Observation(object_id='demo', redshift=z),
       spectroscopy=SpectroscopyData(wave_obs=lam, fluxes=flux, errors=err),
       continuum=ContinuumConfig(fit_feii=True, fit_balmer_continuum=True),
       host=HostConfig(enabled=True, dsps_ssp_fn='tempdata.h5'),
       inference=InferenceConfig(method='nuts'),
       output=OutputConfig(save_result=False, plot_fig=True),
   )
   q = JAXQSOFit(cfg)
   result = q.fit()

Fast mode
---------

For a fast MAP-style fit, use:

.. code-block:: python

   q.config.inference.method = 'optax'
   q.config.inference.map_steps = 1500
   q.config.inference.learning_rate = 1e-2
   result = q.fit()

Hybrid mode
-----------

Warm-start with Optax, then run NUTS:

.. code-block:: python

   q.config.inference.method = 'optax+nuts'
   q.config.inference.map_steps = 800
   q.config.inference.num_warmup = 200
   q.config.inference.num_samples = 400
   result = q.fit()
