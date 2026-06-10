Installation
============

Requirements
------------

- Python 3.10+
- JAX/JAXLIB
- NumPyro
- DSPS
- dustmaps

Setup notes
-----------

1. Configure dustmaps SFD (one-time):

.. code-block:: bash

   python setup.py fetch --map-name=sfd

Then ensure dustmaps points to the directory containing the downloaded SFD maps.

2. Provide DSPS SSP templates.

This code expects an HDF5 SSP file path passed as ``dsps_ssp_fn``, for example
``tempdata.h5``.

.. code-block:: bash

   curl -L -o tempdata.h5 https://portal.nersc.gov/project/hacc/aphearin/DSPS_data/ssp_data_continuum_fsps_v3.2_lgmet_age.h5

Always set ``dsps_ssp_fn="tempdata.h5"`` to the HDF5 SSP template file you want
to use when fitting. The continuum-only DSPS template is recommended when
nebular emission lines are modeled separately.

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/burke86/jaxqsofit.git
   cd jaxqsofit
   pip install -e .

Optional test dependencies
--------------------------

.. code-block:: bash

   pip install pytest pytest-cov astroquery
