API Reference
=============

Top-level exports
-----------------

.. currentmodule:: jaxqsofit

Fitting interface
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   QSOFit
   load_from_samples

Configuration helpers
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   build_default_prior_config
   build_default_bal_components
   DEFAULT_LINE_CONFIG
   DEFAULT_LINE_PRIOR_ROWS

Custom components
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   CustomComponentSpec
   CustomLineComponentSpec
   make_custom_component
   make_custom_line_component
   make_template_component
   negative_gaussian_bal_component

Component evaluation
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   SpectralComponentConfig
   evaluate_joint_spectral_components

Plot styling
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   style_path
   use_style

Core
----

.. currentmodule:: jaxqsofit.core

.. autoclass:: QSOFit
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: jaxqsofit.load_from_samples

Defaults
--------

.. currentmodule:: jaxqsofit.defaults

.. autofunction:: build_default_prior_config

.. autofunction:: build_default_bal_components

.. autodata:: DEFAULT_LINE_CONFIG
   :annotation:

.. autodata:: DEFAULT_LINE_PRIOR_ROWS
   :annotation:

Custom Components
-----------------

.. currentmodule:: jaxqsofit.custom_components

.. autoclass:: CustomComponentSpec
   :members:

.. autoclass:: CustomLineComponentSpec
   :members:

.. autofunction:: make_custom_component

.. autofunction:: make_custom_line_component

.. autofunction:: make_template_component

Components
----------

.. currentmodule:: jaxqsofit.components

.. autoclass:: SpectralComponentConfig
   :members:

.. autofunction:: evaluate_joint_spectral_components

Plot Styling
------------

.. currentmodule:: jaxqsofit.mplstyle

.. autofunction:: style_path

.. autofunction:: use_style

Model internals
---------------

Most users should interact with :class:`jaxqsofit.QSOFit` and the configuration
helpers above. The lower-level model module exposes a few reusable utilities.

.. currentmodule:: jaxqsofit.model

.. autofunction:: negative_gaussian_bal_component

.. autofunction:: build_fsps_template_grid

.. autofunction:: build_tied_line_meta_from_linelist
