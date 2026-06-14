API Reference
=============

Top-level exports
-----------------

.. currentmodule:: jaxqsofit

Fitting interface
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   JAXQSOFit
   load_from_samples

Configuration
~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   FitConfig
   Observation
   SpectroscopyData
   PSFPhotometryData
   PreprocessingConfig
   ContinuumConfig
   HostConfig
   LineConfig
   InferenceConfig
   OutputConfig
   PriorConfig
   ContinuumPriorConfig
   HostPriorConfig
   LinePriorConfig
   FeIIPriorConfig
   PSFPriorConfig
   fit_config_from_mapping

Model defaults
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   build_default_prior_config
   build_default_bal_components

Custom components
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   CustomComponentSpec
   CustomLineComponentSpec
   make_custom_component
   make_custom_line_component
   make_template_component

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

.. autoclass:: JAXQSOFit
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

.. autofunction:: jaxqsofit.load_from_samples
   :no-index:

Configuration
-------------

.. currentmodule:: jaxqsofit.config

.. autoclass:: FitConfig
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

.. autoclass:: Observation
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

.. autoclass:: SpectroscopyData
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

.. autoclass:: PSFPhotometryData
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

.. autoclass:: PreprocessingConfig
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

.. autoclass:: ContinuumConfig
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

.. autoclass:: HostConfig
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

.. autoclass:: LineConfig
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

.. autoclass:: InferenceConfig
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

.. autoclass:: OutputConfig
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

.. autofunction:: fit_config_from_mapping

Defaults
--------

.. currentmodule:: jaxqsofit.defaults

.. autofunction:: build_default_prior_config

.. autofunction:: build_default_bal_components

.. data:: DEFAULT_LINE_CONFIG

   Default emission-line configuration used by
   :func:`build_default_prior_config`.

.. data:: DEFAULT_LINE_PRIOR_ROWS

   Default line-prior table rows used by :func:`build_default_prior_config`.

Custom Components
-----------------

.. currentmodule:: jaxqsofit.custom_components

.. autoclass:: CustomComponentSpec
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

.. autoclass:: CustomLineComponentSpec
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

.. autofunction:: make_custom_component

.. autofunction:: make_custom_line_component

.. autofunction:: make_template_component

Components
----------

.. currentmodule:: jaxqsofit.components

.. autoclass:: SpectralComponentConfig
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

.. autofunction:: evaluate_joint_spectral_components

Plot Styling
------------

.. currentmodule:: jaxqsofit.mplstyle

.. autofunction:: style_path

.. autofunction:: use_style

Model internals
---------------

Most users should interact with :class:`jaxqsofit.JAXQSOFit` and the configuration
helpers above. The lower-level model module exposes a few reusable utilities.

.. currentmodule:: jaxqsofit.model

.. autoclass:: FSPSTemplateGrid
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: qso_fsps_joint_model

.. autofunction:: reconstruct_posterior_components

.. autofunction:: build_fsps_template_grid

.. autofunction:: build_tied_line_meta_from_linelist

.. autofunction:: negative_gaussian_bal_component

.. autofunction:: unred
