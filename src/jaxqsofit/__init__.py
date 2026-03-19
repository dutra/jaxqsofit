from .core import QSOFit
from .custom_components import (
    CustomComponentSpec,
    make_custom_component,
    make_template_component,
)
from .defaults import DEFAULT_LINE_CONFIG, DEFAULT_LINE_PRIOR_ROWS, build_default_prior_config

def load_from_samples(*args, **kwargs):
    """Load a saved posterior bundle and return a QSOFit object."""
    return QSOFit.load_from_samples(*args, **kwargs)

__all__ = [
    "QSOFit",
    "load_from_samples",
    "CustomComponentSpec",
    "make_custom_component",
    "make_template_component",
    "DEFAULT_LINE_CONFIG",
    "DEFAULT_LINE_PRIOR_ROWS",
    "build_default_prior_config",
]
