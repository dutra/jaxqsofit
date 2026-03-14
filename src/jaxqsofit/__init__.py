from .core import QSOFit
from .defaults import DEFAULT_LINE_CONFIG, DEFAULT_LINE_PRIOR_ROWS, build_default_prior_config

def load_from_samples(*args, **kwargs):
    """Load a saved posterior bundle and return a QSOFit object."""
    return QSOFit.load_from_samples(*args, **kwargs)

__all__ = [
    "QSOFit",
    "load_from_samples",
    "DEFAULT_LINE_CONFIG",
    "DEFAULT_LINE_PRIOR_ROWS",
    "build_default_prior_config",
]
