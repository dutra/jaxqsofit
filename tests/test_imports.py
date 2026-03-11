def test_package_imports():
    from jaxqsofit import QSOFit, build_default_prior_config

    assert QSOFit is not None
    assert callable(build_default_prior_config)

