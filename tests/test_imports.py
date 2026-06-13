def test_package_imports():
    from jaxqsofit import (
        JAXQSOFit,
        build_default_prior_config,
        make_custom_component,
        make_custom_line_component,
        make_template_component,
    )

    assert JAXQSOFit is not None
    assert callable(build_default_prior_config)
    assert callable(make_custom_component)
    assert callable(make_custom_line_component)
    assert callable(make_template_component)
