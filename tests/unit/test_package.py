"""Package-level smoke tests."""

import flybrain_interface


def test_package_version_is_defined() -> None:
    assert flybrain_interface.__version__ == "0.1.0"
