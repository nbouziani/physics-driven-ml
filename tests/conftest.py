"""Global test configuration."""

import pytest


def pytest_configure(config):
    """Register an additional marker."""
    config.addinivalue_line(
        "markers",
        "skipcomplex: mark as skipped in complex mode")


def pytest_collection_modifyitems(session, config, items):
    from firedrake.utils import complex_mode

    for item in items:
        if complex_mode and item.get_closest_marker("skipcomplex") is not None:
            item.add_marker(pytest.mark.skip(reason="Test makes no sense in complex mode"))
