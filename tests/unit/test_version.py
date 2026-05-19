"""Guard against version drift between packaging metadata and __version__."""

from importlib.metadata import version

import adaptive_rate_limiter


def test_version_matches_installed_package_metadata():
    """adaptive_rate_limiter.__version__ must match the installed package
    version (sourced from pyproject.toml)."""
    assert adaptive_rate_limiter.__version__ == version("adaptive-rate-limiter")
