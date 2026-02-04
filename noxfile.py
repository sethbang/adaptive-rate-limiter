"""Nox sessions for testing across multiple Python versions."""

import nox

# Test against Python 3.10 through 3.13
nox.options.sessions = ["tests"]
nox.options.default_venv_backend = "uv"


@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
def tests(session):
    """Run the test suite with pytest."""
    session.install(".[full,dev]")
    session.run("pytest", "tests/", "-q", "--no-cov", *session.posargs)


@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
def type_check(session):
    """Run mypy type checking."""
    session.install(".[full,dev]")
    session.install("mypy")
    session.run("mypy", "src/adaptive_rate_limiter", *session.posargs)
