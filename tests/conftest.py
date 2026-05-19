import asyncio
import sys
from pathlib import Path
from unittest.mock import DEFAULT

import pytest

# Add src directory to path for test imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


@pytest.fixture
def close_coro():
    """Factory for a ``side_effect`` that disposes of coroutine arguments.

    Real coroutine runners — ``loop.run_until_complete`` and
    ``asyncio.run_coroutine_threadsafe`` — consume the coroutine they are
    handed. A bare ``MagicMock`` standing in for one does not, so the
    coroutine is never awaited; when the garbage collector later reclaims it
    Python raises ``RuntimeWarning: coroutine ... was never awaited``, which —
    depending on GC timing — can surface as a fatal
    ``PytestUnraisableExceptionWarning``.

    Use the returned callable as the mock's ``side_effect``. It closes any
    coroutine passed as the first positional argument (mirroring the real
    runner) and returns ``unittest.mock.DEFAULT`` so the mock's configured
    ``return_value`` still applies.

    Pass ``raises=`` to simulate a runner that fails: the coroutine is still
    closed first, then the exception is raised.
    """

    def _make(*, raises=None):
        def _side_effect(maybe_coro=None, *args, **kwargs):
            if asyncio.iscoroutine(maybe_coro):
                maybe_coro.close()
            if raises is not None:
                raise raises
            return DEFAULT

        return _side_effect

    return _make
