"""Shared fixtures for ASH-FS unit tests."""
import os

import numpy as np
import pytest

from ashfs.config import load_config, Config

# ---------------------------------------------------------------------------
# Resolve path to config.yaml relative to the project root so tests can be
# run from any working directory.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(__file__)
_CONFIG_PATH = os.path.join(_HERE, "..", "config.yaml")


@pytest.fixture(scope="session")
def cfg() -> Config:
    """Session-scoped Config loaded from config.yaml."""
    return load_config(_CONFIG_PATH)


@pytest.fixture
def rng() -> np.random.Generator:
    """Per-test reproducible RNG seeded at 42."""
    return np.random.default_rng(42)


def make_l2_normed(rng: np.random.Generator, n: int, d: int = 768) -> np.ndarray:
    """Return (n, d) float32 array of L2-normalised random unit vectors.

    Uses the provided RNG so callers control reproducibility.
    """
    vecs = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


# ---------------------------------------------------------------------------
# Slow-test CLI option — registered here (in conftest) so it is available to
# all test files including test_integration.py.  Placing it in a test module
# causes pytest to register it too late / from the wrong plugin scope.
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    """Register --run-slow CLI flag for the entire test suite."""
    try:
        parser.addoption(
            "--run-slow",
            action="store_true",
            default=False,
            help="Run tests marked with @pytest.mark.slow.",
        )
    except ValueError:
        # Option already registered (e.g. by a nested conftest).
        pass


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow — skipped unless --run-slow is passed.",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow", default=False):
        skip_slow = pytest.mark.skip(reason="Pass --run-slow to run this test.")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
