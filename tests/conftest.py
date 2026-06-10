"""Test configuration: set anyio default backend to asyncio."""
import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "anyio: mark test to run with anyio backend (default: asyncio)",
    )
