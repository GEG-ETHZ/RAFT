"""
Module containing demo test.

This module can be deleted or renamed and used for actual tests.  Run tests with `uv run pytest` in the
project root.
"""


def test_import() -> None:
    """
    Demo test that verifies that tango_reservoir_simulator can be imported.
    """
    import tango_reservoir_simulator  # noqa: F401  # type: ignore[import]

    assert tango_reservoir_simulator.__version__ == "0.1.0"
