"""
Module containing demo test.

This module can be deleted or renamed and used for actual tests.  Run tests with `uv run pytest` in the
project root.
"""


def test_import() -> None:
    """
    Demo test that verifies that raft can be imported.
    """
    import raft  # noqa: F401  # type: ignore[import]

    assert raft.__version__ == "0.1.0"
