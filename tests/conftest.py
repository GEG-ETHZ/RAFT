# SPDX-FileCopyrightText: 2026 ETH Zurich
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from geg_raft import postprocessing


@pytest.fixture(autouse=True)
def _redirect_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Send run outputs (domain.vtk, outflow.log, ...) to tmp_path instead of the repo's outputs/."""
    monkeypatch.setattr(postprocessing, "_DEFAULT_OUTPUT_DIR", tmp_path / "outputs")
