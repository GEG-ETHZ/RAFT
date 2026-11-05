# SPDX-FileCopyrightText: 2026 ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Mirror simulation stdout to a run.log file in the output directory, in addition to the console."""

import io
import sys
from pathlib import Path
from typing import IO, Any


class _Tee(io.TextIOBase):
    """Write-only stream that duplicates writes to multiple underlying streams."""

    def __init__(self, *streams: IO[str]) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


class RunLogger:
    """Buffers stdout until the output directory is known, then optionally mirrors it to run.log.

    Console output (gated by `verbose` at each print site, as before) is always passed
    through untouched. Everything printed before `attach_file()` is called is buffered so
    it can still be written to the log file once the output directory is resolved from
    the config.
    """

    def __init__(self) -> None:
        """Create a logger with nothing attached yet; call as a context manager to activate it."""
        self._buffer = io.StringIO()
        self._file: IO[str] | None = None
        self._old_stdout: IO[str] | None = None

    def __enter__(self) -> "RunLogger":
        """Start buffering stdout, still passing it through to the console."""
        self._old_stdout = sys.stdout
        sys.stdout = _Tee(self._old_stdout, self._buffer)
        return self

    def attach_file(self, output_directory: str, filename: str = "run.log", mode: str = "w") -> None:
        """Start mirroring stdout to `filename` in `output_directory`, including anything buffered so far."""
        self._file = open(Path(output_directory) / filename, mode)  # noqa: SIM115 (kept open for the run)
        self._file.write(self._buffer.getvalue())
        self._file.flush()
        assert self._old_stdout is not None
        sys.stdout = _Tee(self._old_stdout, self._file)

    def __exit__(self, *exc_info: Any) -> None:
        """Restore stdout and close the log file, if one was attached."""
        sys.stdout = self._old_stdout
        if self._file is not None:
            self._file.close()
