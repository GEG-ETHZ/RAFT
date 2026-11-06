# SPDX-FileCopyrightText: 2026 ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""THM simulator for native coupling with TANGO"""

from importlib.metadata import version

from .api import RaftModel

__all__ = ["RaftModel"]
__version__ = version("geg-raft")
