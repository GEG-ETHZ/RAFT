# Copyright 2026 The RAFT Community
#
# This file is part of RAFT.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for locating and loading config files."""

from pathlib import Path

import pytest

from raft.preprocessing import load_config
from raft.subsurface import main
from tests.test_api import _transient_config

CONFIG_DIR = Path(__file__).parent / "test_configs"


def test_config_name_without_suffix_loads_same_file() -> None:
    bare = load_config("config_test_wellconvection", CONFIG_DIR, verbose=False)
    full = load_config("config_test_wellconvection.yaml", CONFIG_DIR, verbose=False)

    assert bare == full
    assert bare["config_name"] == "config_test_wellconvection.yaml"


def test_missing_config_names_both_attempts() -> None:
    with pytest.raises(FileNotFoundError, match=r"'no_such_config' not found .*tried 'no_such_config.yaml'"):
        load_config("no_such_config", CONFIG_DIR, verbose=False)


def test_run_without_suffix_writes_to_the_config_output_folder(tmp_path: Path) -> None:
    config_name = _transient_config(tmp_path)
    main(config_name=Path(config_name).stem, config_dir=tmp_path, verbosity="quiet")

    assert (tmp_path / "out" / Path(config_name).stem / "outflow.log").is_file()
