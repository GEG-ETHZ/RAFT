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

"""Spyder debugging script: build the fractured mesh for a range of fracture
angles and plot each result with `debugging_mesh_plot`.

Run cell-by-cell in Spyder (each `# %%` block is a separate cell), or run the
whole file to loop through every angle in `FRACTURE_ANGLES` automatically.
"""

import copy
import random

from raft.meshprocessing import debugging_mesh_plot, load_mesh
from raft.preprocessing import load_config

# %% ----- User settings

CONFIG_NAME = "config_example_2D.yaml"
CONFIG_DIR = None  # None -> repo's default config/ directory

FRACTURE_INDEX = 0  # which entry in config["mesh"]["fractures"] to sweep
num_cases = 50

# %% ----- Load the base config once

config = load_config(CONFIG_NAME, CONFIG_DIR, verbose=False)
config["mesh"]["config_dir"] = config["config_dir"]

# %% ----- Loop through fracture angles, build the mesh, and plot it

for i in range(num_cases):
    config_mesh = copy.deepcopy(config["mesh"])
    xc = random.random() * 51
    yc = random.random() * 75
    L = max(random.random() * 82, 5)
    angle = random.random() * 360

    print(f"Case {i + 1}")
    print(f"xc = {xc}")
    print(f"yc = {yc}")
    print(f"L = {L}")
    print(f"Fracture angle = {angle} deg")
    config_mesh["fractures"][FRACTURE_INDEX]["values"][0][0] = xc
    config_mesh["fractures"][FRACTURE_INDEX]["values"][0][1] = yc
    config_mesh["fractures"][FRACTURE_INDEX]["values"][1][0] = L
    config_mesh["fractures"][FRACTURE_INDEX]["values"][2][0] = angle

    mesh = load_mesh(config_mesh, verbose=True)

    debugging_mesh_plot(mesh["nodes"], mesh["conn"])
