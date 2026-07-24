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
    xc = 20.4  # random.random()*51
    yc = random.random() * 75
    L = random.random() * 82
    angle = 90  # random.random()*360
    config_mesh["fractures"][FRACTURE_INDEX]["values"][0][0] = xc
    config_mesh["fractures"][FRACTURE_INDEX]["values"][0][1] = yc
    config_mesh["fractures"][FRACTURE_INDEX]["values"][1][0] = L
    config_mesh["fractures"][FRACTURE_INDEX]["values"][2][0] = angle

    mesh = load_mesh(config_mesh, verbose=True)

    print(f"Fracture angle = {angle} deg")
    debugging_mesh_plot(mesh["nodes"], mesh["conn"])
