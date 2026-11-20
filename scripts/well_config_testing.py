"""Run symmetric-doublet simulations for gravity/thermoelastic permeability variants.

Cases:
1) gravity = 0, with pressure-temperature dependent permeability (TE cooling on)
2) gravity = 0, without pressure-temperature dependent permeability (TE cooling off)
3) gravity = 10, with pressure-temperature dependent permeability (TE cooling on)

Outputs are written under:
    outputs/symmetric_doublet/
      1 (no gravity with TE cooling)/
      2 (no gravity w/o TE cooling)/
      3 (gravity and TE cooling)/

Run with:
    uv run python scripts/well_config_testing.py
"""

import argparse
import copy
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from raft.subsurface import main  # noqa: E402

BASE_CONFIG = ROOT / "config" / "config_symmetric_doublet.yaml"
TARGET_OUTPUT_ROOT = "outputs/symmetric_doublet"

CASE_DEFS = [
    {
        "title": "1 (no gravity with TE cooling)",
        "gravity": 0.0,
        "te_cooling": True,
    },
    {
        "title": "2 (no gravity w/o TE cooling)",
        "gravity": 0.0,
        "te_cooling": False,
    },
    {
        "title": "3 (gravity and TE cooling)",
        "gravity": 10.0,
        "te_cooling": True,
    },
]


def _build_case_config(base: dict[str, Any], gravity: float, te_cooling: bool) -> dict[str, Any]:
    """Return one case configuration from the symmetric-doublet base config."""
    case_cfg = copy.deepcopy(base)
    case_cfg["mesh"]["file"] = str((ROOT / "config" / case_cfg["mesh"]["file"]).resolve())
    case_cfg["simulation"]["gravity"] = gravity
    case_cfg["simulation"]["output_directory"] = TARGET_OUTPUT_ROOT

    solid = case_cfg["material"]["solid"]
    if te_cooling:
        if "b_BB" not in solid:
            raise ValueError("TE cooling requested but material.solid.b_BB is missing in base config.")
    else:
        solid.pop("b_BB", None)

    return case_cfg


def _output_subfolder(title: str) -> str:
    """Return a Windows-safe output folder name preserving the case title."""
    return title.replace("/", "-")


def run_all_cases(verbosity: str = "lite") -> None:
    """Run all requested cases and place outputs in dedicated symmetric-doublet subfolders."""
    with BASE_CONFIG.open(encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    output_root = ROOT / "outputs" / "symmetric_doublet"
    output_root.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="raft_doublet_cases_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for idx, case in enumerate(CASE_DEFS, start=1):
            title = case["title"]
            gravity = case["gravity"]
            te_cooling = case["te_cooling"]

            cfg = _build_case_config(base_cfg, gravity=gravity, te_cooling=te_cooling)
            run_name = f"well_config_case_{idx}"
            cfg_path = tmp_path / f"{run_name}.yaml"
            with cfg_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)

            print(f"\nRunning case: {title}")
            main(config_name=cfg_path.name, config_dir=tmp_path, verbosity=verbosity)
            src = output_root / run_name
            dst = output_root / _output_subfolder(title)
            if dst.exists():
                shutil.rmtree(dst)
            src.replace(dst)

    print(f"\nDone. Outputs written to: {output_root}")


def main_cli() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--verbosity",
        choices=("quiet", "lite", "full"),
        default="lite",
        help="RAFT console output verbosity for each case.",
    )
    args = parser.parse_args()
    run_all_cases(verbosity=args.verbosity)


if __name__ == "__main__":
    main_cli()
