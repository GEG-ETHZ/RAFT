"""Run well-pattern simulations for gravity/thermoelastic permeability variants.

Cases:
1) gravity = 0, with pressure-temperature dependent permeability (TE cooling on)
2) gravity = 0, without pressure-temperature dependent permeability (TE cooling off)
3) gravity = 10, with pressure-temperature dependent permeability (TE cooling on)

Outputs are written under:
    outputs/symmetric_doublet/
      1 (no gravity with TE cooling)/
      2 (no gravity w/o TE cooling)/
      3 (gravity and TE cooling)/
    outputs/symmetric_five_spot/
      1 (no gravity with TE cooling)/
      2 (no gravity w/o TE cooling)/
      3 (gravity and TE cooling)/
    outputs/winerack/
      1 (no gravity with TE cooling)/
      2 (no gravity w/o TE cooling)/
      3 (gravity and TE cooling)/

Run with:
    uv run python scripts/well_config_testing.py

Choose a configuration from the displayed menu, or bypass it with:
    uv run python scripts/well_config_testing.py --setup doublet
    uv run python scripts/well_config_testing.py --setup five-spot
    uv run python scripts/well_config_testing.py --setup winerack
    uv run python scripts/well_config_testing.py --setup all
"""

import argparse
import copy
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, TypedDict, cast

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from raft.subsurface import main  # noqa: E402

BASE_CONFIG = ROOT / "config" / "config_symmetric_doublet.yaml"
Verbosity = Literal["quiet", "lite", "full"]
SetupChoice = Literal["all", "doublet", "five-spot", "winerack"]


class CaseDef(TypedDict):
    """One well-config test case definition."""

    title: str
    gravity: float
    te_cooling: bool


class SetupDef(TypedDict):
    """One well-pattern setup to run for all test cases."""

    name: str
    config: Path


CASE_DEFS: list[CaseDef] = [
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

SETUPS: dict[str, SetupDef] = {
    "doublet": {"name": "symmetric doublet", "config": ROOT / "config" / "config_symmetric_doublet.yaml"},
    "five-spot": {
        "name": "symmetric five-spot",
        "config": ROOT / "config" / "config_symmetric_five_spot.yaml",
    },
    "winerack": {"name": "wine-rack", "config": ROOT / "config" / "config_winerack.yaml"},
}


def _build_case_config(base: dict[str, Any], gravity: float, te_cooling: bool) -> dict[str, Any]:
    """Return one case configuration from a base config."""
    case_cfg = copy.deepcopy(base)
    case_cfg["mesh"]["file"] = str((ROOT / "config" / case_cfg["mesh"]["file"]).resolve())
    case_cfg["simulation"]["gravity"] = gravity

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


def _prompt_setup() -> str:
    """Ask the user which well pattern to run."""
    choices = list(SETUPS)
    print("\nChoose a well configuration:")
    for number, key in enumerate(choices, start=1):
        print(f"  {number}. {SETUPS[key]['name']}")
    print(f"  {len(choices) + 1}. all configurations")

    while True:
        selection = input(f"Enter a number (1-{len(choices) + 1}): ").strip()
        if selection == str(len(choices) + 1):
            return "all"
        for number, key in enumerate(choices, start=1):
            if selection == str(number):
                return key
        print("Invalid selection. Please enter a number from the list.")


def run_all_cases(verbosity: Verbosity = "lite", setup_choice: SetupChoice = "all") -> None:
    """Run all requested cases for each well-pattern setup."""
    setups = SETUPS.values() if setup_choice == "all" else [SETUPS[setup_choice]]
    for setup in setups:
        setup_name = setup["name"]
        setup_config = setup["config"]
        with setup_config.open(encoding="utf-8") as f:
            base_cfg: dict[str, Any] = yaml.safe_load(f)

        output_root = ROOT / cast(str, base_cfg["simulation"]["output_directory"])
        output_root.mkdir(parents=True, exist_ok=True)

        with TemporaryDirectory(prefix="raft_well_cases_") as tmp_dir:
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

                print(f"\nRunning {setup_name} case: {title}")
                main(config_name=cfg_path.name, config_dir=tmp_path, verbosity=verbosity)
                src = output_root / run_name
                dst = output_root / _output_subfolder(title)
                if dst.exists():
                    shutil.rmtree(dst)
                src.replace(dst)

        print(f"\nCompleted {setup_name}. Outputs written to: {output_root}")


def main_cli() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--verbosity",
        choices=("quiet", "lite", "full"),
        default="lite",
        help="RAFT console output verbosity for each case.",
    )
    parser.add_argument(
        "--setup",
        choices=("all", *SETUPS),
        help="Skip the menu and select a configuration; each runs all three cases.",
    )
    args = parser.parse_args()
    try:
        setup_choice = args.setup if args.setup is not None else _prompt_setup()
    except EOFError:
        parser.exit(2, "\nNo selection provided. Run interactively or specify --setup.\n")
    run_all_cases(verbosity=cast(Verbosity, args.verbosity), setup_choice=cast(SetupChoice, setup_choice))


if __name__ == "__main__":
    main_cli()
