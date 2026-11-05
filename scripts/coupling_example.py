# SPDX-FileCopyrightText: 2026 ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Example coupling loop: a plant heat exchanger cooling the AGS well loop against a cold sink.

The plant sets the inlet flow rate and temperature and the outlet pressure; RAFT returns the
outlet temperature. Each coupling step is iterated until the plant's reinjection temperature
T_in = T_out - effectiveness * (T_out - T_sink) stops changing, then accepted. The temperature
and pressure along the well at each accepted step are plotted to outputs/coupling_example_wells.png.

Run with `uv run python scripts/coupling_example.py`.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from geg_raft.api import RaftModel  # noqa: E402
from geg_raft.postprocessing import WellProfile  # noqa: E402

DAY = 86400.0
PLOT_PATH = Path(__file__).parent.parent / "outputs" / "coupling_example_wells.png"


def plot_well_profiles(profiles: list[tuple[float, WellProfile]], well: int, path: Path) -> None:
    """Temperature and pressure along the well at each accepted step, coloured by time."""
    fig, (ax_T, ax_p) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = matplotlib.colormaps["viridis"](np.linspace(0.0, 1.0, len(profiles)))
    for (day, profile), color in zip(profiles, colors, strict=True):
        ax_T.plot(profile.s, profile.T, color=color, label=f"day {day:.0f}")
        ax_p.plot(profile.s, profile.p / 1e6, color=color)

    # Shade the parts of the well outside the rock domain
    s, inside = profiles[0][1].s, profiles[0][1].inside
    for ax in (ax_T, ax_p):
        for k in np.flatnonzero(~inside[:-1] | ~inside[1:]):
            ax.axvspan(s[k], s[k + 1], color="0.9", lw=0)
        ax.set_xlabel(f"distance along well {well} from its inlet (m)")
        ax.grid(alpha=0.3)
    ax_T.set_ylabel("T (K)")
    ax_p.set_ylabel("p (MPa)")
    ax_T.legend(fontsize=8)
    fig.suptitle("Well profiles at each accepted coupling step (grey: outside the rock domain)")
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=130)


def main() -> None:
    """Run the coupled simulation and print each accepted step."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=float, default=720.0, help="simulated time")
    parser.add_argument("--step-days", type=float, default=60.0, help="coupling step")
    parser.add_argument("--effectiveness", type=float, default=0.8, help="heat exchanger effectiveness")
    parser.add_argument("--sink", type=float, default=300.0, help="cold sink temperature (K)")
    parser.add_argument("--flow", type=float, default=10.0, help="circulation rate (kg/s)")
    parser.add_argument("--tol", type=float, default=1.0e-3, help="reinjection temperature tolerance (K)")
    parser.add_argument("--well", type=int, default=0, help="well to plot")
    args = parser.parse_args()

    cp = 4250.0  # fluid specific heat, as in the config (J/kgK)
    p_out = 1.0e6
    model = RaftModel("config_example_AGS_coupled_single.yaml", 0.0, verbosity="quiet")
    # model = RaftModel("config_example_AGS_coupled_laterals.yaml", 0.0, verbosity="lite")

    profiles = [(0.0, model.well_profiles()[args.well])]  # initial condition
    T_in = 300.0  # initial guess for the reinjection temperature
    times = np.arange(0.0, args.days * DAY + 0.5 * args.step_days * DAY, args.step_days * DAY)
    print(
        f"{'day':>6} {'iterations':>10} {'T_out (K)':>10} {'T_in (K)':>10} {'p_in (kPa)':>10} {'p_out (kPa)':>10} {'power (kW)':>10}"  # noqa
    )
    for t_start, t_end in zip(times[:-1], times[1:], strict=True):
        # Fixed-point iteration between the plant and the reservoir
        for iteration in range(1, 51):
            bcs = {
                "well_1_inlet": {"p": 0.0, "T": T_in, "q": args.flow},  # inlet p is unused
                "well_1_outlet": {"p": p_out, "T": 0.0, "q": 0.0},  # outlet T and q are results
            }
            results = model.solve(t_start, t_end, bcs)
            # print(results)
            T_out = results["well_1_outlet"]["T"]
            p_in = results["well_1_inlet"]["p"]
            T_in_new = T_out - args.effectiveness * (T_out - args.sink)
            if abs(T_in_new - T_in) < args.tol:
                break
            T_in = T_in_new
        else:
            raise RuntimeError(f"Coupling did not converge in step ending at day {t_end / DAY:.0f}.")

        # Converged: accept the step and write its output
        model.get_fresh()
        profiles.append((t_end / DAY, model.well_profiles()[args.well]))
        power = args.flow * cp * (T_out - T_in)
        print(
            f"{t_end / DAY:6.0f} {iteration:10d} {T_out:10.3f} {T_in:10.3f} {p_in / 1e3:10.3f} {p_out / 1e3:10.3f}  {power / 1e3:10.2f}"  # noqa
        )

    plot_well_profiles(profiles, args.well, PLOT_PATH)
    print(f"Well profiles plotted to {PLOT_PATH}")


if __name__ == "__main__":
    main()
