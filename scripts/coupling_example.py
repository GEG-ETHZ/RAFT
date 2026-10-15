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

"""Example coupling loop: a plant heat exchanger cooling the AGS well loop against a cold sink.

The plant sets the inlet flow rate and temperature and the outlet pressure; RAFT returns the
outlet temperature. Each coupling step is iterated until the plant's reinjection temperature
T_in = T_out - effectiveness * (T_out - T_sink) stops changing, then accepted.

Run with `uv run python scripts/coupling_example.py`.
"""

import argparse

import numpy as np

from raft.api import RaftModel

DAY = 86400.0


def main() -> None:
    """Run the coupled simulation and print each accepted step."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=float, default=360.0, help="simulated time")
    parser.add_argument("--step-days", type=float, default=5.0, help="coupling step")
    parser.add_argument("--effectiveness", type=float, default=0.8, help="heat exchanger effectiveness")
    parser.add_argument("--sink", type=float, default=300.0, help="cold sink temperature (K)")
    parser.add_argument("--flow", type=float, default=10.0, help="circulation rate (kg/s)")
    parser.add_argument("--tol", type=float, default=1.0e-3, help="reinjection temperature tolerance (K)")
    args = parser.parse_args()

    cp = 4250.0  # fluid specific heat, as in the config (J/kgK)
    p_out = 1.0e6
    model = RaftModel("config_example_AGS_coupled_single.yaml", 0.0, verbosity="lite")

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
        power = args.flow * cp * (T_out - T_in)
        print(
            f"{t_end / DAY:6.0f} {iteration:10d} {T_out:10.3f} {T_in:10.3f} {p_in / 1e3:10.3f} {p_out / 1e3:10.3f}  {power / 1e3:10.2f}"  # noqa
        )


if __name__ == "__main__":
    main()
