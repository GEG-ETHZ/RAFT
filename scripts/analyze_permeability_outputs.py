"""Analyze and export effective stress and permeability from RAFT VTK files.

The script reconstructs the pressure- and temperature-dependent permeability
law used by ``raft.matprops.get_effective_permeability``. It does not rerun
RAFT. Each saved timestep is written to a separate VTK file with cell-wise
``effective_stress`` and permeability fields.

Example:
    uv run python scripts/analyze_permeability_outputs.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

_TIMESTEP_RE = re.compile(r"Matrix\.vtk\.(\d+)$")


def _read_vtk(path: Path) -> dict[str, Any]:  # noqa: C901
    """Read the legacy ASCII VTK sections needed by this analysis."""
    lines = path.read_text().splitlines()
    result: dict[str, Any] = {}
    data_count = 0
    i = 0

    while i < len(lines):
        parts = lines[i].split()
        if not parts:
            i += 1
            continue

        keyword = parts[0]
        if keyword == "POINTS":
            count = int(parts[1])
            values = np.fromstring(" ".join(lines[i + 1 : i + 1 + count]), sep=" ")
            result["points"] = values.reshape(count, 3)
            i += count + 1
        elif keyword == "CELLS":
            count = int(parts[1])
            cells: list[list[int]] = []
            for line in lines[i + 1 : i + 1 + count]:
                values = [int(value) for value in line.split()]
                cells.append(values[1:])
            result["cells"] = cells
            i += count + 1
        elif keyword == "CELL_TYPES":
            count = int(parts[1])
            result["cell_types"] = np.fromstring(" ".join(lines[i + 1 : i + 1 + count]), dtype=int, sep=" ")
            i += count + 1
        elif keyword == "LOOKUP_TABLE":
            i += 1
        elif keyword == "SCALARS":
            name = parts[1]
            count = data_count
            if i + 1 < len(lines) and lines[i + 1].startswith("LOOKUP_TABLE"):
                i += 2
            else:
                i += 1
            values = np.fromstring(" ".join(lines[i : i + count]), sep=" ")
            result.setdefault("scalars", {})[name] = values
            i += count
        elif keyword == "VECTORS":
            count = int(result.get("point_data_count", result.get("cell_data_count", 0)))
            values = np.fromstring(" ".join(lines[i + 1 : i + 1 + count]), sep=" ")
            result.setdefault("vectors", {})[parts[1]] = values.reshape(count, 3)
            i += count + 1
        elif keyword in {"POINT_DATA", "CELL_DATA"}:
            data_count = int(parts[1])
            i += 1
        else:
            i += 1

    return result


def _timestep_files(output_dir: Path) -> list[tuple[int, Path]]:
    """Return Matrix VTK files ordered by their numeric output-step suffix."""
    files: list[tuple[int, Path]] = []
    for path in output_dir.glob("Matrix.vtk.*"):
        match = _TIMESTEP_RE.fullmatch(path.name)
        if match:
            files.append((int(match.group(1)), path))
    return sorted(files)


def _as_region_array(values: Any, count: int, name: str) -> npt.NDArray[np.float64]:
    """Convert a scalar or per-region configuration value to a 1D array."""
    array = np.atleast_1d(np.asarray(values, dtype=float))
    if array.size == 1:
        return np.full(count, array[0])
    if array.size < count:
        raise ValueError(f"{name} has {array.size} values but region {count - 1} is present")
    return array


def _effective_stress(
    pressure: npt.NDArray[np.float64],
    temperature: npt.NDArray[np.float64],
    pressure_0: npt.NDArray[np.float64],
    temperature_0: npt.NDArray[np.float64],
    cells: list[list[int]],
    regions: npt.NDArray[np.float64],
    drained_bulk_modulus: npt.NDArray[np.float64],
    thermal_expansion: npt.NDArray[np.float64],
    biot_coefficient: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return effective stress at every element's nodes, shape (elements, nodes)."""
    element_regions = np.rint(regions).astype(int)
    if np.any(element_regions < 0) or np.any(element_regions >= drained_bulk_modulus.size):
        raise ValueError("domain.vtk contains a region ID outside the material configuration")

    stresses = np.empty((len(cells), max(len(cell) for cell in cells)))
    stresses.fill(np.nan)
    for element, nodes in enumerate(cells):
        region = element_regions[element]
        node_ids = np.asarray(nodes, dtype=int)
        stresses[element, : len(nodes)] = (
            32.0e6
            + drained_bulk_modulus[region]
            * thermal_expansion[region]
            * (temperature[node_ids] - temperature_0[node_ids])
            - biot_coefficient[region] * (pressure[node_ids] - pressure_0[node_ids])
        )
    return stresses


def _cell_fields(
    pressure: npt.NDArray[np.float64],
    temperature: npt.NDArray[np.float64],
    pressure_0: npt.NDArray[np.float64],
    temperature_0: npt.NDArray[np.float64],
    cells: list[list[int]],
    regions: npt.NDArray[np.float64],
    drained_bulk_modulus: npt.NDArray[np.float64],
    thermal_expansion: npt.NDArray[np.float64],
    biot_coefficient: npt.NDArray[np.float64],
    a_bb: npt.NDArray[np.float64],
    b_bb: npt.NDArray[np.float64],
    permeability_ratios: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return cell-wise effective stress, scalar permeability, and directional ratios."""
    element_regions = np.rint(regions).astype(int)
    cell_pressure = np.array([np.mean(pressure[nodes]) for nodes in cells])
    cell_temperature = np.array([np.mean(temperature[nodes]) for nodes in cells])
    cell_pressure_0 = np.array([np.mean(pressure_0[nodes]) for nodes in cells])
    cell_temperature_0 = np.array([np.mean(temperature_0[nodes]) for nodes in cells])
    region = element_regions
    stress = (
        32.0e6
        + drained_bulk_modulus[region] * thermal_expansion[region] * (cell_temperature - cell_temperature_0)
        - biot_coefficient[region] * (cell_pressure - cell_pressure_0)
    )
    denominator = 1.0 + b_bb[region] * stress
    aperture = a_bb[region] / (b_bb[region] * denominator)
    permeability = aperture**2 / 12.0
    ratios = permeability_ratios[region]
    return stress, permeability, permeability[:, np.newaxis] * ratios


def _write_analysis_vtk(
    path: Path,
    snapshot: dict[str, Any],
    stress: npt.NDArray[np.float64],
    permeability: npt.NDArray[np.float64],
    permeability_directional: npt.NDArray[np.float64],
) -> None:
    """Write the mesh and cell-wise analysis fields in legacy ASCII VTK format."""
    points = snapshot["points"]
    cells = snapshot["cells"]
    cell_types = snapshot["cell_types"]
    with path.open("w") as file:
        file.write("# vtk DataFile Version 2.0\n")
        file.write("RAFT effective stress and permeability analysis\nASCII\n")
        file.write("DATASET UNSTRUCTURED_GRID\n")
        file.write(f"POINTS {len(points)} float\n")
        for point in points:
            file.write(f"{point[0]:.8e} {point[1]:.8e} {point[2]:.8e}\n")
        size = len(cells) + sum(len(cell) for cell in cells)
        file.write(f"CELLS {len(cells)} {size}\n")
        for cell in cells:
            file.write(f"{len(cell)} {' '.join(str(node) for node in cell)}\n")
        file.write(f"CELL_TYPES {len(cell_types)}\n")
        for cell_type in cell_types:
            file.write(f"{int(cell_type)}\n")
        file.write(f"CELL_DATA {len(cells)}\n")
        fields = [("effective_stress", stress)]
        fields.extend(
            (f"permeability_{axis}", permeability_directional[:, index])
            for index, axis in enumerate(("x", "y", "z"))
            if index < permeability_directional.shape[1]
        )
        fields.append(("permeability", permeability))
        for name, values in fields:
            file.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
            for value in values:
                file.write(f"{value:.8e}\n")


def _format_id_list(values: npt.NDArray[np.int_], limit: int = 20) -> str:
    """Format IDs without flooding the console for large affected sets."""
    ids = [str(int(value)) for value in values[:limit]]
    suffix = " ..." if values.size > limit else ""
    return ", ".join(ids) + suffix


def analyze(config_path: Path, output_dir: Path, analysis_dir: Path) -> None:  # noqa: C901
    """Analyze snapshots and write cell-wise stress and permeability fields."""
    config: dict[str, Any] = yaml.safe_load(config_path.read_text())
    solid = config["material"]["solid"]
    initial = config["initial_conditions"]

    files = _timestep_files(output_dir)
    if not files:
        raise FileNotFoundError(f"No Matrix.vtk.* files found in {output_dir}")

    domain = _read_vtk(output_dir / "domain.vtk")
    cells = domain["cells"]
    regions = domain["scalars"]["region"]
    first = _read_vtk(files[0][1])
    points = first["points"]
    cells_first = first["cells"]
    if len(cells) != len(cells_first) or not np.array_equal(cells, cells_first):
        raise ValueError("domain.vtk and the first Matrix VTK file have different connectivity")

    p_0 = first["scalars"]["p"]
    T_0 = first["scalars"]["T"]
    max_region = int(np.max(regions)) + 1
    bulk_modulus = _as_region_array(solid["bulk_mod"], max_region, "bulk_mod")
    biot = _as_region_array(solid["biot_coeff"], max_region, "biot_coeff")
    drained_bulk_modulus = (1.0 - biot) * bulk_modulus
    thermal_expansion = _as_region_array(solid["therm_expansion"], max_region, "therm_expansion")
    a_bb = _as_region_array(solid["a_BB"], max_region, "a_BB")
    b_bb = _as_region_array(solid["b_BB"], max_region, "b_BB")
    reference_aperture = a_bb / (b_bb * (1.0 + b_bb * 32.0e6))
    reference_permeability = reference_aperture**2 / 12.0
    configured_permeability = solid["permeability"]
    permeability_values = [np.atleast_1d(np.asarray(value, dtype=float)) for value in configured_permeability]
    max_dimensions = max(value.size for value in permeability_values)
    permeability_ratios = np.ones((max_region, max_dimensions))
    for region, value in enumerate(permeability_values):
        permeability_ratios[region, : value.size] = value / reference_permeability[region]
    p_initial = float(initial["p_ini"])
    T_initial = float(initial["T_ini"])
    if not (np.allclose(p_0, p_initial) and np.allclose(T_0, T_initial)):
        print("Warning: using Matrix.vtk.0 as p0/T0 because the configured initial state has gradients.")

    negative_timesteps: list[int] = []
    negative_node_occurrences = 0
    negative_element_occurrences = 0
    node_timesteps: dict[int, list[int]] = {}
    element_timesteps: dict[int, list[int]] = {}
    minimum = (np.inf, -1, -1)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    for timestep, path in files:
        snapshot = _read_vtk(path)
        if not np.array_equal(snapshot["points"], points):
            raise ValueError(f"Node coordinates changed in {path.name}")
        stress = _effective_stress(
            snapshot["scalars"]["p"],
            snapshot["scalars"]["T"],
            p_0,
            T_0,
            cells,
            regions,
            drained_bulk_modulus,
            thermal_expansion,
            biot,
        )
        cell_stress, cell_permeability, cell_permeability_directional = _cell_fields(
            snapshot["scalars"]["p"],
            snapshot["scalars"]["T"],
            p_0,
            T_0,
            cells,
            regions,
            drained_bulk_modulus,
            thermal_expansion,
            biot,
            a_bb,
            b_bb,
            permeability_ratios,
        )
        _write_analysis_vtk(
            analysis_dir / f"Matrix.vtk.{timestep}",
            snapshot,
            cell_stress,
            cell_permeability,
            cell_permeability_directional,
        )
        finite_stress = stress[np.isfinite(stress)]
        local_min = float(np.min(finite_stress))
        if local_min < minimum[0]:
            element, local_node = np.argwhere(np.isclose(stress, local_min, equal_nan=False))[0]
            node = cells[int(element)][int(local_node)]
            minimum = (local_min, timestep, node)

        negative = stress < 0.0
        negative_nodes = {cells[e][n] for e, n in zip(*np.nonzero(negative), strict=True)}
        negative_elements = set(np.nonzero(np.any(negative, axis=1))[0])
        if negative_nodes:
            negative_timesteps.append(timestep)
            negative_node_occurrences += len(negative_nodes)
            negative_element_occurrences += len(negative_elements)
            for node in negative_nodes:
                node_timesteps.setdefault(node, []).append(timestep)
            for element in negative_elements:
                element_timesteps.setdefault(int(element), []).append(timestep)

    print(f"Output directory: {output_dir}")
    print(f"Analysis VTK files: {analysis_dir}")
    print(f"Snapshots analyzed: {len(files)}")
    print(f"Negative effective stress observed: {'yes' if negative_timesteps else 'no'}")
    print(f"Minimum effective stress: {minimum[0]:.6e} Pa (timestep {minimum[1]}, node {minimum[2]})")
    print(f"Output timesteps containing negative values: {len(negative_timesteps)}")
    print(f"Negative node/timestep occurrences: {negative_node_occurrences}")
    print(f"Negative element/timestep occurrences: {negative_element_occurrences}")

    if not node_timesteps:
        return

    affected_nodes = np.array(sorted(node_timesteps), dtype=int)
    affected_elements = np.array(sorted(element_timesteps), dtype=int)
    print(f"Affected nodes ({affected_nodes.size}): {_format_id_list(affected_nodes)}")
    print(f"Affected elements ({affected_elements.size}): {_format_id_list(affected_elements)}")
    print("Affected node coordinates and timestep counts:")
    for node in affected_nodes:
        print(
            f"  node {node}: ({points[node, 0]:.6g}, {points[node, 1]:.6g}, "
            f"{points[node, 2]:.6g}), {len(node_timesteps[int(node)])} timestep(s)"
        )
    print("Affected element timestep counts:")
    for element in affected_elements:
        print(f"  element {element}: {len(element_timesteps[int(element)])} timestep(s)")


def main() -> None:
    """Parse command-line arguments and run the analysis."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "config" / "config_example_2D.yaml",
        help="RAFT YAML configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "config_example_2D",
        help="Directory containing Matrix.vtk.* and domain.vtk",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=root / "outputs" / "config_example_2D" / "permeability_analysis",
        help="Directory for generated cell-wise analysis VTK files",
    )
    args = parser.parse_args()
    analyze(args.config, args.output_dir, args.analysis_dir)


if __name__ == "__main__":
    main()
