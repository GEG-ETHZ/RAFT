import sys
from pathlib import Path

import pytest
import yaml
from scripts import well_config_testing as runner


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["--setup", "all"], ["doublet", "five-spot", "winerack"]),
        (["--setup", "doublet"], ["doublet"]),
        (["--setup", "five-spot"], ["five-spot"]),
        (["--setup", "winerack"], ["winerack"]),
    ],
)
def test_setup_selection_runs_three_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arguments: list[str], expected: list[str]
) -> None:
    setups: dict[str, runner.SetupDef] = {}
    for key in runner.SETUPS:
        config_path = tmp_path / f"{key}.yaml"
        config = {
            "mesh": {"file": "example.msh"},
            "material": {"solid": {"b_BB": [1]}},
            "simulation": {"output_directory": str(tmp_path / key)},
        }
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        setups[key] = {"name": key, "config": config_path}

    calls: list[tuple[str, float, bool]] = []

    def fake_main(*, config_name: str, config_dir: Path, verbosity: str) -> None:
        assert verbosity == "lite"
        cfg = yaml.safe_load((config_dir / config_name).read_text(encoding="utf-8"))
        output = Path(cfg["simulation"]["output_directory"])
        calls.append((output.name, cfg["simulation"]["gravity"], "b_BB" in cfg["material"]["solid"]))
        (output / Path(config_name).stem).mkdir(parents=True)

    monkeypatch.setattr(runner, "SETUPS", setups)
    monkeypatch.setattr(runner, "main", fake_main)
    monkeypatch.setattr(sys, "argv", ["well_config_testing.py", *arguments])
    runner.main_cli()

    assert calls == [
        (key, gravity, cooling)
        for key in expected
        for gravity, cooling in [(0.0, True), (0.0, False), (10.0, True)]
    ]
    for key in expected:
        for case in runner.CASE_DEFS:
            assert (tmp_path / key / runner._output_subfolder(case["title"])).is_dir()


def test_invalid_setup_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["well_config_testing.py", "--setup", "invalid"])
    with pytest.raises(SystemExit) as exc:
        runner.main_cli()
    assert exc.value.code == 2


@pytest.mark.parametrize(
    ("selection", "expected"),
    [("1", "doublet"), ("2", "five-spot"), ("3", "winerack"), ("4", "all")],
)
def test_menu_dispatches_selection(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], selection: str, expected: str
) -> None:
    monkeypatch.setattr(sys, "argv", ["well_config_testing.py"])
    monkeypatch.setattr("builtins.input", lambda _: selection)
    calls: list[tuple[str, str]] = []

    def fake_run_all_cases(verbosity: str, setup_choice: str) -> None:
        calls.append((verbosity, setup_choice))

    monkeypatch.setattr(runner, "run_all_cases", fake_run_all_cases)
    runner.main_cli()
    assert calls == [("lite", expected)]
    output = capsys.readouterr().out
    assert all(name in output for name in ("symmetric doublet", "symmetric five-spot", "wine-rack", "all"))


def test_menu_reprompts_on_invalid_input(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    answers = iter(["", "5", "hello", "2"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert runner._prompt_setup() == "five-spot"
    assert capsys.readouterr().out.count("Invalid selection") == 3


def test_menu_reports_end_of_input(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["well_config_testing.py"])

    def end_input(_: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", end_input)
    with pytest.raises(SystemExit) as exc:
        runner.main_cli()
    assert exc.value.code == 2
    assert "No selection provided" in capsys.readouterr().err
