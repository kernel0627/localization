"""Contract checks for the self-contained Q2 single-file runner."""

from __future__ import annotations

import ast
import csv
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from scripts.q2 import triangle_reference as core


Q2_FILE = Path(__file__).resolve().parents[2] / "scripts" / "q2" / "q2.py"
COMMANDS = ("triangle", "residual", "geometry", "budget", "noise", "protocols", "e0")


def _load_standalone_module():
    spec = importlib.util.spec_from_file_location("q2_standalone_under_test", Q2_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _isolated_env() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    return environment


def test_copied_file_help_and_budget_run_without_repository_imports(tmp_path: Path) -> None:
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    copied_file = isolated / "q2.py"
    shutil.copy2(Q2_FILE, copied_file)
    environment = _isolated_env()

    help_result = subprocess.run(
        [sys.executable, str(copied_file), "--help"],
        cwd=isolated,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert all(command in help_result.stdout for command in COMMANDS)

    output_dir = isolated / "budget_outputs"
    budget_result = subprocess.run(
        [sys.executable, str(copied_file), "budget", "--output-dir", str(output_dir)],
        cwd=isolated,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "Traceback" not in budget_result.stderr

    rows = list(csv.DictReader((output_dir / "position_to_angle.csv").open()))
    row = next(item for item in rows if item["apex_position_budget_d"] == "0.0048")
    assert np.isclose(float(row["equal_angle_bound_rad"]), 0.00059937703726082, atol=1e-12)
    assert np.isclose(float(row["verified_exact_bound_d"]), 0.0048, atol=1e-12)


def test_standalone_core_matches_repository_core_for_apex_and_all_receivers() -> None:
    standalone = _load_standalone_module()
    np.testing.assert_allclose(standalone.TARGET_TEMPLATE, core.TARGET_TEMPLATE)
    assert tuple(standalone.RECEIVER_IDS) == tuple(core.RECEIVER_IDS)

    for angles in ([np.pi / 3, np.pi / 3], [0.7, 1.3], [0.9, 1.0]):
        np.testing.assert_allclose(
            standalone.estimate_apex(angles), core.estimate_apex(angles), atol=1e-12
        )

    for receiver_id in core.RECEIVER_IDS:
        target = core.TARGET_TEMPLATE[receiver_id - 1]
        observed = core.receiver_angles(target, core.TARGET_ANCHORS)
        expected = core.estimate_receiver(receiver_id, observed, initial=target)
        actual = standalone.estimate_receiver(
            receiver_id, standalone.receiver_angles(target, standalone.TARGET_ANCHORS), initial=target
        )
        np.testing.assert_allclose(actual["position"], expected["position"], atol=1e-10)
        assert actual["success"] == expected["success"]
        assert np.isclose(
            actual["max_angle_residual"], expected["max_angle_residual"], atol=1e-12
        )


def test_standalone_source_has_no_repository_or_dynamic_import_wrapper() -> None:
    source = Q2_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(Q2_FILE))

    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)
    assert not any(name == "scripts" or name.startswith("scripts.") for name in imported_modules)
    assert not any(name == "importlib" or name.startswith("importlib.") for name in imported_modules)

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert not (node.value.id == "sys" and node.attr == "path")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"exec", "eval", "compile", "__import__"}
