"""
Tests für preflight_check.py
"""
import json
import sys
import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Skript importieren (liegt eine Ebene höher)
sys.path.insert(0, str(Path(__file__).parent.parent))
from preflight_check import load_spec, resolve_path, check_imports, check_r_environment, main


# ── load_spec ────────────────────────────────────────────────────────────────

def test_load_spec_valid(tmp_path):
    """Gültige run_spec.json wird korrekt geladen."""
    spec = {"inputs": {"data_dir": "/data"}, "outputs": {"output_dir": "/out"}}
    spec_file = tmp_path / "run_spec.json"
    spec_file.write_text(json.dumps(spec))

    result = load_spec(spec_file)

    assert result["inputs"]["data_dir"] == "/data"
    assert result["_spec_dir"] == str(tmp_path)


def test_load_spec_missing_file(tmp_path):
    """Fehlende run_spec.json wirft FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Run spec not found"):
        load_spec(tmp_path / "nicht_vorhanden.json")


def test_load_spec_invalid_json(tmp_path):
    """Ungültiges JSON wirft einen Fehler."""
    spec_file = tmp_path / "run_spec.json"
    spec_file.write_text("{ kein gültiges json }")

    with pytest.raises(Exception):
        load_spec(spec_file)


# ── resolve_path ─────────────────────────────────────────────────────────────

def test_resolve_path_absolute(tmp_path):
    """Absoluter Pfad bleibt absolut."""
    result = resolve_path(tmp_path, str(tmp_path))
    assert result.is_absolute()


def test_resolve_path_relative(tmp_path):
    """Relativer Pfad wird relativ zu spec_dir aufgelöst."""
    result = resolve_path(tmp_path, "unterordner/datei.txt")
    assert result == (tmp_path / "unterordner" / "datei.txt").resolve()


def test_resolve_path_with_env_var(tmp_path, monkeypatch):
    """Umgebungsvariablen im Pfad werden aufgelöst."""
    monkeypatch.setenv("MEIN_ORDNER", str(tmp_path))
    result = resolve_path(tmp_path, "$MEIN_ORDNER/datei.txt")
    assert str(tmp_path) in str(result)


# ── check_imports ─────────────────────────────────────────────────────────────

def test_check_imports_all_present():
    """Bekannte installierte Pakete werden nicht als fehlend gemeldet."""
    missing = check_imports(["json", "pathlib", "os"])
    assert missing == []


def test_check_imports_missing_package():
    """Nicht installiertes Paket wird als fehlend erkannt."""
    missing = check_imports(["dieses_paket_existiert_sicher_nicht_xyz"])
    assert "dieses_paket_existiert_sicher_nicht_xyz" in missing


def test_check_imports_mixed():
    """Mischung aus vorhandenen und fehlenden Paketen."""
    missing = check_imports(["os", "paket_fehlt_abc123", "sys"])
    assert "paket_fehlt_abc123" in missing
    assert "os" not in missing
    assert "sys" not in missing


def test_check_imports_empty_list():
    """Leere Liste ergibt keine fehlenden Pakete."""
    missing = check_imports([])
    assert missing == []


# ── main() ───────────────────────────────────────────────────────────────────

def _make_spec(tmp_path, extra=None):
    """Hilfsfunktion: erstellt eine gültige run_spec.json mit echten Pfaden."""
    script = tmp_path / "dummy_script.py"
    script.write_text("pass")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    metadata = tmp_path / "meta.tsv"
    metadata.write_text("participant_id\tsession\nSUB01\t1\n")
    out_dir = tmp_path / "output"

    spec = {
        "script": str(script),
        "inputs": {
            "data_dir": str(data_dir),
            "metadata_file": str(metadata),
        },
        "outputs": {
            "output_dir": str(out_dir),
        },
    }
    if extra:
        spec.update(extra)

    spec_file = tmp_path / "run_spec.json"
    spec_file.write_text(json.dumps(spec))
    return spec_file


def test_main_success(tmp_path, capsys):
    """main() läuft durch wenn alle Pfade existieren und Pakete vorhanden sind."""
    spec_file = _make_spec(tmp_path)

    with patch("sys.argv", ["preflight_check.py", str(spec_file)]):
        with patch("preflight_check.check_imports", return_value=[]):
            main()

    captured = capsys.readouterr()
    assert "✓ Script found" in captured.out
    assert "✓ Data directory" in captured.out
    assert "✓ All required packages are installed" in captured.out


def test_main_missing_packages(tmp_path, capsys):
    """main() gibt Fehlermeldung aus und beendet sich wenn Pakete fehlen."""
    spec_file = _make_spec(tmp_path)

    with patch("sys.argv", ["preflight_check.py", str(spec_file)]):
        with patch("preflight_check.check_imports", return_value=["numpy", "pandas"]):
            with pytest.raises(SystemExit) as exc:
                main()

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "numpy" in captured.out


def test_main_missing_script(tmp_path):
    """main() wirft FileNotFoundError wenn Script-Pfad nicht existiert."""
    spec = {
        "script": str(tmp_path / "nicht_vorhanden.py"),
        "inputs": {
            "data_dir": str(tmp_path),
            "metadata_file": str(tmp_path / "meta.tsv"),
        },
        "outputs": {"output_dir": str(tmp_path / "out")},
    }
    spec_file = tmp_path / "run_spec.json"
    spec_file.write_text(json.dumps(spec))

    with patch("sys.argv", ["preflight_check.py", str(spec_file)]):
        with pytest.raises(FileNotFoundError, match="Script not found"):
            main()


def test_main_missing_data_dir(tmp_path):
    """main() wirft FileNotFoundError wenn data_dir nicht existiert."""
    script = tmp_path / "script.py"
    script.write_text("pass")
    spec = {
        "script": str(script),
        "inputs": {
            "data_dir": str(tmp_path / "nicht_vorhanden"),
            "metadata_file": str(tmp_path / "meta.tsv"),
        },
        "outputs": {"output_dir": str(tmp_path / "out")},
    }
    spec_file = tmp_path / "run_spec.json"
    spec_file.write_text(json.dumps(spec))

    with patch("sys.argv", ["preflight_check.py", str(spec_file)]):
        with pytest.raises(FileNotFoundError, match="Data directory not found"):
            main()


def test_main_wrong_venv(tmp_path):
    """main() beendet sich mit Code 3 wenn falsches Python verwendet wird."""
    spec_file = _make_spec(tmp_path, extra={"venv_python": "/anderer/python"})

    with patch("sys.argv", ["preflight_check.py", str(spec_file)]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 3


def test_main_without_script_is_allowed(tmp_path, capsys):
    """main() akzeptiert run_spec ohne script und prüft nur Pfade/Pakete."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    metadata = tmp_path / "meta.tsv"
    metadata.write_text("participant_id\tsession\nSUB01\t1\n")

    spec = {
        "inputs": {
            "data_dir": str(data_dir),
            "metadata_file": str(metadata),
        },
        "outputs": {
            "output_dir": str(tmp_path / "output"),
        },
    }
    spec_file = tmp_path / "run_spec.json"
    spec_file.write_text(json.dumps(spec))

    with patch("sys.argv", ["preflight_check.py", str(spec_file)]):
        with patch("preflight_check.check_imports", return_value=[]):
            main()

    captured = capsys.readouterr()
    assert "Script not specified" in captured.out
    assert "✓ Data directory" in captured.out


# ── check_r_environment ─────────────────────────────────────────────────────

def test_check_r_environment_rscript_missing(monkeypatch, tmp_path):
    """Fehlt Rscript im PATH, wird das als Problem gemeldet."""
    monkeypatch.setattr("preflight_check.shutil.which", lambda name: None)
    problems = check_r_environment(tmp_path)
    assert any("Rscript not found" in p for p in problems)


def test_check_r_environment_missing_lock(monkeypatch, tmp_path):
    """Fehlende renv.lock wird gemeldet, auch wenn Rscript vorhanden ist."""
    monkeypatch.setattr("preflight_check.shutil.which", lambda name: "/usr/bin/Rscript")
    monkeypatch.setattr(
        "preflight_check.subprocess.run",
        lambda *a, **k: SimpleNamespace(stdout="", returncode=0),
    )
    problems = check_r_environment(tmp_path)
    assert any("renv.lock not found" in p for p in problems)


def test_check_r_environment_missing_packages(monkeypatch, tmp_path):
    """Fehlende R-Pakete werden über die Rscript-Ausgabe erkannt."""
    (tmp_path / "renv.lock").write_text("{}")
    monkeypatch.setattr("preflight_check.shutil.which", lambda name: "/usr/bin/Rscript")
    monkeypatch.setattr(
        "preflight_check.subprocess.run",
        lambda *a, **k: SimpleNamespace(stdout="lme4,emmeans", returncode=0),
    )
    problems = check_r_environment(tmp_path)
    assert any("lme4" in p and "emmeans" in p for p in problems)


def test_check_r_environment_all_ok(monkeypatch, tmp_path):
    """Keine Probleme, wenn Rscript, renv.lock und alle Pakete vorhanden sind."""
    (tmp_path / "renv.lock").write_text("{}")
    monkeypatch.setattr("preflight_check.shutil.which", lambda name: "/usr/bin/Rscript")
    monkeypatch.setattr(
        "preflight_check.subprocess.run",
        lambda *a, **k: SimpleNamespace(stdout="", returncode=0),
    )
    problems = check_r_environment(tmp_path)
    assert problems == []