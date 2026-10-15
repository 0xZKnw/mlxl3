import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from quantize_lfm25_modal import conversion_command, convert_model, run_logged


@pytest.mark.parametrize("probe,rows,cols", [(False, "250", "2048"), (True, "16", "128")])
def test_quality_recipe_preserves_head_and_uses_mcg_hq(tmp_path, probe, rows, cols):
    source = tmp_path / "source with spaces; $(false)"
    command = conversion_command(source, tmp_path / "work", tmp_path / "output", probe)
    assert command[command.index("-i") + 1] == str(source)
    for flag, expected in [
        ("-b", "4"),
        ("-hb", "16"),
        ("-cb", "mcg"),
        ("-cr", rows),
        ("-cc", cols),
        ("--out_scales", "auto"),
    ]:
        assert command[command.index(flag) + 1] == expected
    assert "--hq" in command and "--resume" not in command


def test_live_log_quotes_arguments_and_propagates_failure(tmp_path):
    argument = "literal 'quote' $(exit 8); `exit 9`"
    log = tmp_path / "log 'literal'; $(false)"
    run_logged([sys.executable, "-c", "import sys; print(sys.argv[1])", argument], log, 5)
    assert log.read_text().strip() == argument
    with pytest.raises(subprocess.CalledProcessError):
        run_logged([sys.executable, "-c", "print('failed'); raise SystemExit(7)"], log, 5)
    assert log.read_text().strip() == "failed"


def test_silent_timeout_reaps_child_group(tmp_path):
    pid_file = tmp_path / "pid"
    code = (
        "import subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "open(sys.argv[1],'w').write(str(p.pid)); time.sleep(60)"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        run_logged([sys.executable, "-c", code, str(pid_file)], tmp_path / "log", 1)
    import os
    import time

    pid = int(pid_file.read_text())
    for _ in range(50):
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False
        )
        if not state.stdout.strip() or state.stdout.strip().startswith("Z"):
            break
        time.sleep(0.02)
    else:
        os.kill(pid, 9)
        pytest.fail("Timed-out conversion left a live grandchild")


def test_remote_job_refuses_overwrite_before_loading_gpu(tmp_path, monkeypatch):
    import quantize_lfm25_modal as module

    monkeypatch.setattr(module, "DATA", tmp_path)
    (tmp_path / "v2").mkdir()
    (tmp_path / "v2" / "original").write_text("preserved")
    with pytest.raises(FileExistsError):
        convert_model()
    assert (tmp_path / "v2" / "original").read_text() == "preserved"


@pytest.mark.parametrize("change", ["none", "changed", "missing"])
def test_converter_hash_guard_catches_wrong_wheel_sources(tmp_path, monkeypatch, change):
    import hashlib

    import quantize_lfm25_modal as module

    path = tmp_path / "quantize.py"
    path.write_bytes(b"pinned source")
    expected = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}
    monkeypatch.setattr(module, "SOURCE_HASHES", expected)
    if change == "changed":
        path.write_bytes(b"different converter")
    elif change == "missing":
        path.unlink()
    if change == "none":
        assert module.check_converter_sources(tmp_path) == expected
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            module.check_converter_sources(tmp_path)


@pytest.mark.parametrize(
    "approval", [None, [], {"status": "failed"}, {"status": "passed", "converter_commit": "wrong"}]
)
def test_full_job_requires_matching_native_probe_approval(tmp_path, monkeypatch, approval):
    from types import SimpleNamespace

    import quantize_lfm25_modal as module

    monkeypatch.setattr(module, "DATA", tmp_path)
    committed = []
    monkeypatch.setattr(module, "volume", SimpleNamespace(commit=lambda: committed.append(True)))
    if approval is not None:
        (tmp_path / "probe-approved.json").write_text(json.dumps(approval))
    with pytest.raises((FileNotFoundError, ValueError, AttributeError)):
        convert_model()
    report = json.loads((tmp_path / "v2" / "report.json").read_text())
    assert report["status"] == "failed" and report["publication"] is False
    assert committed and not (tmp_path / "v2" / "output").exists()
