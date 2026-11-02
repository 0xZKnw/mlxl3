"""Disposable CPU validation of GPU1 admission; never starts an engine or GPU."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("gpu-runner.py")


def load(path):
    spec = importlib.util.spec_from_file_location("cold_gpu_runner_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sut():
    return load(SCRIPT)


@pytest.fixture
def documents(sut, tmp_path, monkeypatch):
    anchor_path = tmp_path / "anchor-gpu1.json"
    policy_path = tmp_path / "policy-gpu1.json"
    anchor = {
        "experiment": sut.EXPERIMENT,
        "timestamp": "2026-11-01T13:00:00+00:00",
        "vm_bytes": {"Swapouts": 75119296512, "Swapins": 30750539776},
    }
    anchor_path.write_text(json.dumps(anchor))
    policy = {
        "experiment": sut.EXPERIMENT,
        "authorized_by_user": sut.AUTHORIZATION,
        "anchor_path": str(anchor_path),
        "anchor_sha256": sut.F02["file_sha"](anchor_path),
        "maximum_new_swapout_bytes": 32 << 20,
        "maximum_new_swapin_bytes": 256 << 20,
        "no_reanchor": True,
        "previous_anchor": dict(sut.PREVIOUS_ANCHOR),
    }
    policy_path.write_text(json.dumps(policy))
    monkeypatch.setattr(sut, "ANCHOR_PATH", anchor_path)
    monkeypatch.setattr(sut, "POLICY_PATH", policy_path)
    return anchor_path, policy_path, anchor, policy


def validate(sut, documents, policy=None, anchor=None, path=None, digest=None):
    anchor_path, _, original_anchor, original_policy = documents
    return sut.validate_documents(
        original_policy if policy is None else policy,
        original_anchor if anchor is None else anchor,
        str(anchor_path) if path is None else path,
        original_policy["anchor_sha256"] if digest is None else digest,
        str(anchor_path),
        sut.PREVIOUS_ANCHOR,
    )


def test_positive_fixture_and_loader_never_write_anchor(sut, documents):
    anchor_path, policy_path, anchor, _ = documents
    before = anchor_path.read_bytes(), policy_path.read_bytes()
    assert validate(sut, documents) == anchor["vm_bytes"]
    assert sut.load_baseline(anchor_path) == anchor["vm_bytes"]
    assert (anchor_path.read_bytes(), policy_path.read_bytes()) == before
    assert sut.F02["run"].__globals__["load_baseline"] is sut.load_baseline


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("experiment", "POSTP08-F02-01-w1"),
        ("authorized_by_user", "yes"),
        ("authorized_by_user", None),
        ("anchor_path", "/another/anchor-gpu1.json"),
        ("anchor_path", "anchor-gpu1.json"),
        ("anchor_sha256", "0" * 64),
        ("maximum_new_swapout_bytes", (32 << 20) + 1),
        ("maximum_new_swapout_bytes", (32 << 20) - 1),
        ("maximum_new_swapout_bytes", True),
        ("maximum_new_swapout_bytes", float(32 << 20)),
        ("maximum_new_swapin_bytes", (256 << 20) + 1),
        ("maximum_new_swapin_bytes", "268435456"),
        ("no_reanchor", False),
        ("no_reanchor", 1),
        ("previous_anchor", {}),
        ("previous_anchor", {"path": "/wrong", "sha256": "0" * 64}),
    ],
)
def test_changed_policy_refused(sut, documents, field, value):
    policy = copy.deepcopy(documents[3])
    policy[field] = value
    with pytest.raises(AssertionError):
        validate(sut, documents, policy=policy)


@pytest.mark.parametrize(
    "field",
    ["authorized_by_user", "anchor_sha256", "no_reanchor", "maximum_new_swapout_bytes"],
)
def test_incomplete_policy_refused(sut, documents, field):
    policy = copy.deepcopy(documents[3])
    del policy[field]
    with pytest.raises(KeyError):
        validate(sut, documents, policy=policy)


@pytest.mark.parametrize("counter", ["Swapouts", "Swapins"])
@pytest.mark.parametrize("value", [-1, None, True, 0.0, "0"])
def test_noninteger_or_negative_anchor_counter_refused(sut, documents, counter, value):
    anchor = copy.deepcopy(documents[2])
    anchor["vm_bytes"][counter] = value
    with pytest.raises(AssertionError, match="VM baseline"):
        validate(sut, documents, anchor=anchor)


@pytest.mark.parametrize("timestamp", [None, "", "garbage", "2026-11-01T13:00:00"])
def test_invalid_or_unzoned_timestamp_refused(sut, documents, timestamp):
    anchor = copy.deepcopy(documents[2])
    anchor["timestamp"] = timestamp
    with pytest.raises((AssertionError, ValueError)):
        validate(sut, documents, anchor=anchor)


def test_transplanted_changed_and_incomplete_anchor_refused(sut, documents, tmp_path):
    anchor_path, _, anchor, _ = documents
    transplant = tmp_path / "transplanted.json"
    transplant.write_bytes(anchor_path.read_bytes())
    with pytest.raises(AssertionError, match="fixed anchor path"):
        sut.load_baseline(transplant)
    with pytest.raises(AssertionError, match="absolute anchor path"):
        validate(sut, documents, path="anchor-gpu1.json")
    anchor_path.write_text(anchor_path.read_text() + "\n")
    with pytest.raises(AssertionError, match="changed anchor"):
        sut.load_baseline(anchor_path)
    altered = copy.deepcopy(anchor)
    altered["experiment"] = "another"
    with pytest.raises(AssertionError, match="experiment"):
        validate(sut, documents, anchor=altered)
    del altered["vm_bytes"]["Swapouts"]
    altered["experiment"] = sut.EXPERIMENT
    with pytest.raises(KeyError):
        validate(sut, documents, anchor=altered)


def identity(sut, documents):
    return {"sources": {str(path): sut.F02["file_sha"](path) for path in documents[:2]}}


@pytest.mark.parametrize("missing", [0, 1, "both"])
def test_identity_absence_refused_before_delegate(sut, documents, tmp_path, monkeypatch, missing):
    document = identity(sut, documents)
    if missing == "both":
        document["sources"] = {}
    else:
        del document["sources"][str(documents[missing])]
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(document))

    def forbidden(*_):
        pytest.fail("runner reached without frozen anchor and policy identity")

    monkeypatch.setitem(sut.F02, "run", forbidden)
    with pytest.raises(AssertionError):
        sut.run(tmp_path / "unused-plan.json", documents[0], identity_path)


def test_wrapper_delegates_original_arguments_with_both_identity_sources(
    sut, documents, tmp_path, monkeypatch
):
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(identity(sut, documents)))
    arguments = tmp_path / "plan.json", documents[0], identity_path
    observed = []
    monkeypatch.setitem(sut.F02, "run", lambda *args: observed.append(args))
    sut.run(*arguments)
    assert observed == [arguments]


@pytest.mark.parametrize("changed", [0, 1])
def test_original_runner_checks_sources_before_baseline_or_launch(
    sut, documents, tmp_path, monkeypatch, changed
):
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(identity(sut, documents)))
    documents[changed].write_text(documents[changed].read_text() + "\n")

    def forbidden(_):
        pytest.fail("baseline reached before rejecting changed identity source")

    monkeypatch.setitem(sut.F02["run"].__globals__, "load_baseline", forbidden)
    with pytest.raises(AssertionError, match=str(documents[changed])):
        sut.run(tmp_path / "unused-plan.json", documents[0], identity_path)


@pytest.mark.parametrize("counter", ["Swapouts", "Swapins"])
def test_reused_guard_keeps_exact_budget_boundary(sut, documents, counter):
    baseline = sut.load_baseline(documents[0])
    current = dict(baseline)
    limit = 32 << 20 if counter == "Swapouts" else 256 << 20
    current[counter] += limit
    sut.F02["check_vm"](current, baseline)
    current[counter] += 1
    with pytest.raises(AssertionError, match=counter):
        sut.F02["check_vm"](current, baseline)


def test_authorization_guard_mutation_is_detected(sut, documents, tmp_path):
    policy = copy.deepcopy(documents[3])
    policy["authorized_by_user"] = "not authorized"

    def require_rejection(function):
        try:
            function(
                policy,
                documents[2],
                str(documents[0]),
                policy["anchor_sha256"],
                str(documents[0]),
                sut.PREVIOUS_ANCHOR,
            )
        except AssertionError:
            return
        raise AssertionError("accepted unauthorized campaign")

    require_rejection(sut.validate_documents)
    source = SCRIPT.read_text()
    needle = 'assert policy["authorized_by_user"] == AUTHORIZATION, "explicit authorization"'
    assert source.count(needle) == 1
    mutant = tmp_path / "mutant.py"
    mutant.write_text(
        source.replace(needle, 'assert True, "mutant disabled authorization"').replace(
            "ROOT = Path(__file__).resolve().parents[3]", f"ROOT = Path({str(sut.ROOT)!r})"
        )
    )
    with pytest.raises(AssertionError, match="accepted unauthorized campaign"):
        require_rejection(load(mutant).validate_documents)
    require_rejection(sut.validate_documents)
    assert SCRIPT.read_text() == source
