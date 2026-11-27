"""CPU regressions for DFlash2 proof of activation, parity and process peak."""

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import dflash_agent_trial as trial
from test_flash_agent_trial import inventory_events


def trial_report(label, live, peak):
    *tools, complete = inventory_events()
    history = []
    for index in range(0, len(tools), 2):
        call, result = tools[index : index + 2]
        history.append(
            {
                "content": "",
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "arguments": call["arguments"],
                            "name": "sandbox.read",
                        },
                    }
                ],
            }
        )
        history.append({"content": result["text"], "role": "tool", "name": "sandbox.read"})
    history.append({"content": "TOTAL=18", "role": "assistant"})
    complete.update(
        cache_context="TOTAL=18",
        token_hash="aaee49135bd200dc",
        turn_context=json.dumps(history),
    )
    for index, stats in enumerate(complete["round_stats"], 1):
        stats.update(
            dflash_proposed_tokens=9,
            dflash_accepted_tokens=5,
            dflash_blocks=5,
            memory={
                "process_lifetime_peak_bytes": peak - (4 - index) * 10,
                "process_footprint_bytes": peak - 40,
                "mlx_peak_bytes": 260,
                "process_id": 1001,
            },
        )
        if not live:
            stats.update(
                cached_prompt_tokens=0,
                evaluated_prompt_tokens=stats["prompt_tokens"],
                live_prompt_tokens=0,
            )
    complete["stats"].update(
        dflash_proposed_tokens=36,
        dflash_accepted_tokens=20,
        dflash_blocks=20,
        draft_load_seconds=0.7,
        memory=complete["round_stats"][-1]["memory"].copy(),
    )
    if not live:
        complete["stats"].update(
            cached_prompt_tokens=0, evaluated_prompt_tokens=520, live_prompt_tokens=0
        )
    return {
        "experiment": trial.EXPERIMENT,
        "status": "complete",
        "label": label,
        "model": "flash-fixture",
        "context_length": 4096,
        "fixture": trial.inventory.FILES,
        "engine_sha256": "same-engine",
        "helper_sha256": "same-helper",
        "model_registry_entry_sha256": "same-model",
        "draft_revision": trial.DRAFT_REVISION,
        "draft_sha256": {name: sha for name, (_, sha) in trial.DRAFT_FILES.items()},
        "environment": trial.ENVIRONMENT.copy(),
        "ready": {
            "mlx_version": "0.32.2",
            "runtime_commit": "mock-sha",
            "runtime_profile": "release",
            "dflash_tuning_key": "mock-key",
        },
        "request": {
            "request_id": "inventory",
            "dflash_draft_path": "/temporary/pin",
            "live_prompt_cache": live,
        },
        "task": {
            "tools": tools,
            "complete": complete,
            "wall_seconds": 12.3,
            "generation_mode": {
                "type": "generation_mode",
                "request_id": "inventory",
                "dflash_requested": True,
                "dflash_active": True,
                "mtp_requested": False,
                "mtp_active": False,
                "dflash_mode": 2,
                "dflash_proposals": 2,
                "dflash_draft_path": "/temporary/pin",
            },
        },
    }


class DFlashTrials(unittest.TestCase):
    def test_validate_requires_dflash_every_round_and_actual_acceptance(self):
        record = trial_report("B1", True, 300)
        self.assertEqual(trial.validate_task(record["task"], record["request"]), 3)
        for mutate in (
            lambda task: task.pop("generation_mode"),
            lambda task: task["generation_mode"].update(dflash_active=False),
            lambda task: task["generation_mode"].update(dflash_proposals=3),
            lambda task: task["complete"]["round_stats"][0].update(dflash_proposed_tokens=11),
            lambda task: task["complete"]["stats"].update(dflash_accepted_tokens=0),
            lambda task: task["complete"]["round_stats"][2].update(live_prompt_tokens=0),
            lambda task: task["complete"]["round_stats"][1]["memory"].update(
                process_lifetime_peak_bytes=269
            ),
            lambda task: task["complete"]["round_stats"][3]["memory"].update(
                process_footprint_bytes=301
            ),
            lambda task: task["complete"]["round_stats"][3]["memory"].update(process_id=1002),
            lambda task: task["complete"]["stats"]["memory"].pop("mlx_peak_bytes"),
        ):
            task = copy.deepcopy(record["task"])
            mutate(task)
            with self.assertRaises(trial.inventory.BridgeError):
                trial.validate_task(task, record["request"])

    def test_reject_duplicate_generation_mode(self):
        record = trial_report("A1", False, 300)
        task = record["task"]
        bridge = Mock()
        bridge.event.side_effect = [task["generation_mode"], task["generation_mode"]]
        with self.assertRaisesRegex(trial.inventory.BridgeError, "duplicate generation mode"):
            trial.run_task(bridge, record["request"], 10, {"tools": []})

    def run_compare(self, records, folder):
        paths = []
        for record in records:
            path = folder / f"{record['label']}.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            paths.append(path)
        output = folder / "comparison.json"
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                trial.__file__,
                "--output",
                str(output),
                "--compare",
                *map(str, paths),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        return result.returncode, json.loads(output.read_text(encoding="utf-8"))

    def test_abba_preserves_peaks_and_medians_in_exact_bytes(self):
        a1, b1, b2, a2 = (
            trial_report(label, label.startswith("B"), peak)
            for label, peak in (("A1", 310), ("B1", 295), ("B2", 298), ("A2", 300))
        )
        with tempfile.TemporaryDirectory() as temporary:
            code, report = self.run_compare([a1, b1, b2, a2], Path(temporary))
        self.assertEqual(code, 0, report)
        self.assertIs(report["parity"], True)
        self.assertIs(report["memory_gate_passed"], True)
        self.assertEqual(report["process_peaks_bytes"], {"A": 310, "B": 298})
        self.assertEqual(report["medians"]["A"]["process_lifetime_peak_bytes"], 305)
        self.assertEqual(report["medians"]["B"]["evaluated_prompt_tokens"], 220)

    def test_memory_regression_reports_bytes_and_fails_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            code, report = self.run_compare(
                [trial_report("A1", False, 300), trial_report("B1", True, 301)], Path(temporary)
            )
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "memory_regression")
        self.assertIs(report["parity"], True)
        self.assertIs(report["memory_gate_passed"], False)
        self.assertEqual(report["process_peaks_bytes"], {"A": 300, "B": 301})

    def test_divergence_and_profile_mismatch_do_not_claim_parity(self):
        a = trial_report("A1", False, 300)
        for mutate in (
            lambda b: b["task"]["complete"].update(token_hash="0000000000000000"),
            lambda b: b["task"]["complete"].update(assistant_context="TOTAL=19"),
            lambda b: b["environment"].update(MLXL3_FLASH_DRAFT_Q4="0"),
            lambda b: b.update(status="failed"),
        ):
            b = trial_report("B1", True, 290)
            mutate(b)
            with tempfile.TemporaryDirectory() as temporary:
                code, report = self.run_compare([a, b], Path(temporary))
            self.assertEqual(code, 1)
            self.assertIs(report["parity"], False)
            self.assertNotIn("memory_gate_passed", report)

    def test_broken_process_leaves_failed_trial(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            engine = folder / "mock-engine"
            engine.write_text(
                "#!/bin/sh\nprintf 'fixture engine failed\\n' >&2\nexit 2\n", encoding="utf-8"
            )
            engine.chmod(0o700)
            pinned_data = b"test checkpoint"
            (folder / "weights").write_bytes(pinned_data)
            pin = {"weights": (len(pinned_data), hashlib.sha256(pinned_data).hexdigest())}
            registry = folder / "registry.json"
            registry.write_text('{"models":{"fake":{"path":"/unused"}}}', encoding="utf-8")
            output = folder / "trial.json"
            argv = [
                trial.__file__,
                "--engine",
                str(engine),
                "--helper",
                str(engine),
                "--registry",
                str(registry),
                "--model",
                "fake",
                "--dflash-draft",
                str(folder),
                "--output",
                str(output),
            ]
            with patch.object(trial, "DRAFT_FILES", pin), patch.object(sys, "argv", argv):
                self.assertEqual(trial.main(), 1)
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["engine_exit"], 2)
        self.assertIn("fixture engine failed", report["stderr_tail"])


if __name__ == "__main__":
    unittest.main()
