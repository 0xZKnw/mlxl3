"""CPU regressions for the benchmark's protocol and validation."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import flash_agent_trial as trial


def inventory_events():
    records = [("stock.txt", 17), ("returns.txt", 5), ("dispatch.txt", 4)]
    events = []
    for index, (name, units) in enumerate(records):
        common = {
            "request_id": "inventory",
            "tool_call_id": f"mcp-{index}",
            "tool_name": "sandbox.read",
        }
        events.append({"type": "tool_start", "arguments": {"file_path": name}, **common})
        events.append(
            {"type": "tool_result", "is_error": False, "text": f"1| units={units}", **common}
        )
    first = {
        "prompt_tokens": 100,
        "cached_prompt_tokens": 0,
        "evaluated_prompt_tokens": 100,
        "live_prompt_tokens": 0,
        "generated_tokens": 11,
        "decode_tokens": 10,
        "prepare_seconds": 0.25,
        "prefill_seconds": 2.0,
        "decode_seconds": 0.5,
    }
    next_step = {
        **first,
        "prompt_tokens": 140,
        "cached_prompt_tokens": 100,
        "evaluated_prompt_tokens": 40,
        "live_prompt_tokens": 20,
    }
    total = {
        "prompt_tokens": 520,
        "cached_prompt_tokens": 300,
        "evaluated_prompt_tokens": 220,
        "live_prompt_tokens": 60,
        "generated_tokens": 44,
        "decode_tokens": 40,
        "prepare_seconds": 1.0,
        "prefill_seconds": 8.0,
        "decode_seconds": 2.0,
        "tool_rounds": 3,
        "round_count": 4,
        "elapsed_seconds": 11.5,
        "tool_seconds": 0.5,
    }
    events.append(
        {
            "type": "complete",
            "request_id": "inventory",
            "round_stats": [first, *[next_step.copy() for _ in range(3)]],
            "stats": total,
            "assistant_context": "TOTAL=18",
            "stop_reason": "eos",
            "context_full": False,
            "tool_rounds": 3,
        }
    )
    return events


class TaskValidation(unittest.TestCase):
    def setUp(self):
        *tools, complete = inventory_events()
        self.task = {"tools": tools, "complete": complete}

    def test_three_real_tool_events_and_final_answer_are_required(self):
        self.assertEqual(trial.check_task(self.task, require_live=True), 3)
        for change in [
            lambda task: task["tools"].pop(),
            lambda task: task["tools"][0]["arguments"].update(file_path="returns.txt"),
            lambda task: task["tools"][1].update(tool_call_id="another-call"),
            lambda task: task["tools"][1].update(is_error=True),
            lambda task: task["tools"][3].update(text="1| units=15"),
            lambda task: task["complete"].update(assistant_context="TOTAL=19"),
            lambda task: task["complete"].update(stop_reason="max_tokens"),
            lambda task: task["complete"].update(context_full=True),
            lambda task: task["complete"].update(tool_rounds=2),
            lambda task: task["complete"]["round_stats"].clear(),
            lambda task: task["complete"]["round_stats"][0].update(prefill_seconds=float("nan")),
            lambda task: task["complete"]["stats"].update(evaluated_prompt_tokens=221),
            lambda task: task["complete"]["stats"].update(live_prompt_tokens=True),
        ]:
            malformed = copy.deepcopy(self.task)
            change(malformed)
            with self.assertRaises(trial.BridgeError):
                trial.check_task(malformed, require_live=True)

    def test_baseline_without_live_fields(self):
        for record in [self.task["complete"]["stats"], *self.task["complete"]["round_stats"]]:
            del record["live_prompt_tokens"]
        self.assertEqual(trial.check_task(self.task), 0)
        with self.assertRaisesRegex(trial.BridgeError, "no live tokens"):
            trial.check_task(self.task, require_live=True)

    def test_complete_for_other_request_does_not_finish_task(self):
        bridge = Mock()
        bridge.event.return_value = {"type": "complete", "request_id": "another-task"}
        with self.assertRaisesRegex(trial.BridgeError, "unexpected request ID"):
            trial.wait_for(bridge, 2, "complete", "inventory")

    def test_absolute_deadline_is_not_renewed_by_events(self):
        bridge = Mock()
        bridge.event.return_value = {"type": "generation_status", "request_id": "inventory"}
        with (
            patch.object(trial.time, "monotonic", side_effect=[10.0, 10.0, 10.25, 11.0]),
            self.assertRaisesRegex(trial.BridgeError, "no complete event within 1 s"),
        ):
            trial.wait_for(bridge, 1, "complete", "inventory")
        self.assertEqual([call.args for call in bridge.event.call_args_list], [(1.0,), (0.75,)])

    def test_broken_engine_leaves_failed_report_and_stderr(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            engine = folder / "mock-engine"
            engine.write_text("#!/bin/sh\nprintf 'fixture admission failed\\n' >&2\nexit 2\n")
            engine.chmod(0o700)
            registry = folder / "source.json"
            registry.write_text('{"models":{"fake":{"path":"/unused"}}}')
            output = folder / "trial.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    trial.__file__,
                    "--engine",
                    str(engine),
                    "--helper",
                    str(engine),
                    "--registry",
                    str(registry),
                    "--model",
                    "fake",
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            record = json.loads(output.read_text())
            self.assertEqual(record["status"], "failed")
            self.assertIs(record["task_passed"], False)
            self.assertEqual(record["engine_exit"], 2)
            self.assertEqual(record["stderr_tail"], "fixture admission failed")


if __name__ == "__main__":
    unittest.main()
