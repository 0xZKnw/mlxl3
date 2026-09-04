from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import mlx.core as mx
import mlx_lm
import pytest
from mlx_lm.models.cache import ArraysCache

from mlxl3 import cli
from mlxl3.registry import ModelEntry


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert messages == [{"role": "user", "content": "hello"}]
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "preserve_thinking": True,
        }
        return "rendered prompt"


class FakeStateCache:
    def __init__(self, state=None):
        self._state = mx.array([], dtype=mx.int32) if state is None else state

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def meta_state(self):
        return ""

    @classmethod
    def from_state(cls, state, meta_state):
        assert meta_state == ""
        return cls(state)

    def is_trimmable(self):
        return False


class FakeCachedModel:
    def __init__(self):
        self.inputs = []

    def make_cache(self):
        return [FakeStateCache()]

    def __call__(self, tokens, *, cache):
        self.inputs.append(list(map(int, tokens.reshape(-1).tolist())))
        cache[0].state = mx.concatenate([cache[0].state, tokens.reshape(-1)])
        return mx.zeros((*tokens.shape, 4))


def test_generation_session_promotes_exact_completed_turn_cache() -> None:
    model = FakeCachedModel()
    session = cli.GenerationSession()

    suffix, generation_cache, common, evaluated = session.prepare(
        model,
        [1, 2, 9],
        [1, 2],
    )
    assert suffix == [9]
    assert common == 0
    assert evaluated == 3
    assert model.inputs == [[1, 2]]

    generation_cache[0].state = mx.concatenate(
        [generation_cache[0].state, mx.array([9, 7])]
    )
    session.finish([1, 2, 9], [7], generation_cache)
    suffix, next_cache, common, evaluated = session.prepare(
        model,
        [1, 2, 9, 7, 3, 4, 9],
        [1, 2, 9, 7, 3, 4],
    )

    assert suffix == [9]
    assert common == 4
    assert evaluated == 3
    assert model.inputs == [[1, 2], [3, 4]]
    assert next_cache[0].state.tolist() == [1, 2, 9, 7, 3, 4]
    next_cache[0].state[0] = 99
    assert session.prompt_cache[0].state.tolist() == [1, 2, 9, 7, 3, 4]


def test_generation_session_forks_recurrent_state_without_copying_arrays() -> None:
    recurrent = ArraysCache(2)
    first = mx.arange(8)
    second = mx.arange(4)
    recurrent[0] = first
    recurrent[1] = second

    cloned = cli.GenerationSession._clone_cache([recurrent])[0]

    assert cloned.cache is not recurrent.cache
    assert cloned[0] is first
    assert cloned[1] is second
    cloned[0] = mx.zeros((8,))
    assert recurrent[0] is first


def test_generation_session_chunks_cached_prefill(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_PREFILL_STEP_SIZE", 3)
    model = FakeCachedModel()

    suffix, generation_cache, common, evaluated = cli.GenerationSession().prepare(
        model,
        [1, 2, 3, 4, 5, 6, 7, 9],
        [1, 2, 3, 4, 5, 6, 7],
    )

    assert suffix == [9]
    assert common == 0
    assert evaluated == 8
    assert model.inputs == [[1, 2, 3], [4, 5, 6], [7]]
    assert generation_cache[0].state.tolist() == [1, 2, 3, 4, 5, 6, 7]


def test_generation_session_restores_nearest_block_on_divergence(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_PREFILL_STEP_SIZE", 2)
    monkeypatch.setattr(cli, "_PREFIX_CACHE_BLOCK_SIZE", 4)
    monkeypatch.setattr(cli, "_PREFIX_CACHE_BUDGET_BYTES", 1_000_000)
    model = FakeCachedModel()
    session = cli.GenerationSession()

    session.prepare(
        model,
        [1, 2, 3, 4, 5, 6, 7, 8, 99],
        [1, 2, 3, 4, 5, 6, 7, 8],
    )
    suffix, generation_cache, common, evaluated = session.prepare(
        model,
        [1, 2, 3, 4, 20, 21, 99],
        [1, 2, 3, 4, 20, 21],
    )

    assert suffix == [99]
    assert common == 4
    assert evaluated == 3
    assert model.inputs == [[1, 2], [3, 4], [5, 6], [7, 8], [20, 21]]
    assert generation_cache[0].state.tolist() == [1, 2, 3, 4, 20, 21]


def test_generation_session_pool_evicts_oldest_cache() -> None:
    pool = cli.GenerationSessionPool(budget_bytes=24)
    first = pool.acquire("first")
    first.prompt_cache = [FakeStateCache(mx.zeros((8,), dtype=mx.int32))]
    second = pool.acquire("second")
    second.prompt_cache = [FakeStateCache(mx.zeros((8,), dtype=mx.int32))]

    pool.prune("second")

    assert list(pool.sessions) == ["second"]
    assert first.prompt_cache is None


def test_prefill_step_uses_explicit_override(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_PREFILL_STEP_SIZE", 1536)
    assert cli._select_prefill_step_size(17_000_000_000, 18_000_000_000) == 1536


def test_prefill_step_adapts_to_available_memory(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_PREFILL_STEP_SIZE", 0)
    assert cli._select_prefill_step_size(10_000_000_000, 18_000_000_000) == 2048
    assert cli._select_prefill_step_size(17_100_000_000, 18_000_000_000) == 1024
    assert cli._select_prefill_step_size(17_500_000_000, 18_000_000_000) == 512


def test_prefill_step_limits_very_long_large_moe_prompts(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_PREFILL_STEP_SIZE", 0)
    assert (
        cli._select_prefill_step_size(
            10_000_000_000,
            18_000_000_000,
            token_count=32_768,
            num_experts=256,
        )
        == 512
    )
    assert (
        cli._select_prefill_step_size(
            10_000_000_000,
            18_000_000_000,
            token_count=32_768,
            num_experts=64,
        )
        == 2048
    )


def test_streaming_stats_separate_ttft_prefill_and_decode(monkeypatch, capsys) -> None:
    responses = [
        SimpleNamespace(
            text="bon",
            generation_tokens=1,
            prompt_tps=100.0,
            prompt_tokens=20,
            peak_memory=4.0,
        ),
        SimpleNamespace(
            text="jour",
            generation_tokens=2,
            prompt_tps=100.0,
            prompt_tokens=20,
            peak_memory=4.0,
        ),
    ]

    def fake_stream(*args, **kwargs):
        assert args[2] == "rendered prompt"
        yield from responses

    clock = iter((10.0, 10.5, 12.5))
    monkeypatch.setattr(mlx_lm, "stream_generate", fake_stream)
    monkeypatch.setattr(cli.time, "perf_counter", lambda: next(clock))

    text, stats = cli._stream_response(
        object(),
        FakeTokenizer(),
        [{"role": "user", "content": "hello"}],
        max_tokens=2,
        temperature=0.0,
        top_k=0,
        repetition_penalty=0.0,
    )

    assert text == "bonjour"
    assert stats.ttft_seconds == 0.5
    assert stats.prefill_tps == 100.0
    assert stats.decode_tps == 0.5
    assert capsys.readouterr().out == "── Réponse ──\nbonjour\n"


def test_streaming_generation_can_be_cancelled_cooperatively(monkeypatch) -> None:
    response = SimpleNamespace(
        text="ignored",
        token=1,
        generation_tokens=1,
        prompt_tps=100.0,
        prompt_tokens=20,
        peak_memory=4.0,
    )
    monkeypatch.setattr(mlx_lm, "stream_generate", lambda *args, **kwargs: iter([response]))

    with pytest.raises(cli.GenerationCancelled):
        cli._stream_response(
            object(),
            FakeTokenizer(),
            [{"role": "user", "content": "hello"}],
            max_tokens=2,
            temperature=0.0,
            top_k=0,
            repetition_penalty=0.0,
            should_cancel=lambda: True,
        )


def test_thinking_renderer_handles_tags_split_across_tokens() -> None:
    stream = io.StringIO()
    renderer = cli.ThinkingRenderer(stream=stream, color=False)
    for piece in ("<thi", "nk>\nraison", "nement\n</thi", "nk>\nréponse"):
        renderer.feed(piece)
    renderer.finish()

    assert stream.getvalue() == ("── Réflexion ──\nraisonnement\n── Réponse ──\nréponse\n")


def test_thinking_splitter_emits_machine_readable_phases() -> None:
    splitter = cli.ThinkingSplitter()
    fragments = []
    for piece in ("<thi", "nk>\nraison", "nement\n</thi", "nk>\nréponse"):
        fragments.extend(splitter.feed(piece))
    fragments.extend(splitter.finish())

    assert fragments == [
        ("thinking", "raison"),
        ("thinking", "nement\n"),
        ("answer", "réponse"),
    ]


def test_thinking_splitter_handles_opening_tag_prefilled_by_prompt() -> None:
    splitter = cli.ThinkingSplitter(initial_mode="thinking")
    fragments = []
    for piece in ("raison", "nement</thi", "nk>\nréponse"):
        fragments.extend(splitter.feed(piece))
    fragments.extend(splitter.finish())

    assert fragments == [
        ("thinking", "raison"),
        ("thinking", "nement"),
        ("answer", "réponse"),
    ]


def test_thinking_renderer_marks_unclosed_reasoning_as_reasoning() -> None:
    stream = io.StringIO()
    renderer = cli.ThinkingRenderer(stream=stream, color=False)
    renderer.feed("<think>still reasoning")
    renderer.finish()
    assert stream.getvalue() == "── Réflexion ──\nstill reasoning\n"


def test_thinking_renderer_uses_bright_magenta_in_a_color_terminal() -> None:
    stream = io.StringIO()
    renderer = cli.ThinkingRenderer(stream=stream, color=True)
    renderer.feed("<think>visible</think>final")
    renderer.finish()
    rendered = stream.getvalue()
    assert "\033[1;35m── Réflexion ──\033[0m" in rendered
    assert "\033[35mvisible\033[0m" in rendered
    assert "\033[1;36m── Réponse ──\033[0m" in rendered


def test_tool_call_stream_filter_hides_split_xml_payload() -> None:
    stream_filter = cli.ToolCallStreamFilter()
    visible = []
    for piece in (
        "Avant <tool_",
        "call><function=demo.echo><parameter=value>hi</parameter>",
        "</function></tool_call> Après",
    ):
        visible.extend(stream_filter.feed(piece))
    visible.extend(stream_filter.finish())

    assert "".join(visible) == "Avant  Après"


def test_parse_qwen_and_json_tool_calls() -> None:
    response = """
<tool_call>
<function=files.read>
<parameter=path>
/tmp/demo.txt
</parameter>
<parameter=limit>
12
</parameter>
</function>
</tool_call>
<tool_call>{"name":"clock.now","arguments":{"zone":"Europe/Paris"}}</tool_call>
"""
    assert cli._parse_tool_calls(response) == [
        cli.ToolCallRequest("files.read", {"path": "/tmp/demo.txt", "limit": 12}),
        cli.ToolCallRequest("clock.now", {"zone": "Europe/Paris"}),
    ]


def test_bridge_generation_executes_mcp_tool_then_returns_answer(monkeypatch, capsys) -> None:
    stats = cli.GenerationStats(0.1, 20, 30, 5, 6, 4)
    responses = iter(
        (
            (
                "inspect</think><tool_call><function=demo.echo>"
                "<parameter=value>bonjour</parameter></function></tool_call>"
            ),
            "respond</think>Résultat MCP",
        )
    )
    observed_messages = []

    def fake_stream(model, tokenizer, messages, *, on_text, on_prompt, tools, **kwargs):
        observed_messages.append(messages)
        response = next(responses)
        on_prompt("<|im_start|>assistant\n<think>\n")
        on_text(response)
        return response, stats

    class FakeMCP:
        def __init__(self):
            self.chat_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "demo.echo",
                        "description": "Echo",
                        "parameters": {"type": "object"},
                    },
                }
            ]
            self.tools = {"demo.echo": SimpleNamespace(server="demo")}

        def call(self, name, arguments):
            assert name == "demo.echo"
            assert arguments == {"value": "bonjour"}
            return SimpleNamespace(text="bonjour", is_error=False)

    monkeypatch.setattr(cli, "_stream_response", fake_stream)
    cli._bridge_generate(
        object(),
        object(),
        [{"role": "user", "content": "utilise echo"}],
        request_id="turn-mcp",
        max_tokens=-1,
        temperature=0.2,
        top_k=80,
        repetition_penalty=1.05,
        mcp=FakeMCP(),
    )

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["type"] for event in events] == [
        "delta",
        "tool_start",
        "tool_result",
        "delta",
        "delta",
        "complete",
    ]
    assert events[0]["phase"] == "thinking"
    assert events[1]["tool_name"] == "demo.echo"
    assert events[2]["text"] == "bonjour"
    assert events[-2]["text"] == "Résultat MCP"
    assert events[-1]["assistant_context"] == "Résultat MCP"
    assert observed_messages[1][-1] == {
        "role": "tool",
        "name": "demo.echo",
        "content": "bonjour",
    }


def test_chat_history_is_forwarded_to_the_next_turn(monkeypatch, capsys) -> None:
    observed = []
    replies = iter(("première réponse", "deuxième réponse"))
    stats = cli.GenerationStats(
        ttft_seconds=0.1,
        prefill_tps=50.0,
        decode_tps=40.0,
        prompt_tokens=10,
        generated_tokens=5,
        peak_memory_gb=4.0,
    )

    def fake_stream(model, tokenizer, messages, **kwargs):
        observed.append([dict(message) for message in messages])
        return next(replies), stats

    monkeypatch.setattr(cli, "_stream_response", fake_stream)
    args = SimpleNamespace(
        max_tokens=32,
        temperature=0.0,
        top_k=0,
        repetition_penalty=0.0,
    )
    messages = []
    cli._generate_turn(object(), FakeTokenizer(), messages, "question une", args)
    cli._generate_turn(object(), FakeTokenizer(), messages, "question deux", args)

    assert observed[0] == [{"role": "user", "content": "question une"}]
    assert observed[1] == [
        {"role": "user", "content": "question une"},
        {"role": "assistant", "content": "première réponse"},
        {"role": "user", "content": "question deux"},
    ]
    assert messages[-1] == {"role": "assistant", "content": "deuxième réponse"}
    capsys.readouterr()


def test_chat_history_keeps_final_answer_but_not_thinking() -> None:
    assert cli._assistant_context("<think>raisonnement</think>\nLa réponse.") == "La réponse."
    assert cli._assistant_context("<think>raisonnement tronqué") == (
        "(La réponse précédente a été interrompue avant sa conclusion.)"
    )


def test_model_table_contains_registered_metadata() -> None:
    entry = ModelEntry(
        name="lfm",
        path="/models/lfm",
        model_type="lfm2_moe",
        format="EXL3",
        bits=3.1,
        size_bytes=3_920_000_000,
        modules=2179,
        added_at="2026-09-01T00:00:00+00:00",
    )
    table = cli._format_models({entry.name: entry})
    assert "NAME" in table
    assert "lfm" in table
    assert "3.10" in table
    assert "3.9 GB" in table


def test_list_json_is_stable_for_the_native_app(monkeypatch, capsys) -> None:
    entry = ModelEntry(
        name="lfm",
        path="/models/lfm",
        model_type="lfm2_moe",
        format="EXL3",
        bits=3.1,
        size_bytes=3_920_000_000,
        modules=2179,
        added_at="2026-09-01T00:00:00+00:00",
    )
    monkeypatch.setattr(cli, "load_registry", lambda: {entry.name: entry})

    assert cli.main(["list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == [
        {
            "name": "lfm",
            "path": "/models/lfm",
            "model_type": "lfm2_moe",
            "format": "EXL3",
            "bits": 3.1,
            "size_bytes": 3_920_000_000,
            "modules": 2179,
            "added_at": "2026-09-01T00:00:00+00:00",
            "size": "3.9 GB",
        }
    ]


def test_download_command_uses_managed_storage_and_registers(monkeypatch, tmp_path, capsys) -> None:
    destination = tmp_path / "managed"
    downloaded = destination / "qwen-test"
    observed = {}
    entry = ModelEntry(
        name="qwen-test",
        path=str(downloaded),
        model_type="qwen3_5_moe",
        format="EXL3",
        bits=2.49,
        size_bytes=12_000_000_000,
        modules=42,
        added_at="2026-09-04T00:00:00+00:00",
    )

    def fake_download(**kwargs):
        observed.update(kwargs)
        return str(downloaded)

    def fake_register(name, path, *, force=False):
        assert name == "qwen-test"
        assert Path(path) == downloaded
        assert force
        return entry

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    monkeypatch.setattr(cli, "managed_models_path", lambda: destination)
    monkeypatch.setattr(cli, "register_model", fake_register)

    assert cli.main(
        [
            "download",
            "owner/Qwen-EXL3",
            "--revision",
            "2.49bpw",
            "--name",
            "qwen-test",
            "--json",
        ]
    ) == 0
    assert observed == {
        "repo_id": "owner/Qwen-EXL3",
        "revision": "2.49bpw",
        "local_dir": downloaded.resolve(),
        "force_download": False,
    }
    assert json.loads(capsys.readouterr().out)["name"] == "qwen-test"


def test_bridge_handles_thinking_tag_prefilled_by_chat_template(monkeypatch, capsys) -> None:
    stats = cli.GenerationStats(
        ttft_seconds=0.12,
        prefill_tps=70.0,
        decode_tps=90.0,
        prompt_tokens=12,
        generated_tokens=8,
        peak_memory_gb=4.1,
    )

    def fake_stream(model, tokenizer, messages, *, on_text, on_prompt, **kwargs):
        assert messages == [{"role": "user", "content": "hello"}]
        assert kwargs["max_tokens"] == -1
        on_prompt("<|im_start|>assistant\n<think>\n")
        on_text("reason</think>")
        on_text("answer")
        return "reason</think>answer", stats

    monkeypatch.setattr(cli, "resolve_model", lambda name: (name, "/tmp/model"))
    monkeypatch.setattr(cli, "_load_model", lambda path: (object(), object(), 42, 0.5, 3.9))
    monkeypatch.setattr(cli, "_stream_response", fake_stream)
    monkeypatch.setattr(
        cli.sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "type": "generate",
                    "request_id": "turn-1",
                    "messages": [{"role": "user", "content": "hello"}],
                }
            )
            + "\n"
        ),
    )

    assert cli._bridge(SimpleNamespace(model="lfm")) == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["type"] for event in events] == [
        "loading",
        "ready",
        "delta",
        "delta",
        "complete",
    ]
    assert events[2] == {
        "type": "delta",
        "request_id": "turn-1",
        "phase": "thinking",
        "text": "reason",
    }
    assert events[3]["phase"] == "answer"
    assert events[3]["text"] == "answer"
    assert events[4]["assistant_context"] == "answer"
    assert events[4]["stats"]["decode_tps"] == 90.0


def test_bridge_reports_cooperative_cancellation(monkeypatch, capsys) -> None:
    def cancel(*args, **kwargs):
        raise cli.GenerationCancelled

    monkeypatch.setattr(cli, "resolve_model", lambda name: (name, "/tmp/model"))
    monkeypatch.setattr(cli, "_load_model", lambda path: (object(), object(), 42, 0.5, 3.9))
    monkeypatch.setattr(cli, "_bridge_generate", cancel)
    monkeypatch.setattr(
        cli.sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "type": "generate",
                    "request_id": "turn-cancelled",
                    "messages": [{"role": "user", "content": "hello"}],
                }
            )
            + "\n"
        ),
    )

    assert cli._bridge(SimpleNamespace(model="lfm")) == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["type"] for event in events] == ["loading", "ready", "cancelled"]
    assert events[-1]["request_id"] == "turn-cancelled"
