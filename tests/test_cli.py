from __future__ import annotations

import io
from types import SimpleNamespace

import mlx_lm

from mlxl3 import cli
from mlxl3.registry import ModelEntry


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert messages == [{"role": "user", "content": "hello"}]
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return "rendered prompt"


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


def test_thinking_renderer_handles_tags_split_across_tokens() -> None:
    stream = io.StringIO()
    renderer = cli.ThinkingRenderer(stream=stream, color=False)
    for piece in ("<thi", "nk>\nraison", "nement\n</thi", "nk>\nréponse"):
        renderer.feed(piece)
    renderer.finish()

    assert stream.getvalue() == ("── Réflexion ──\nraisonnement\n── Réponse ──\nréponse\n")


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
