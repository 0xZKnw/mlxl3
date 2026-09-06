"""Failure-path regressions from the v1 audit; no network or large weights."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
from mlx import nn
from mlx_lm.models.cache import ArraysCache
import pytest

from mlxl3 import cli, checkpoint, registry, hub


def test_truncated_prefilled_thinking_is_not_an_answer_or_tool_call():
    tool = '<tool_call>{"name":"search","arguments":{}}</tool_call>'
    for marker in ('<think>', '<|channel>thought'):
        raw = cli._restore_reasoning_opener(tool, 'assistant\n' + marker + '\n')
        assert raw.startswith(marker)
        assert not cli._parse_tool_calls(raw)
        assert tool not in cli._assistant_context(raw)
        assert cli._restore_reasoning_opener(raw, marker) == raw


def test_failed_prefill_cannot_poison_next_turn():
    class Model:
        calls = 0
        def make_cache(self):
            cache = ArraysCache(size=1)
            cache[0] = mx.array([], dtype=mx.int32)
            return [cache]
        def __call__(self, tokens, *, cache):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError('injected failure')
            cache[0][0] = mx.concatenate([cache[0][0], tokens.reshape(-1)])
    session, model = cli.GenerationSession(), Model()
    with patch.object(cli, '_select_prefill_step_size', return_value=2):
        with pytest.raises(RuntimeError):
            session.prepare(model, [1, 2, 3, 4, 99], [1, 2, 3, 4])
        assert session.prompt_cache is None and session.tokens == []
        session.prepare(model, [1, 2, 3, 4, 99], [1, 2, 3, 4])
        assert session.prompt_cache[0][0].tolist() == [1, 2, 3, 4]


@pytest.mark.parametrize('payload', [[], None, 'broken'])
def test_bad_registry_shape_is_actionable(tmp_path, monkeypatch, payload):
    monkeypatch.setenv('MLXL3_HOME', str(tmp_path))
    (tmp_path / 'models.json').write_text(json.dumps(payload))
    with pytest.raises(registry.RegistryError):
        registry.load_registry()


def test_tool_examples_and_bad_arguments_are_not_executed():
    call = '<tool_call>{"name":"search","arguments":{}}</tool_call>'
    assert not cli._parse_tool_calls('<think>' + call + '</think>No search needed')
    assert not cli._parse_tool_calls('```xml\n' + call + '\n```')
    assert not cli._parse_tool_calls('Example: ' + call)
    with pytest.raises(cli.MCPError):
        cli._parse_tool_calls('<tool_call>{"name":"search","arguments":"broken"}</tool_call>')
    assert cli._parse_tool_calls('<|tool_call>call:search{query:<|"|>test<|"|>}<tool_call|>') == [cli.ToolCallRequest('search', {'query': 'test'})]


def test_cancel_between_tools_and_transcript_replay():
    cancelled = False
    calls = []
    class MCP:
        chat_tools = [{'type': 'function', 'function': {'name': 'first'}}]
        tools = {}
        def call(self, name, arguments):
            nonlocal cancelled
            calls.append(name)
            cancelled = True
            return SimpleNamespace(text='ok', is_error=False)
    raw = ''.join('<tool_call>{"name":"%s","arguments":{}}</tool_call>' % x for x in ('first', 'second'))
    with patch.object(cli, '_stream_response', return_value=(raw, SimpleNamespace(context_used=1))), patch.object(cli, '_json_event'):
        with pytest.raises(cli.GenerationCancelled):
            cli._bridge_generate(None, None, [], request_id='audit', max_tokens=1, temperature=0, top_k=1, repetition_penalty=1, mcp=MCP(), should_cancel=lambda: cancelled)
    assert calls == ['first']
    transcript = [{'role': 'assistant', 'content': '', 'tool_calls': [{'type': 'function', 'function': {'name': 'search', 'arguments': {'q': 'test'}}}]}, {'role': 'tool', 'name': 'search', 'content': 'source text'}, {'role': 'assistant', 'content': 'answer'}]
    assert cli._bridge_messages([{'role': 'assistant', 'content': 'answer', 'turn_context': json.dumps(transcript)}]) == transcript


def test_missing_ordinary_weights_rejected(tmp_path):
    class Model(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.proj = nn.Linear(2, 2, bias=False)
    (tmp_path / 'config.json').write_text('{"model_type":"audit"}')
    args = SimpleNamespace(from_dict=lambda value: value)
    with patch.object(checkpoint, 'list_exl3_modules', return_value=[]), patch.object(checkpoint, '_get_classes', return_value=(Model, args)), patch.object(checkpoint, '_load_all_safetensors', return_value={}):
        with pytest.raises(ValueError, match='missing.*tensors'):
            checkpoint.load_exl3_model(tmp_path)


def test_http_progress_is_visible(tmp_path):
    events = []
    def snapshot(**kwargs):
        bar = kwargs['tqdm_class'](total=100, unit='B', desc='Downloading')
        bar.update(70)
        bar.close()
        raise RuntimeError('stop before assembly')
    with patch.object(hub, 'snapshot_download', side_effect=snapshot):
        with pytest.raises(RuntimeError):
            hub._download_selected('test/repo', 'a'*40, {'size_bytes': 100, 'files': {}}, tmp_path, tmp_path / 'dest', 'name', events.append)
    assert any(e['completed'] == 70 for e in events)
