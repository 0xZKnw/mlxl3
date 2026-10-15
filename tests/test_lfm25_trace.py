import ast

import pytest
from generate_lfm25_trace import generated_rows, relative_rms, source_prompts


def test_prompt_reader_uses_only_literals_and_rejects_missing_or_ambiguous_lists(tmp_path):
    path = tmp_path / "prompts.py"
    prompts = [(f"Question {i}", []) for i in range(39)]
    path.write_text("CONVERSATIONS = " + repr(prompts))
    assert source_prompts(path) == [row[0] for row in prompts]
    for text in (
        "",
        "CONVERSATIONS = []",
        "CONVERSATIONS = danger()",
        path.read_text() + "\nCONVERSATIONS = []",
    ):
        path.write_text(text)
        with pytest.raises(ValueError):
            source_prompts(path)
    assert isinstance(ast.parse(path.read_text()), ast.Module)


def test_response_export_crops_eos_and_padding_and_preserves_actual_prompt_ids():
    prompts = [[5], [4, 5]]
    rows = generated_rows(prompts, [[0, 5, 6, 9, 0], [4, 5, 8, 8, 8]], 2, [9], 64, 10)
    assert rows[0]["input_ids"] == [5] and rows[0]["response_ids"] == [6, 9]
    assert rows[0]["conversation"] == 64 and rows[0]["eos_reason"] == "stop_token"
    assert rows[1]["response_ids"] == [8, 8, 8] and rows[1]["eos_reason"] == "max_new_tokens"


@pytest.mark.parametrize(
    "prompts,sequences,prefix",
    [
        ([], [], 1),
        ([[1]], [], 1),
        ([[1]], [[1]], 1),
        ([[1]], [[1, -1]], 1),
        ([[1]], [[1, 10]], 1),
        ([[1]], [[1, True]], 1),
        ([[1]], [[0, 2]], 1),
        ([[1]], [[1, 2]], 0),
    ],
)
def test_bad_source_traces_are_never_calibration_rows(prompts, sequences, prefix):
    with pytest.raises(ValueError):
        generated_rows(prompts, sequences, prefix, [9], 0, 10)


def test_expert_parity_has_nonempty_finite_reference_and_detects_mutation():
    torch = pytest.importorskip("torch")
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert relative_rms(a, a) == 0
    assert relative_rms(-a, a) == 2
    for actual, reference in (
        (a, torch.zeros_like(a)),
        (a[:0], a[:0]),
        (a[:1], a),
        (a * float("nan"), a),
    ):
        with pytest.raises(ValueError):
            relative_rms(actual, reference)


def test_small_bf16_transformers_model_generates_with_fresh_recurrent_state():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import Lfm2MoeConfig, Lfm2MoeForCausalLM

    config = Lfm2MoeConfig(
        hidden_size=128,
        intermediate_size=128,
        moe_intermediate_size=128,
        num_experts=8,
        num_experts_per_tok=2,
        num_dense_layers=1,
        num_hidden_layers=3,
        num_attention_heads=1,
        num_key_value_heads=1,
        vocab_size=128,
        layer_types=["conv", "full_attention", "conv"],
        max_position_embeddings=512,
        eos_token_id=127,
        pad_token_id=0,
        tie_word_embeddings=True,
        use_expert_bias=True,
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    torch.manual_seed(25)
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = Lfm2MoeForCausalLM(config).eval()
    finally:
        torch.set_default_dtype(old_dtype)
    ids = torch.tensor([[4, 5, 6]])
    with torch.inference_mode():
        logits = model(input_ids=ids, logits_to_keep=0).logits
        first = model.generate(
            ids, attention_mask=torch.ones_like(ids), max_new_tokens=4, do_sample=False
        )
        second = model.generate(
            ids, attention_mask=torch.ones_like(ids), max_new_tokens=4, do_sample=False
        )
        padded = torch.tensor([[0, 4, 5], [4, 5, 6]])
        batched = model.generate(
            padded,
            attention_mask=torch.tensor([[0, 1, 1], [1, 1, 1]]),
            max_new_tokens=4,
            do_sample=False,
        )
    assert logits.shape == (1, 3, 128) and torch.isfinite(logits).all()
    assert first.shape[0] == 1 and 3 < first.shape[1] <= 7 and torch.equal(first, second)
    assert first.min() >= 0 and first.max() < 128
    rows = generated_rows([[4, 5], [4, 5, 6]], batched.tolist(), 3, [127], 0, 128)
    assert len(rows) == 2 and all(r["response_ids"] for r in rows)
