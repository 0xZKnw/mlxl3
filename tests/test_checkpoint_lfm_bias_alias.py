"""Header-only compatibility for the known LFM expert-bias descriptor alias."""

import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from mlxl3.checkpoint import validate_checkpoint_files

LEGACY = 'model.layers.2.feed_forward.gate.expert_bias'
ACTUAL = 'model.layers.2.feed_forward.expert_bias'


def checkpoint(tmp_path, *, model_type='lfm2_moe', alias_shape=(32,), original_shape=None,
               missing_quant=False, extra_missing=None):
    tensors = {
        'lm_head.trellis': np.zeros((8, 8, 48), dtype=np.int16),
        'lm_head.suh': np.ones((128,), dtype=np.float16),
        'lm_head.svh': np.ones((128,), dtype=np.float16),
    }
    specs = {key: {'shape': list(value.shape)} for key, value in tensors.items()}
    specs[LEGACY] = {'shape': [32]}
    if extra_missing is not None:
        specs[extra_missing] = {'shape': [32]}
    if alias_shape is not None:
        tensors[ACTUAL] = np.zeros(alias_shape, dtype=np.float16)
    if original_shape is not None:
        tensors[LEGACY] = np.zeros(original_shape, dtype=np.float16)
    if missing_quant:
        del tensors['lm_head.trellis']
    save_file(tensors, tmp_path / 'model.safetensors')
    (tmp_path / 'config.json').write_text(json.dumps({'model_type': model_type}))
    (tmp_path / 'quantization_config.json').write_text(json.dumps({
        'quant_method': 'exl3',
        'tensor_storage': {'lm_head': {'quant_format': 'exl3', 'stored_tensors': specs}},
    }))
    return tmp_path


def test_lfm_bias_alias_accepts_matching_header_without_mutating_files(tmp_path):
    root = checkpoint(tmp_path)
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    validate_checkpoint_files(root)
    assert before == {path.name: path.read_bytes() for path in root.iterdir()}


@pytest.mark.parametrize('options', [
    {'alias_shape': (31,)},
    {'alias_shape': None},
    {'model_type': 'lfm2'},
    {'model_type': 'qwen3_5_moe'},
    {'original_shape': (31,)},
])
def test_lfm_bias_alias_rejects_wrong_shape_absence_or_architecture(tmp_path, options):
    root = checkpoint(tmp_path, **options)
    with pytest.raises(ValueError, match=LEGACY.replace('.', r'\.')):
        validate_checkpoint_files(root)


@pytest.mark.parametrize('options,key', [
    ({'missing_quant': True}, 'lm_head.trellis'),
    ({'extra_missing': 'model.layers.2.feed_forward.gate.weight'},
     'model.layers.2.feed_forward.gate.weight'),
    ({'extra_missing': 'other.layers.2.feed_forward.gate.expert_bias'},
     'other.layers.2.feed_forward.gate.expert_bias'),
])
def test_lfm_bias_alias_does_not_mask_other_missing_tensors(tmp_path, options, key):
    root = checkpoint(tmp_path, **options)
    with pytest.raises(ValueError, match=key.replace('.', r'\.')):
        validate_checkpoint_files(root)
