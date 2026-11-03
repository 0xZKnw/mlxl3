"""Exercise the AR checker through real disposable pipes, including broken engines."""

import copy
import subprocess
from pathlib import Path
from unittest.mock import patch

import physical_flash_ar_bridge as driver
import pytest


@pytest.mark.parametrize(
    "fault",
    [
        "ok",
        "hash",
        "count",
        "nonfinite",
        "mode",
        "empty",
        "invalid",
        "silent",
        "endless",
        "accept-invalid",
        "corrupt",
    ],
)
def test_real_transport_rejects_faults_and_reaps_child(tmp_path, fault):
    engine = tmp_path / "engine"
    engine.write_text(
        "#!/usr/bin/env python3\n"
        + f"fault={fault!r}\n"
        + """
import json,signal,sys,time
cancelled=False
corrupt=False
def stop(*args):
 global cancelled
 cancelled=True
signal.signal(signal.SIGUSR1,stop)
def emit(**v): print(json.dumps(v),flush=True)
print("dyld[1]: /System/Library/Frameworks/Metal.framework/Metal",file=sys.stderr,flush=True)
emit(type='ready')
for line in sys.stdin:
 r=json.loads(line)
 if r['type']=='shutdown': break
 if r['type']=='flash_memory':
  emit(type='flash_memory',request_id=r['request_id'],ledger={'expert_slots_by_layer':[53]*48,'prefetch':{'submitted':1,'published':1}});continue
 cancelled=False
 rid=r['request_id']
 assert r['mtp'] is False and r['dflash2'] is False
 assert r['reuse_prompt_cache'] is False
 if fault=='silent': time.sleep(10)
 if fault=='endless':
  while True:
   emit(type='heartbeat')
   time.sleep(.005)
 if fault=='invalid': print('null',flush=True);continue
 if not r['messages'] and fault!='accept-invalid':
  emit(type='error',request_id=rid,message='rejected');continue
 emit(type='generation_mode',request_id=rid,mtp_active=fault=='mode',dflash_active=False)
 emit(type='context_usage',request_id=rid)
 if rid=='cancel-context_usage':
  limit=time.monotonic()+1
  while not cancelled and time.monotonic()<limit: time.sleep(.001)
 if not cancelled: emit(type='delta',request_id=rid,text='' if fault=='empty' else 'answer')
 if r['max_tokens']==128:
  limit=time.monotonic()+1
  while not cancelled and time.monotonic()<limit: time.sleep(.001)
 if cancelled:
  corrupt=True
  emit(type='cancelled',request_id=rid);continue
 emit(type='complete',request_id=rid,token_hash='bad' if fault=='hash' or (fault=='corrupt' and corrupt) else '9f850af1cbba074b',cache_context='answer',stats={'generated_tokens':0 if fault=='count' else 2,'decode_tokens':1,'decode_tps':float('nan') if fault=='nonfinite' else 10,'decode_seconds':.1,'prefill_tps':10,'ttft_seconds':.1})
"""
    )
    engine.chmod(0o755)
    root = tmp_path / "logs"
    root.mkdir()
    children = []
    original = subprocess.Popen

    def spawn(*args, **kwargs):
        p = original(*args, **kwargs)
        children.append(p)
        return p

    report = {"status": "started", "parity": False}
    with patch("scripts.native_json_process.subprocess.Popen", side_effect=spawn):
        if fault == "ok":
            driver.verify(engine, Path("model"), root, report, 2)
            assert report["parity"] and report["completed"] == 4 and len(report["cancelled"]) == 2
        else:
            with pytest.raises((AssertionError, ValueError, RuntimeError, TimeoutError)):
                driver.verify(engine, Path("model"), root, report, 0.2)
            assert report["parity"] is False
    assert len(children) == 1 and children[0].returncode is not None
    assert (root / "events.jsonl").exists()


def test_completion_counter_type_is_strict():
    event = {"stats": {"generated_tokens": 2.0, "decode_tokens": 1}, "token_hash": driver.GOLDEN}
    with pytest.raises(AssertionError):
        driver.complete(copy.deepcopy(event), {}, "answer")
