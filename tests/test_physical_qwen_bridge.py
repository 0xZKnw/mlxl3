import json
import subprocess

import physical_qwen_bridge as driver
import pytest


@pytest.mark.parametrize(
    "fault", ["ok", "empty", "hash", "count", "history", "crash", "silent", "invalid", "malformed"]
)
def test_bridge_gate_rejects_invalid_outputs_and_reaps(tmp_path, fault):
    fake = tmp_path / "engine"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        + f"fault={fault!r}\n"
        + """
import json,sys,time
print(json.dumps({'type':'ready'}),flush=True)
for line in sys.stdin:
    r=json.loads(line)
    if r['type']=='shutdown':break
    if fault=='silent':time.sleep(30)
    if fault=='crash':sys.exit(3)
    if fault=='invalid':print('null',flush=True);continue
    if fault=='malformed':print('{oops',flush=True);continue
    rid=r['request_id']
    if r['max_tokens']==0:
        print(json.dumps({'type':'error','request_id':rid}),flush=True);continue
    print(json.dumps({'type':'delta','request_id':rid,'text':'' if fault=='empty' else 'valid'}),flush=True)
    print(json.dumps({'type':'complete','request_id':rid,'token_hash':'' if fault=='hash' else 'abc',
        'cache_context':'' if fault=='history' else 'history','stats':{'generated_tokens':0 if fault=='count' else r['max_tokens']}}),flush=True)
"""
    )
    fake.chmod(0o755)
    output = tmp_path / "report"
    if fault == "ok":
        assert len(driver.exercise(fake, tmp_path, output, native=False, timeout=1)) == 3
    else:
        with pytest.raises(
            (ValueError, RuntimeError, TimeoutError, subprocess.SubprocessError, AttributeError)
        ):
            driver.exercise(fake, tmp_path, output, native=False, timeout=0.2)
    report = json.loads((output / "report.json").read_text())
    assert report["parity"] is (fault == "ok")
    assert report["status"] == ("validated" if fault == "ok" else "failed")
    assert report["reaped"] is True
    with pytest.raises(FileExistsError):
        driver.exercise(fake, tmp_path, output, native=False, timeout=1)
