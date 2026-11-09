import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import time

out=Path(__file__).resolve().parent
exe=out/"q04-sushi-gpu-test"
identity=json.loads((out/"q04-sushi-gpu-identity.json").read_text())
assert hashlib.sha256(exe.read_bytes()).hexdigest()==identity["binary_sha256"]
for name,digest in identity["source"].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
quiet=runpy.run_path("docs/measurements/slotstream-20261026/supervisor.py")["check_quiet"]
env={k:v for k,v in os.environ.items() if not k.startswith("MLXL3_")}
env.update(MLXL3_FLASH_MODEL="/Users/justin/Documents/mix-stq1_0/models/Qwen3.8-Flash-Next-exl3-205-65c89531",
           MLXL3_SSD_CACHE_GIB="3",MLXL3_NATIVE_CACHE_SLRU="0",MLXL3_NATIVE_DEVICE_RESIDENT_ROUTE="0",
           DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer")
state={"status":"running","steps":[]}
status=out/"q04-sushi-validation-r4-status.json"
def save():status.write_text(json.dumps(state,indent=2,allow_nan=False)+"\n")
def run(name,test,extra):
    quiet({})
    folder=out/"q04-sushi-validation-r4"/name;folder.mkdir(parents=True,exist_ok=False)
    record={"name":name,"status":"running"};state["steps"].append(record);save()
    with (folder/"test.log").open("w") as f:
        child=subprocess.Popen([str(exe),test,"--exact","--ignored","--nocapture","--test-threads=1"],env={**env,**extra},stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait(timeout=240)
        except BaseException:
            if child.poll() is None:os.killpg(child.pid,signal.SIGKILL)
            child.wait();record.update(status="failed_or_interrupted",reaped=True);save();raise
    record.update(exit=code,reaped=True)
    assert code==0 and "1 passed;" in (folder/"test.log").read_text(),name
    return folder,record
try:
    folder,rec=run("oracle","gpu::route_swap_trace::gpu_picker_matches_reference_without_changing_original_routes",{})
    rec.update(status="passed",cases=12);save();print("GPU oracle PASS",flush=True)
    for tau in (0,.1,.2):
        name=f"tau{tau:g}"
        folder=out/"q04-sushi-validation-r4"/name
        folder,rec=run(name,"gpu::fast_profile_tests::native_numerical_profile_teacher_forced_quality",{
            "MLXL3_NATIVE_QUALITY_CORPUS":str(out/"quality-corpus-pilot.json"),
            "MLXL3_NATIVE_QUALITY_REFERENCE":str(out/"q01-quality-reference"),
            "MLXL3_NATIVE_QUALITY_OUTPUT":str(folder),"MLXL3_NATIVE_Q04_QUALITY_TAU":str(tau),
            "MLXL3_NATIVE_Q04_QUALITY_LIMIT":"3"})
        new=json.loads((folder/"report.json").read_text())
        prior=out/("q04-quality/tau0-limit1/report.json" if tau==0 else f"q04-quality-resume/tau{tau:g}-limit3/report.json")
        old=json.loads(prior.read_text())
        assert new["status"]=="complete" and len(new["documents"])==6 and new["policy"]["rows"]==18144
        assert new["documents"]==old["documents"],f"quality scores changed at tau{tau}"
        assert new["policy"]["swaps"]==old["policy"]["swaps"] and new["policy"]["actual_decode_cold_reads"]==old["policy"]["actual_decode_cold_reads"]
        rec.update(status="passed",scores=384,decode_layer_rows=18144,prior_scores_exactly_equal=True)
        save();print(name,"GPU/host choices + prior384 quality scores PASS",flush=True)
        with Path("opti.md").open("a") as f:f.write(f"\n- I165-Q04 Sushi-GPU {name} qualityPASS:384score records exactly equal priorqualifiedARtarget,18144actualdecodeGPU-vs-host picks checked,same swap/readcounts; raw {folder}. NoTPSyet.\n")
    state["status"]="complete_quality_and_gpu_oracle"
except BaseException as error:
    state.update(status="failed_or_interrupted",error=repr(error))
    if state["steps"] and state["steps"][-1]["status"]=="running":state["steps"][-1]["status"]="failed"
    with Path("opti.md").open("a") as f:f.write(f"\n- I165-Q04 Sushi-GPU validationFAILED/interrupted: {error!r}; {status}; no speed started.\n")
    raise
finally:save()
subprocess.run(["/Users/justin/Documents/mix-stq1_0/.venv/bin/python",str(out/"q04-sushi-speed-campaign.py")],check=True)
