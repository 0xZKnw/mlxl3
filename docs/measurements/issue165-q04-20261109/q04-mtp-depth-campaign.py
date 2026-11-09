import hashlib,json,math,os,runpy,signal,subprocess,time
from pathlib import Path
out=Path(__file__).resolve().parent
folder=out/"q04-mtp-depth-speed";folder.mkdir(exist_ok=False)
exe=out/"q04-mtp-depth-test"
identity=json.loads((out/"q04-mtp-depth-identity.json").read_text())
assert hashlib.sha256(exe.read_bytes()).hexdigest()==identity["binary_sha256"]
for name,digest in identity["source"].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
quiet=runpy.run_path("docs/measurements/slotstream-20261026/supervisor.py")["check_quiet"]
env={k:v for k,v in os.environ.items() if not k.startswith("MLXL3_")}
env.update(MLXL3_FLASH_MODEL="/Users/justin/Documents/mix-stq1_0/models/Qwen3.8-Flash-Next-exl3-205-65c89531",MLXL3_SSD_CACHE_GIB="3",MLXL3_SSD_READERS="4",MLXL3_NATIVE_CACHE_SLRU="0",MLXL3_NATIVE_DEVICE_RESIDENT_ROUTE="1",MLXL3_NATIVE_HC_WORKSPACE="0",MLXL3_NATIVE_Q04_SPEED_MTP="1",MLXL3_NATIVE_Q04_MTP_BLOCK="3",MLXL3_NATIVE_MTP_BLOCK_VERIFY="0",DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer")
state={"status":"running","supervisor_pid":os.getpid(),"verify_block_max_rows":3,"steps":[],"standalone_quality_rerun":False}
status=folder/"status.json"
def save():
    temp=status.with_suffix(".tmp");temp.write_text(json.dumps(state,indent=2,allow_nan=False)+"\n");temp.replace(status)
def interrupt(*_):raise KeyboardInterrupt("user interruption")
signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
def conditions():
    data={}
    for args in [["pmset","-g","batt"],["pmset","-g","therm"],["sysctl","vm.swapusage"],["vm_stat"]]:
        p=subprocess.run(args,capture_output=True,text=True,timeout=5);data[" ".join(args)]={"exit":p.returncode,"output":p.stdout+p.stderr}
    return data
try:
    save()
    for depth in [1,2,3,4]:
        for tag,tau in ([('A',None),('B',.1),('C',.2)] if depth%2 else [('C',.2),('B',.1),('A',None)]):
            name=f"B3-D{depth}-{tag}";arm=folder/name;arm.mkdir()
            rec={"name":name,"depth":depth,"tau":tau,"status":"running","before":conditions()};state["steps"].append(rec)
            quiet(rec);save()
            extra={"MLXL3_NATIVE_Q04_MTP_DEPTH":str(depth),"MLXL3_NATIVE_Q04_SPEED_OUTPUT":str(arm/"report.json")}
            if tau is not None:extra["MLXL3_NATIVE_Q04_QUALITY_TAU"]=str(tau)
            with (arm/"test.log").open('w') as f:
                child=subprocess.Popen([str(exe),"gpu::fast_profile_tests::native_q04_ar_generation_pilot","--exact","--ignored","--nocapture","--test-threads=1"],env={**env,**extra},stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                rec["model_pid"]=child.pid;save();start=time.monotonic()
                try:
                    while child.poll() is None:
                        if time.monotonic()-start>300:raise TimeoutError(name)
                        time.sleep(.5);quiet(rec,child.pid)
                    code=child.wait()
                except BaseException:
                    if child.poll() is None:os.killpg(child.pid,signal.SIGKILL)
                    child.wait();rec.update(status="failed_or_interrupted",exit=child.returncode,reaped=True);save();raise
            rec.update(exit=code,reaped=True,after=conditions())
            assert code==0 and "1 passed;" in (arm/"test.log").read_text(),name
            report=json.loads((arm/"report.json").read_text())
            assert report["status"]=="complete_native_mtp_depth_prototype_pilot" and report["mtp"] and report["experimental_mtp"] and report["draft_depth"]==depth and report["verify_block_max_rows"]==3
            old_name={'A':'A1','B':'B1','C':'C1'}[tag]
            old=json.loads((out/"q04-hot-speed-r3"/old_name/"report.json").read_text())
            assert report["prompt_tokens"]==old["prompt_tokens"] and report["checkpoint"]==old["checkpoint"]
            assert len(report["runs"])==2
            for run,prior in zip(report["runs"],old["runs"],strict=True):
                assert run["tokens"]==prior["tokens"] and len(run["tokens"])==64 and run["decode_tokens"]==63
                assert run["text"].strip() and math.isfinite(run["decode_tokens_per_second"]) and run["decode_tokens_per_second"]>0
                assert run["mtp_proposed"]>0 and 0<=run["mtp_accepted"]<=run["mtp_proposed"]
                lengths,widths=run["draft_lengths"],run["verify_widths"]
                assert len(lengths)==len(widths)==4 and lengths[depth-1]>0 and all(n==0 for n in lengths[depth:])
                assert sum((i+1)*n for i,n in enumerate(lengths))==run["mtp_proposed"]
                assert widths[3]==0 and widths[min(3,depth+1)-1]>0
                if tau is not None:assert run["policy"]["retained_layer_rows"]==3024 and run["policy"]["rows"]>=3024 and run["policy"]["swaps"]>0
            m=report["runs"][1]
            rec.update(status="validated_mtp_depth_pilot",tokens_match_same_policy_ar=True,decode_tokens_per_second=m["decode_tokens_per_second"],ttft_seconds=m["ttft_seconds"],total_seconds=m["total_seconds"],proposed=m["mtp_proposed"],accepted=m["mtp_accepted"],draft_lengths=m["draft_lengths"],verify_widths=m["verify_widths"])
            save();print(name,rec["decode_tokens_per_second"],rec["accepted"],rec["proposed"],flush=True)
            with Path("opti.md").open('a') as f:f.write(f"\n- I165-Q04 {name} complete/reaped: {rec['decode_tokens_per_second']:.6f}nativeMTPtok/s,accepted/proposed{rec['accepted']}/{rec['proposed']},actualdraft/verifyhist{rec['draft_lengths']}/{rec['verify_widths']},64warm+64measuredwords matchsame-policyAR. Raw {arm}/report.json; singlepilotnotstablegain.\n")
    state["status"]="complete_mtp_depth_pilot_matrix"
except BaseException as error:
    state.update(status="failed_or_interrupted",error=repr(error))
    if state["steps"] and state["steps"][-1]["status"]=="running":state["steps"][-1]["status"]="failed"
    with Path("opti.md").open('a') as f:f.write(f"\n- I165-Q04 deeperMTPmatrix failed/interrupted {error!r}; {status}; incomplete configs not validated.\n")
    raise
finally:save()
