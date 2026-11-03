from pathlib import Path
import hashlib,json,math,os,selectors,signal,subprocess,sys,time,traceback
import numpy as np
from safetensors import safe_open
from mlxl3_quantizer import list_exl3_modules,validate_checkpoint_files
root=Path.cwd(); evidence=root/'docs/measurements/desktop-v2-beta5'; app=root/'build/beta5-release/mount/MLXL3 Desktop.app';worker=app/'Contents/Resources/quantizer/mlxl3-quantize'
report={'status':'running'}; target=evidence/'packaged-qualification.json'
def save(): target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def exists(pid):
    try: os.kill(pid,0); return True
    except ProcessLookupError: return False
save()
try:
    source=root/'build/beta5-mini/source3-noindex';out=root/'build/beta5-mini/output4-frozen-final'
    before=json.loads((root/'build/beta5-mini/source3-hashes.json').read_text())
    assert {p.name:sha(p) for p in source.iterdir() if p.is_file()}==before
    validate_checkpoint_files(out); modules=list_exl3_modules(out);assert len(modules)==11,len(modules)
    plain='model.layers.1.self_attn.q_proj.weight'; found=None;tensors=0;values=0
    with safe_open(source/'model.safetensors',framework='numpy') as f: original=f.get_tensor(plain)
    for path in out.glob('*.safetensors'):
        with safe_open(path,framework='numpy') as f:
            for key in f.keys():
                array=f.get_tensor(key);assert array.size>0 and np.isfinite(array).all(),key
                tensors+=1;values+=array.size
                if key==plain: found=array
    assert found is not None and found.dtype==original.dtype and np.array_equal(found,original)
    events=[json.loads(s) for s in (evidence/'packaged-mini-final.log').read_text().splitlines() if s.startswith('{')]
    result=events[-1]['result'];assert result['status']=='complete'
    quality=result['quality'];assert 32<=quality['tokens']<=256 and all(math.isfinite(quality[k]) for k in ['ppl','teacher_ppl','kl']) and quality['kl']>=-1e-7
    build=json.loads((app/'Contents/Resources/build-info.json').read_text()); manifest=json.loads((worker.parent/'build-info.json').read_text())
    assert build['commit']==manifest['source'] and not build['tracked_changes']
    assert all(sha(root/p)==digest for p,digest in manifest['files'].items())
    report.update(source=build['commit'],modules=len(modules),checked_tensors=tensors,checked_values=values,source_byte_exact=True,excluded_projection_byte_exact=True,quality=quality,worker_sha256=sha(worker),worker_manifest_sha256=sha(worker.parent/'build-info.json'),worker_source_hashes_match=True)
    p=subprocess.run([worker],input=json.dumps({'operation':'probe','source':'https://huggingface.co/LiquidAI/LFM2.5-2.6B'}).encode(),capture_output=True,timeout=60,env={**os.environ,'MLXL3_OWNER_PID':str(os.getpid())})
    (evidence/'packaged-hf-probe.json').write_bytes(p.stdout);(evidence/'packaged-hf-probe.stderr').write_bytes(p.stderr)
    assert p.returncode==0,p.stderr[-1000:]
    hub=json.loads(p.stdout)['result'];assert len(hub['revision'])==40 and hub['conversion_supported'] and not hub['header_checked']
    report['hub_metadata_only']={'source':hub['source'],'revision':hub['revision'],'source_bytes':hub['source_bytes']}
    child=subprocess.Popen([worker],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        time.sleep(1);child.terminate();stdout,stderr=child.communicate(timeout=8)
        assert child.returncode!=0 and not exists(child.pid)
        report['stop']={'exit':child.returncode,'reaped':True}
    finally:
        if child.poll() is None: child.kill();child.wait(timeout=3)
    readfd,writefd=os.pipe();launcher=None;pid=None
    try:
        code="import os,subprocess,sys,time;p=subprocess.Popen([sys.argv[1]],stdin=int(sys.argv[2]),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env={**os.environ,'MLXL3_OWNER_PID':str(os.getpid())});print(p.pid,flush=True);time.sleep(30)"
        launcher=subprocess.Popen([sys.executable,'-c',code,str(worker),str(readfd)],pass_fds=(readfd,),stdout=subprocess.PIPE,text=True)
        os.close(readfd);readfd=-1
        with selectors.DefaultSelector() as selector:
            selector.register(launcher.stdout,selectors.EVENT_READ);assert selector.select(5),'silent launcher'
            pid=int(launcher.stdout.readline())
        time.sleep(1);assert exists(pid)
        launcher.kill();launcher.wait(timeout=3)
        deadline=time.monotonic()+5
        while exists(pid) and time.monotonic()<deadline: time.sleep(.05)
        assert not exists(pid),'orphan worker while inherited stdin writer stayed open'
        report['parent_death']={'worker_exited_with_stdin_still_open':True,'launcher_reaped':True}
    finally:
        if launcher is not None and launcher.poll() is None: launcher.kill();launcher.wait(timeout=3)
        if pid is not None and exists(pid): os.kill(pid,signal.SIGKILL)
        if readfd>=0: os.close(readfd)
        os.close(writefd)
    report['status']='passed';save();print(json.dumps(report,indent=2))
except BaseException as error:
    report.update(status='failed',error=str(error),traceback=traceback.format_exc());save();raise
