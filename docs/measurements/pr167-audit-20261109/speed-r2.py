import hashlib,json,math,os,runpy,signal,subprocess,time
from pathlib import Path
out=Path.cwd()/'build/pr167-audit-20261109';exe=out/'native-test'
identity=json.loads((out/'identity.json').read_text())
assert hashlib.sha256(exe.read_bytes()).hexdigest()==identity['binary_sha256']
for path,digest in identity['source'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
assert json.loads((out/'physical-prefix-status-r2.json').read_text())['status']=='passed'
assert all(r['status']=='passed' for r in json.loads((out/'physical-status.json').read_text())['steps'][:4])
quiet=runpy.run_path('docs/measurements/slotstream-20261026/supervisor.py')['check_quiet']
folder=out/'speed-r2';folder.mkdir(exist_ok=False)
# name, MTP, depth, block, tau, legacy CPU hidden, MB3, complete Q04 oracle
matrix=[('D1-off-A0-resumed',1,1,3,None,0,0,0),('D2-old-B1',1,2,3,None,1,0,0),('D2-mb3-C1',1,2,3,None,0,1,0),('AR-off',0,1,1,None,0,0,0),('scalar-D1-off',1,1,1,None,0,0,0),('D1-q20',1,1,3,.2,0,1,0),('D2-q20',1,2,3,.2,0,1,0),('scalar-D1-q20',1,1,1,.2,0,1,0),('AR-q20-oracle',0,1,1,.2,0,0,1),('AR-q20-fast',0,1,1,.2,0,0,0),('D4-q20',1,4,3,.2,0,1,0),('D3-q20',1,3,3,.2,0,1,0),('D2-gpu-mb2',1,2,3,None,0,0,0),('D2-mb3-C2',1,2,3,None,0,1,0),('D2-old-B2',1,2,3,None,1,0,0),('D1-off-A2',1,1,3,None,0,0,0)]
(folder/'matrix.json').write_text(json.dumps(matrix,indent=2)+'\n')
entry="""
### I167-SPEED-RESUME-20261109 — before resumed measurements
- User confirms continue after the external build interruption. All prior owned children are joined and no compiler/prover/engine is active. Same frozen source/binary and all correctness gates reused; original speed/status.json and D1-off-A1=8.327686 remain preserved. Only interrupted/unstarted variants resume in speed-r2; add one fresh D1 reference A0-resumed because the user pause and external build changed the measurement window. Do not mix the pre-pause A1 into the new rotation means. Other16-arm settings and workload unchanged. Metadata receipts retain only MLXL3 flags and DEVELOPER_DIR; unrelated process environment is omitted.
"""
for root in (Path.cwd(),Path('/Users/justin/Documents/mix-stq1_0')):
    with (root/'opti.md').open('a') as f:f.write(entry)
env={k:v for k,v in os.environ.items() if not k.startswith('MLXL3_')}
env.update(DEVELOPER_DIR='/Applications/Xcode.app/Contents/Developer',MLXL3_FLASH_MODEL='/Users/justin/Documents/mix-stq1_0/models/Qwen3.8-Flash-Next-exl3-205-65c89531',MLXL3_SSD_CACHE_GIB='3',MLXL3_SSD_READERS='4',MLXL3_NATIVE_CACHE_SLRU='0',MLXL3_NATIVE_DEVICE_RESIDENT_ROUTE='1',MLXL3_NATIVE_HC_WORKSPACE='0',MLXL3_NATIVE_DEVICE_GREEDY='1',MLXL3_NATIVE_FLASH_VERIFY_QMV='1',MLXL3_NATIVE_FLASH_VERIFY_SPECIALIZED='1',MLXL3_NATIVE_FLASH_EXPERT_CONSUMERS='1',MLXL3_NATIVE_MTP_BLOCK_VERIFY='0',MLXL3_NATIVE_QMV_SHAPES='0')
state={'status':'running','identity':'../identity.json','steps':[]};status=folder/'status.json'
def save():
    temp=status.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n');temp.replace(status)
def interrupt(*_):raise KeyboardInterrupt('user interruption')
signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
def conditions():
    result={}
    for cmd in [['pmset','-g','batt'],['pmset','-g','therm'],['sysctl','vm.swapusage'],['vm_stat']]:
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=5);result[' '.join(cmd)]={'exit':p.returncode,'output':p.stdout+p.stderr}
    return result
try:
    save()
    for name,mtp,depth,block,tau,cpu,mb3,oracle in matrix:
        arm=folder/name;arm.mkdir();r={'name':name,'status':'running','before':conditions()};state['steps'].append(r);quiet(r);save()
        extra={'MLXL3_NATIVE_Q04_SPEED_OUTPUT':str(arm/'report.json'),'MLXL3_NATIVE_Q04_SPEED_MTP':str(mtp),'MLXL3_NATIVE_Q04_MTP_DEPTH':str(depth),'MLXL3_NATIVE_Q04_MTP_BLOCK':str(block),'MLXL3_NATIVE_MTP_RECURSIVE_CPU':str(cpu),'MLXL3_NATIVE_VERIFY_MB3':str(mb3),'MLXL3_NATIVE_Q04_ORACLE':str(oracle)}
        if tau is not None:extra['MLXL3_NATIVE_Q04_QUALITY_TAU']=str(tau)
        r['settings']={k:v for k,v in {**env,**extra}.items() if k.startswith('MLXL3_') or k=='DEVELOPER_DIR'};save()
        with (arm/'test.log').open('w') as log:
            child=subprocess.Popen([str(exe),'gpu::fast_profile_tests::native_q04_ar_generation_pilot','--exact','--ignored','--nocapture','--test-threads=1'],env={**env,**extra},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            r['pid']=child.pid;started=time.monotonic();save()
            try:
                while child.poll() is None:
                    if time.monotonic()-started>300:raise TimeoutError(name)
                    time.sleep(.5);quiet(r,child.pid)
                code=child.wait()
            finally:
                if child.poll() is None:child.kill()
                child.wait();r['reaped']=True;save()
        r['exit']=code
        assert code==0 and '1 passed;' in (arm/'test.log').read_text(),name
        report=json.loads((arm/'report.json').read_text())
        oldname='C1' if tau is not None else 'A1'
        prior=json.loads((Path('build/issue165-20261108/q04-hot-speed-r3')/oldname/'report.json').read_text())
        assert report['prompt_tokens']==prior['prompt_tokens'] and len(report['prompt_tokens'])==47 and report['checkpoint']==prior['checkpoint']
        assert len(report['runs'])==2 and not report['qmv_observation'] and report['q04_oracle']==bool(oracle)
        for run,old in zip(report['runs'],prior['runs'],strict=True):
            assert run['tokens']==old['tokens'] and len(run['tokens'])==64 and run['decode_tokens']==63 and run['active_context_final']==110
            assert run['text'].strip() and math.isfinite(run['decode_tokens_per_second']) and run['decode_tokens_per_second']>0
            assert run['ssd_records_requested']>0 and run['ssd_bytes_requested']>0
            if report['experimental_mtp']:
                work=run['mtp_work'];assert work['target_retained_rows']==63 and work['draft_forwards']==run['mtp_proposed']
                assert work['target_executed_rows']==sum((i+1)*n for i,n in enumerate(run['verify_widths']))+work['tail_steps']
                assert run['device_hot_submissions']==(47+work['target_executed_rows'])*48
                if cpu:assert work['gpu_conversions']==0
                else:assert work['cpu_conversions']==0
            if tau is not None:
                assert run['policy']['oracle_enabled']==bool(oracle) and run['policy']['retained_layer_rows']==3024
                if not oracle:assert run['policy']['oracle_calls']==0
        measured=report['runs'][1]
        r.update(status='validated_short_pilot',tokens_match_same_policy_ar=True,decode_tokens_per_second=measured['decode_tokens_per_second'],accepted=measured['mtp_accepted'] if mtp else None,proposed=measured['mtp_proposed'] if mtp else None,after=conditions());save()
        print(name,round(r['decode_tokens_per_second'],4),r['accepted'],r['proposed'],flush=True)
        with Path('opti.md').open('a') as f:f.write(f"\n- I167 speed {name}: {r['decode_tokens_per_second']:.6f}tok/s,accepted/proposed={r['accepted']}/{r['proposed']}; both64-token trajectories matchsame-policyAR, actualwork/I/O/countschecked, reaped; raw {arm}/report.json. Shortpilot, aggregatependingcontrols.\n")
    state['status']='complete_short_pilot'
except BaseException as error:
    state.update(status='failed_or_interrupted',error=repr(error))
    if state['steps'] and state['steps'][-1]['status']=='running':state['steps'][-1]['status']='failed_or_interrupted'
    raise
finally:save()
