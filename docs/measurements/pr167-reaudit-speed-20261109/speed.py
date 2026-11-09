import hashlib,json,math,os,runpy,signal,subprocess,time
from pathlib import Path
out=Path.cwd()/'build/pr167-reaudit-speed-20261109';exe=out/'native-test'
identity=json.loads((out/'identity.json').read_text())
assert hashlib.sha256(exe.read_bytes()).hexdigest()==identity['binary_sha256']
for path,digest in identity['source'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
assert json.loads((out/'physical-status-r3.json').read_text())['status']=='passed'
assert any(r['name']=='projection-mb3-family-r2' and r['status']=='passed' for r in json.loads((out/'physical-status-r2.json').read_text())['steps'])
assert all(r['status']=='passed' for r in json.loads((out/'physical-status.json').read_text())['steps'][:2])
helpers=runpy.run_path('docs/measurements/slotstream-20261026/supervisor.py');quiet=helpers['check_quiet'];stop=helpers['stop']
folder=out/'speed';folder.mkdir(exist_ok=False)
matrix=json.loads((out/'matrix.json').read_text())
(folder/'matrix.json').write_text(json.dumps(matrix,indent=2)+'\n')
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
    for item in matrix:
        name=item['name'];extra=item['flags'];mtp=extra['MLXL3_NATIVE_Q04_SPEED_MTP']=='1';oracle=False
        tau=extra.get('MLXL3_NATIVE_Q04_QUALITY_TAU');profile=extra['MLXL3_NATIVE_Q02_PROFILE']
        arm=folder/name;arm.mkdir();r={'name':name,'status':'running','before':conditions()};state['steps'].append(r);quiet(r);save()
        extra={**extra,'MLXL3_NATIVE_Q04_SPEED_OUTPUT':str(arm/'report.json')}
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
                if child.poll() is None:stop(child)
                child.wait();r['reaped']=True;save()
        r['exit']=code
        assert code==0 and '1 passed;' in (arm/'test.log').read_text(),name
        report=json.loads((arm/'report.json').read_text())
        if profile=='exact':
            oldname='C1' if tau is not None else 'A1'
            prior=json.loads((Path('build/issue165-20261108/q04-hot-speed-r3')/oldname/'report.json').read_text())
        elif mtp:
            reference='AR-q02-deferred' if profile=='mul1-deferred' else 'AR-q02-centered'
            prior=json.loads((folder/reference/'report.json').read_text())
        else:
            prior=report
        assert report['prompt_tokens']==prior['prompt_tokens'] and len(report['prompt_tokens'])==47 and report['checkpoint']==prior['checkpoint']
        assert len(report['runs'])==2 and not report['qmv_observation'] and report['q04_oracle']==bool(oracle)
        assert report['numerical_profile']==profile and report['detail_collection']==(extra['MLXL3_NATIVE_BENCH_DETAILS']=='1')
        for run,old in zip(report['runs'],prior['runs'],strict=True):
            assert run['tokens']==old['tokens'] and len(run['tokens'])==64 and run['decode_tokens']==63 and run['active_context_final']==110
            assert run['text'].strip() and math.isfinite(run['decode_tokens_per_second']) and run['decode_tokens_per_second']>0
            assert run['ssd_records_requested']>0 and run['ssd_bytes_requested']>0
            if report['experimental_mtp']:
                work=run['mtp_work'];assert work['target_retained_rows']==63 and work['draft_forwards']==run['mtp_proposed']
                assert work['target_executed_rows']==sum((i+1)*n for i,n in enumerate(run['verify_widths']))+work['tail_steps']
                assert run['device_hot_submissions']==(47+work['target_executed_rows'])*48
                assert work['cpu_conversions']==0
            assert run['ssd_target_decode'][0] + run['ssd_draft_decode'][0] == run['ssd_records_requested']
            assert run['ssd_target_decode'][1]+run['ssd_draft_decode'][1] == run['ssd_bytes_requested']
            if not report['detail_collection']:assert run['detail_events_decode']==[0,0]
            if report['block_chain']:assert run['hc_gdn_chain_decode']>0
            if tau is not None:
                assert run['policy']['oracle_enabled']==bool(oracle) and run['policy']['retained_layer_rows']==3024
                if not oracle:assert run['policy']['oracle_calls']==0
        if extra['MLXL3_NATIVE_VERIFY_MB3']=='1':assert report['mb3']['uses'][0]>0
        measured=report['runs'][1]
        r.update(status='validated_short_pilot',tokens_match_same_policy_ar=(profile=='exact' or mtp),numerical_profile=profile,decode_tokens_per_second=measured['decode_tokens_per_second'],accepted=measured['mtp_accepted'] if mtp else None,proposed=measured['mtp_proposed'] if mtp else None,after=conditions());save()
        print(name,round(r['decode_tokens_per_second'],4),r['accepted'],r['proposed'],flush=True)
        with Path('opti.md').open('a') as f:f.write(f"\n- I167 re-audit speed {name}: {r['decode_tokens_per_second']:.6f}tok/s,accepted/proposed={r['accepted']}/{r['proposed']}; two64-token trajectories checked within numerical profile {profile} (AR-Q02 has no independent text oracle), actualwork/I/O/countschecked; quality deferred, reaped; raw {arm}/report.json. Shortpilot, aggregatependingcontrols.\n")
    state['status']='complete_short_pilot'
except BaseException as error:
    state.update(status='failed_or_interrupted',error=repr(error))
    if state['steps'] and state['steps'][-1]['status']=='running':state['steps'][-1]['status']='failed_or_interrupted'
    raise
finally:save()
