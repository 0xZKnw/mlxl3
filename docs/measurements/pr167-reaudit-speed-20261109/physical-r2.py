import hashlib,json,os,runpy,shutil,signal,subprocess,time
from pathlib import Path
root=Path.cwd();out=root/'build/pr167-reaudit-speed-20261109'
exe=out/'native-test'
identity=json.loads((out/'identity.json').read_text())
assert hashlib.sha256(exe.read_bytes()).hexdigest()==identity['binary_sha256']
for name,digest in identity['source'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
helpers=runpy.run_path('docs/measurements/slotstream-20261026/supervisor.py');quiet=helpers['check_quiet'];stop=helpers['stop']
env={k:v for k,v in os.environ.items() if not k.startswith('MLXL3_')}
model='/Users/justin/Documents/mix-stq1_0/models/Qwen3.8-Flash-Next-exl3-205-65c89531'
env.update(DEVELOPER_DIR='/Applications/Xcode.app/Contents/Developer',MLXL3_FLASH_MODEL=model,MLXL3_MOE_CHECKPOINT=model,MLXL3_SSD_CACHE_GIB='3',MLXL3_SSD_READERS='4',MLXL3_NATIVE_CACHE_SLRU='0',MLXL3_NATIVE_DEVICE_RESIDENT_ROUTE='1',MLXL3_NATIVE_HC_WORKSPACE='0',MLXL3_NATIVE_DEVICE_GREEDY='1',MLXL3_NATIVE_FLASH_VERIFY_QMV='1',MLXL3_NATIVE_FLASH_VERIFY_SPECIALIZED='1',MLXL3_NATIVE_FLASH_EXPERT_CONSUMERS='1',MLXL3_NATIVE_VERIFY_MB3='1',MLXL3_NATIVE_MTP_BLOCK_VERIFY='0')
state={'status':'running','steps':[],'identity':'identity.json'}
status=out/'physical-status-r2.json'
def save():
    temp=status.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(status)
def interrupt(*_):raise KeyboardInterrupt('user interruption')
signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
fixture='/Users/justin/Documents/mlxl3-pr161-validation/docs/measurements/flash-ar30-20261105/quality/resident-fixtures'
projection='gpu::projection::physical_flash_verify_shared_rows_match_scalar'
prefix='gpu::flash::mtp::pilot::depth_prefix_and_final_cancellation_match_scalar_target'
steps=[
('picker','gpu::route_swap_trace::gpu_picker_matches_reference_without_changing_original_routes',{'MLXL3_NATIVE_VERIFY_MB3':'0'}),
('projection-mb2',projection,{'MLXL3_NT2_FIXTURES':fixture,'MLXL3_NATIVE_VERIFY_MB3':'0'}),
('projection-mb3-family',projection,{'MLXL3_NT2_FIXTURES':fixture,'MLXL3_NATIVE_MB3_SCOPE':'gdn-qkv','MLXL3_NATIVE_MB3_NT':'1','MLXL3_NATIVE_QMV_THREADS128':'1','MLXL3_NATIVE_FLASH_VERIFY_SPECIALIZED':'0'}),
('projection-q02-deferred',projection,{'MLXL3_NT2_FIXTURES':fixture,'MLXL3_NATIVE_VERIFY_MB3':'0','MLXL3_NATIVE_Q02_PROFILE':'mul1-deferred'}),
('projection-q02-centered',projection,{'MLXL3_NT2_FIXTURES':fixture,'MLXL3_NATIVE_VERIFY_MB3':'0','MLXL3_NATIVE_Q02_PROFILE':'mul1-centered'}),
('experts','gpu::moe::block::tests::block_experts_read_once_and_recover_after_failed_fill',{'MLXL3_NATIVE_VERIFY_MB3':'0','MLXL3_NATIVE_DEVICE_RESIDENT_ROUTE':'0','MLXL3_NATIVE_MOE_DEFERRED':'0','MLXL3_NATIVE_MOE_FINISH_MIX':'0'}),
('prefix-base',prefix,{'MLXL3_NATIVE_VERIFY_MB3':'0'}),
('prefix-chain-family-publication',prefix,{'MLXL3_NATIVE_BLOCK_HC_GDN_CHAIN':'1','MLXL3_NATIVE_MB3_SCOPE':'gdn-qkv','MLXL3_NATIVE_MB3_NT':'1','MLXL3_NATIVE_QMV_THREADS128':'1','MLXL3_NATIVE_MTP_DRAFT_PUBLICATION':'1'}),
]
steps=[(name+"-r2",test,extra) for name,test,extra in steps[2:]]
try:
    save()
    for name,test,extra in steps:
        record={'name':name,'test':test,'settings':extra,'status':'running'};state['steps'].append(record);quiet(record);save()
        with (out/(name+'.log')).open('w') as log:
            child=subprocess.Popen([str(exe),test,'--exact','--ignored','--nocapture','--test-threads=1'],env={**env,**extra},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record['pid']=child.pid;save();started=time.monotonic()
            try:
                while child.poll() is None:
                    if time.monotonic()-started>300:raise TimeoutError(name)
                    time.sleep(.5);quiet(record,child.pid)
                code=child.wait()
            finally:
                if child.poll() is None:stop(child)
                child.wait();record['reaped']=True;save()
        record['exit']=code
        assert code==0 and '1 passed;' in (out/(name+'.log')).read_text(),name
        record.update(status='passed',elapsed_seconds=time.monotonic()-started);save();print(name,'PASS',flush=True)
        with Path('opti.md').open('a') as f:f.write(f'\n- I167 re-audit gate {name}: PASS,1selectedphysicaltest, {record["elapsed_seconds"]:.2f}s; same frozen source/binary, process joined. Log {out}/{name}.log. Not model quality or speed.\n')
    state['status']='passed'
except BaseException as error:
    state.update(status='failed_or_interrupted',error=repr(error))
    if state['steps'] and state['steps'][-1]['status']=='running':state['steps'][-1]['status']='failed'
    raise
finally:save()
