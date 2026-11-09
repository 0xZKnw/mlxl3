import hashlib,json,os,runpy,shutil,signal,subprocess,time
from pathlib import Path
root=Path.cwd();out=root/'build/pr167-audit-20261109'
exe=out/'native-test'
shutil.copy2('/Users/justin/Documents/mlxl3-pr161-validation/target/release/deps/mlxl3_native-cbc0d743b1bbc6fe',exe)
paths=subprocess.check_output(['git','ls-files','native/src','native/shaders','Cargo.toml','Cargo.lock'],text=True).splitlines()
identity={'base_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'local_changes':True,'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'source':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}}
(out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
quiet=runpy.run_path('docs/measurements/slotstream-20261026/supervisor.py')['check_quiet']
env={k:v for k,v in os.environ.items() if not k.startswith('MLXL3_')}
model='/Users/justin/Documents/mix-stq1_0/models/Qwen3.8-Flash-Next-exl3-205-65c89531'
env.update(DEVELOPER_DIR='/Applications/Xcode.app/Contents/Developer',MLXL3_FLASH_MODEL=model,MLXL3_MOE_CHECKPOINT=model,MLXL3_SSD_CACHE_GIB='3',MLXL3_SSD_READERS='4',MLXL3_NATIVE_CACHE_SLRU='0',MLXL3_NATIVE_DEVICE_RESIDENT_ROUTE='1',MLXL3_NATIVE_HC_WORKSPACE='0',MLXL3_NATIVE_DEVICE_GREEDY='1',MLXL3_NATIVE_FLASH_VERIFY_QMV='1',MLXL3_NATIVE_FLASH_VERIFY_SPECIALIZED='1',MLXL3_NATIVE_FLASH_EXPERT_CONSUMERS='1',MLXL3_NATIVE_VERIFY_MB3='1',MLXL3_NATIVE_MTP_BLOCK_VERIFY='0')
state={'status':'running','steps':[],'identity':'identity.json'}
status=out/'physical-status.json'
def save():
    temp=status.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(status)
def interrupt(*_):raise KeyboardInterrupt('user interruption')
signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
steps=[
('widen','gpu::flash::mtp::pilot::recursive_gpu_widen_preserves_every_finite_half_word',{}),
('picker','gpu::route_swap_trace::gpu_picker_matches_reference_without_changing_original_routes',{}),
('mb3-projections','gpu::projection::physical_flash_verify_shared_rows_match_scalar',{'MLXL3_NT2_FIXTURES':'/Users/justin/Documents/mlxl3-pr161-validation/docs/measurements/flash-ar30-20261105/quality/resident-fixtures'}),
('mb3-consumers','gpu::moe::block::tests::block_experts_read_once_and_recover_after_failed_fill',{'MLXL3_NATIVE_DEVICE_RESIDENT_ROUTE':'0','MLXL3_NATIVE_MOE_DEFERRED':'0','MLXL3_NATIVE_MOE_FINISH_MIX':'0'}),
('prefix-cancel','gpu::flash::mtp::pilot::depth_prefix_and_final_cancellation_match_scalar_target',{}),
]
try:
    save()
    for name,test,extra in steps:
        record={'name':name,'test':test,'status':'running'};state['steps'].append(record);quiet(record);save()
        with (out/(name+'.log')).open('w') as log:
            child=subprocess.Popen([str(exe),test,'--exact','--ignored','--nocapture','--test-threads=1'],env={**env,**extra},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record['pid']=child.pid;save();started=time.monotonic()
            try:
                while child.poll() is None:
                    if time.monotonic()-started>300:raise TimeoutError(name)
                    time.sleep(.5);quiet(record,child.pid)
                code=child.wait()
            finally:
                if child.poll() is None:child.kill()
                child.wait();record['reaped']=True;save()
        record['exit']=code
        assert code==0 and '1 passed;' in (out/(name+'.log')).read_text(),name
        record.update(status='passed',elapsed_seconds=time.monotonic()-started);save();print(name,'PASS',flush=True)
    state['status']='passed'
except BaseException as error:
    state.update(status='failed_or_interrupted',error=repr(error))
    if state['steps'] and state['steps'][-1]['status']=='running':state['steps'][-1]['status']='failed'
    raise
finally:save()
