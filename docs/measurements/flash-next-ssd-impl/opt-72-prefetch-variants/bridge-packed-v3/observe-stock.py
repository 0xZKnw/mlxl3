import json,runpy,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
plan=Path(sys.argv[1])
assert not (plan.parent/'vm-curve.json').exists()
trace=[]
base=Path('docs/measurements/flash-next-ssd-impl')
sampler=base/'opt-49-variance-diagnostic/sample-system.txt'
supervisor=base/'opt-45-audit-prerequisites/run-supervisor.txt'
monitor=base/'opt-43-mtp-route-head/speed-heads/bench-supervisor.txt'
def load(path):
    ns=runpy.run_path(str(path))
    selected=Path(path).resolve()
    if selected==monitor.resolve():
        original=ns['monitor']
        def captured(child,record,parse_vm):
            def parse(text):
                value=parse_vm(text)
                trace.append({'name':record['name'],'epoch':time.time(),'raw':text,'vm_bytes':value.copy()})
                return value
            return original(child,record,parse)
        ns['monitor']=captured
    if selected in {sampler.resolve(),supervisor.resolve()}:
        ns['main'].__globals__['runpy']=SimpleNamespace(run_path=load)
    return ns
try:
    load(sampler)['main'](plan)
finally:
    (plan.parent/'vm-curve.json').write_text(json.dumps({'scope':'system VM, not attribution','samples':trace},indent=2)+'\n')
