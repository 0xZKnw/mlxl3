from pathlib import Path
import hashlib,json,statistics,shutil
out=Path('build/pr167-reaudit-speed-20261109');dest=Path('docs/measurements/pr167-reaudit-speed-20261109');dest.mkdir(exist_ok=True)
s=json.loads((out/'speed/status.json').read_text());assert s['status']=='complete_short_pilot' and len(s['steps'])==14 and all(r['status']=='validated_short_pilot' and r['reaped'] for r in s['steps'])
identity=json.loads((out/'identity.json').read_text());assert hashlib.sha256((out/'native-test').read_bytes()).hexdigest()==identity['binary_sha256']
changes={p:{'measured':h,'delivery':hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p,h in identity['source'].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h};assert list(changes)==['native/src/flash_profile.rs'],changes
(out/'delivery-source-delta.json').write_text(json.dumps({'post_measurement_changes':changes,'reason':'Only add macOS+direct-metal cfg to two GPU-only test accessors; no GPU function body or benchmark setting change. Affected CPU/Metal compilation/lint rerun; timing binary remains the recorded measured revision.'},indent=2)+'\n')
by={r['name']:r for r in s['steps']};groups=[('AR exact',['AR-clean-A1','AR-clean-A2']),('D2 MB2',['D2-base-A1','D2-base-A2'])]+[(r['name'],[r['name']]) for r in s['steps'] if r['name'] not in ('AR-clean-A1','AR-clean-A2','D2-base-A1','D2-base-A2')];rows=[]
for name,names in groups:
 reports=[json.loads((out/'speed'/n/'report.json').read_text()) for n in names];runs=[r['runs'][1] for r in reports];mtp=reports[0]['mtp'];accepted=sum(v['mtp_accepted'] for v in runs) if mtp else None;proposed=sum(v['mtp_proposed'] for v in runs) if mtp else None
 row={'configuration':name,'arms':names,'responses':len(runs),'decode_tokens_per_second':statistics.mean(v['decode_tokens_per_second'] for v in runs),'decode_seconds_mean':statistics.mean(v['decode_seconds'] for v in runs),'prefill_ttft_seconds_mean':statistics.mean(v['prefill_seconds'] for v in runs),'total_seconds_mean':statistics.mean(v['total_seconds'] for v in runs),'accepted':accepted,'proposed':proposed,'acceptance_percent':100*accepted/proposed if proposed else None,'numerical_profile':reports[0]['numerical_profile'],'target_service_bytes_mean':statistics.mean(v['ssd_target_decode'][1] for v in runs),'draft_service_bytes_mean':statistics.mean(v['ssd_draft_decode'][1] for v in runs),'native_metal_allocated_bytes_max':max(v['memory']['native_metal_allocated_bytes'] for v in runs),'process_footprint_bytes_max':max(v['memory']['process_footprint_bytes'] for v in runs),'process_lifetime_peak_bytes_max':max(v['memory']['process_lifetime_peak_bytes'] for v in runs)};rows.append(row)
ar=rows[0]['decode_tokens_per_second'];base=rows[1]['decode_tokens_per_second'];drift={k:(by[b]['decode_tokens_per_second']/by[a]['decode_tokens_per_second']-1)*100 for k,a,b in [('AR','AR-clean-A1','AR-clean-A2'),('D2','D2-base-A1','D2-base-A2')]}
for r in rows:r['delta_percent_to_AR_or_D2_control']=100*(r['decode_tokens_per_second']/(ar if r['configuration'].startswith('AR') else base)-1)
exact=json.loads((out/'speed/AR-clean-A1/report.json').read_text());same=0
for step in s['steps']:
 report=json.loads((out/'speed'/step['name']/'report.json').read_text());same+=sum(a['tokens']==b['tokens'] for a,b in zip(report['runs'],exact['runs']))
summary={'status':'complete_short_pilot','baseline_sha':identity['base_sha'],'binary_sha256':identity['binary_sha256'],'local_source_changes':True,'quality_q02':'not evaluated, explicitly deferred by user','control_drifts_percent':drift,'rows':rows,'trajectories_equal_exact_ar':same,'total_trajectories':28,'all_trajectories_checked_against_appropriate_policy_profile_ar':True,'general_quality_or_numerical_parity_claim':False};(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
table=['| Configuration | Decode tok/s | Draft acceptance | Accepted/proposed | Decode seconds | Responses |','|---|---:|---:|---:|---:|---:|']
for r in rows:
 acc='—' if r['accepted'] is None else f"{r['acceptance_percent']:.1f}%";counts='—' if r['accepted'] is None else f"{r['accepted']}/{r['proposed']}";table.append(f"| {r['configuration']} | {r['decode_tokens_per_second']:.3f} | {acc} | {counts} | {r['decode_seconds_mean']:.3f} | {r['responses']} |")
table='\n'.join(table)+'\n';(out/'throughput-acceptance.md').write_text(table+'\nAcceptance excludes target bonus tokens. Throughput is the arithmetic mean of measured responses; decode seconds are averaged independently. Warmups are excluded.\n')
chain=next(r for r in rows if r['configuration']=='D2-chain');family=next(r for r in rows if r['configuration']=='D2-chain-mb3-qkv')
readme=f'''# PR167 — second audit speed pilot (2026-11-09)

Baseline `{identity['base_sha']}`, branch `codex/issue165-fast-native`.
Frozen release test binary SHA-256: `{identity['binary_sha256']}`.
Measured source hashes: [identity.json](identity.json). The final CPU-only compile
restriction is identified separately in [delivery-source-delta.json](delivery-source-delta.json).
The timed binary was not rebuilt after that non-GPU guard change; CPU and Metal
compilation/lint were rerun. No application/default/merge/release change.

## Results

{table}
D2 HC→GDN composition: **{chain['delta_percent_to_AR_or_D2_control']:+.2f}%** versus the current D2 MB2 control.
With family-scoped MB3/NT1/128-thread QMV compilation: **{family['delta_percent_to_AR_or_D2_control']:+.2f}%**.
The incremental family/NT/compiler effect is only
{100*(family['decode_tokens_per_second']/chain['decode_tokens_per_second']-1):+.2f}% with one response; it is not established independently.
Neither exceeds current exact AR throughput. D1/Q04 uses a different depth and an
approximate policy, so its percentage versus D2 is not a single-change gain.

Q02 deferred: AR−5.24%, D2−3.12%; centered: AR−8.91%, D2+1.47%.
These prototypes have no demonstrated useful gain as a whole. Retain them as
optional research, with model quality explicitly deferred by the user.

## Protocol

Apple M5 /24GiB, AC power, no thermal/performance warning recorded by pmset.
Qwen3.8-Flash-Next EXL3 ~2.05bpw/head4/MUL1, existing checkpoint artifact
`d6c73ad963bf3f23`; LFU53 /3GiB caches,4 readers, resident routing on,
HC workspace off. Same47-token MAIN prompt, context capacity4096, final active
offset110. Each fresh process generates64 warmup outputs then64 measured outputs,
with63 timed decode steps. Model load is outside response prefill/total time.
Raw reports and summary separate prefill/TTFT, decode, total, Metal allocation and
process footprint. Service bytes are not unique disk reads or NAND traffic.

There are14 arms /28 trajectories. Repeated AR/D2 controls have2 measured responses;
every candidate has1. All28 match the appropriate AR policy/profile reference;
{same}/28 also match exact AR. Both Q02 profiles happen to produce exact-AR words
on this prompt. This does not establish their quality or logits/state parity to
the exact model. Q04 remains its existing approximate20% policy.

Control drift: AR{drift['AR']:+.3f}%, D2{drift['D2']:+.3f}%.
This is a short single-prompt pilot, not a sustained or long-context speed claim.
Do not add historical percentages or promote the small centered-Q02 D2 signal to
a validated gain. Q02+chain composition and exhaustive family ablation are unmeasured.

## Validation and preserved failures

- Release library375 passed /139 ignored. Three new ordinary CPU tests pass.
- Kani0.68 /CBMC6.11:2 actual helpers pass. MB3 family76 checks and2 covers;
  depth481 checks (6 unreachable),1 cover, unwind8. Costs/options/limits/rounds
  symbolic; returned depth never exceeds budget or4. These are bounded CPU
  checks, not Metal/MTP/allocator/concurrency proofs. Initial unstable CLI export
  failure is retained; retry adds the required option.
- Eight explicit physical controls pass: independent F16/F32 picker; ordinary
  MB2 projections; corrected family MB3/NT1/128-thread projections; deferred and
  centered Q02 scalar/B2–B4 projections; expert-consumer failed-fill recovery;
  baseline and composed full-model prefix/state controls. Each projection run
  compares11,538,432 finite FP16 words. Q02 compares its own profile across
  scalar/block execution; independent affine-error and task-quality oracles remain pending.
- Full-model controls compare target hidden/GDN/QSA and draft states, logical
  cache state, forced rejection ranks, EOS, final cancellation, rollback and resume.
- Initial family MB3 control detected a dispatch bug: scoped PSO selection still
  used global MB3 presence for dispatch Y. Unselected MB2 B3 paths omitted row3.
  Dispatch now uses the same per-call predicate as PSO selection. Failed binary
  identity/log/status retained; affected fixture and full-model controls rerun.
  Effective NT observation was also corrected.
- Q02 admission rejected unsupported K7/8 fixtures before the old post-load skip.
  Retry retains all42 nongeneric K2–6 fixtures, the original test's domain, with6
  exclusions/reasons saved. No runtime guard was removed. The timed model has
  no MUL1 K≥7 projection. This admission failure is retained separately.
- Timed runs assert64 nonempty outputs, real work/geometry,63 retained target
  rows, active offset, read drain, service-counter sums and zero detailed events
  in clean mode. MB3 and HC-chain actual use is checked. Every child is bounded,
  joined and monitored for competing builds/provers/engines; no series interruption.
- Final strict Metal/CPU Clippy and formatting pass. The first CPU Clippy failure
  (two GPU-only accessors) and its platform-guard fix remain explicit.
  Focused Python6 passed /44 Linux-only skipped /2 subtests. Prior broad suites
  remain historical; no new full Python suite, quality campaign or CLI/app qualification.

## Reproduction and remaining work

`cargo test --locked --release --lib --features chat --no-run --message-format=json`
builds the test executable. Ordinary run uses `--test-threads=1`.
Kani commands/status/output are in `kani-r2-status.json`, `kani-r2.log`, `kani-r2.json`.
Physical commands/settings are in `physical*.py` and `physical-status*.json`;
throughput is in `matrix.json`, `speed.py`, `speed/status.json` and each raw report.
Xcode `/Applications/Xcode.app/Contents/Developer`, Rust1.98.1, existing Cargo
validation target and Python3.12 environment; no download.

No Q02 quality approval is implied. Existing H128/fusions are reused; full GPU
policy transitions, GPU token chaining, full QSA/PLE grouping and grouped KV repair
remain follow-up. Detailed capture and production CLI/bridge/app qualification
remain unmeasured. This delivery pushes research code/results only.
'''
(out/'README.md').write_text(readme)
for p in out.rglob('*'):
 if p.is_file() and p.suffix in ('.json','.py','.log','.md'):
  target=dest/p.relative_to(out);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
p=Path('docs/measurements/pr167-reaudit-20261109/README.md');t=p.read_text();t=t.replace('Request: implement','**Follow-up:** [runtime checks and speed](../pr167-reaudit-speed-20261109/README.md) now document the later MB3 dispatch fix and measurements. The compilation-only state below is the earlier handoff.\n\nRequest: implement',1);p.write_text(t)
entry=f'''\n### I167-REAUDIT-SPEED-HANDOFF-20261109 — completed / authorized PR push
- COMPLETE14arms/28warm+measured64-token trajectories, all match appropriate AR policy/profile; {same}/28exactAR as well (not Q02 quality or logit/state parity). All children joined, no contention interruption. Current ARcontrol{ar:.6f}tok/s,N2,drift{drift['AR']:+.3f}%; D2control{base:.6f},N2,drift{drift['D2']:+.3f}%. HC→GDN D2{chain['decode_tokens_per_second']:.6f} (+{chain['delta_percent_to_AR_or_D2_control']:.2f}%); +MB3qkv/NT1/128threads{family['decode_tokens_per_second']:.6f} (+{family['delta_percent_to_AR_or_D2_control']:.2f}%). Each candidateN1; promising short-pilot signal, no sustained/general gain claim, MTP still below exactAR. Q02deferredAR9.836364/D2=8.696637; centeredAR9.455192/D2=9.108914tok/s; no demonstrated economic Q02 win as a whole, quality explicitly deferred. All prefills/TTFT/decode/total, acceptance/work/resources and individual repetitions archived in docs/measurements/pr167-reaudit-speed-20261109.\n- Validation:375ordinarypassed/139ignored;2actual bounded Kani helpers pass (family76checks+2covers; depth481checks/6unreachable+1cover,unwind8);8physicalcontrols pass incl11,538,432 finite projection words perrun and full target/draft/cache prefix/reject/EOS/cancel/resume. Initial Kani CLI, MB3 grid regression, unsupported-Q02-fixture admission and CPU Clippy failures are retained. Strict Metal/CPU Clippy final PASS; Pythonfocused6passed/44Linux-onlyskipped/2subtests; formatting/inventory/whitespace checked. Prior broad suites historical; full Python, Q02 quality, independent affine-error, CLI/bridge/app not requalified.\n- User explicitly authorizes code/results push to existing draftPR167. Frozen measured source/binary identities retained; final source differs only by two GPU-helper cfg restrictions (no GPU body change), separately recorded; rerun affected CPU/Metal compilation/lint only. No extra unchanged speed repetition. Preserve all old negatives/Q01 withdrawal. All experiments CLOSED, no owned GPU/model/compiler/prover after checks; no default/app/merge/release.\n'''
for root in (Path.cwd(),Path('/Users/justin/Documents/mix-stq1_0')):
 with (root/'opti.md').open('a') as f:f.write(entry)
print(table)
