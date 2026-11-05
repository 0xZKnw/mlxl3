# T18a — dynamic SSD readers, final local decision — 31 October 2026

Status: **numerical/integration quality validated; performance mixed; retained locally as opt-in, Static remains the default**. No app replacement, commit, push or publication. Enable the candidate with `MLXL3_SSD_DYNAMIC_READERS=1`; explicit0 or an absent option preserves Static. Dynamic only applies with more than one worker and more jobs than workers. Cache stays3GiB,48×53slots,4readers.

| Mode | Campaign | Reference → candidate decode tok/s | Observed decode | Observed prefill |
| --- | --- | ---: | ---: | ---: |
| AR | w1, before reboot | 10.435302 →10.659297 | +2.1465% | +1.9158% |
| MTP depth1 | w2, after reboot | 10.588494 →10.899938 | +2.9413% | +3.4376% |
| DFlash mode1/width1 | w2, after reboot | 11.621159 →11.428779 | −1.6554% | +1.3020% |

These are descriptive pilot results, not a guaranteed causal effect. No modes or campaigns are pooled. Each completed mode used ACCA,2independent processes per arm,128output tokens/127decode transitions,MAIN87andHOLD82input tokens,one warmup then one measured response per prompt per process. Decode aggregation is total transitions/total decode_seconds; prefill is total prompt_tokens/total prefill_seconds. MTP reference MAIN drift−8.0478% and DFlash candidate HOLD drift−6.4254% remain included, following the user's tolerance for practical variation; no repeat campaign was launched solely to reduce noise.

[AR data](pilot128-AR-inspection.md), [MTP data](postreboot-w2/pilot128-MTP-inspection.md), [DFlash data](postreboot-w2/pilot128-DF-inspection.md), [machine-readable summary](final-summary.json).

## Adaptation audit and decision

[Final adaptation audit](postreboot-w2/adaptation-final-audit.md) found no concrete new copy/allocation/fence, fallback or queue-port defect. Numerical routes,logical read demand and DFlash counters agree. The slow DFlash HOLD adds0native process decompressions over MAIN→HOLD; endpoint compression and system compressor growth do not identify the cause. Release receipts do not contain a per-batch Dynamic histogram or CAS retries, so their rate and exclusive cost remain unmeasured.

Keep the option for further use in AR/MTP, with Static as the general default given mixed DFlash results. This does not reject all dynamic scheduling or claim a reproducible DFlash regression. No extra model run is justified solely by the noisy row. The independent controller hashing fix is retained separately:529identicalSHA results,~205.6MB avoided parent RSS in the CPU comparison; no model tok/s gain attributed to it.

## Identity and conditions

- GitHEAD7978fceac30d991a8c06d72114fc932b2443f555, pre-existing local changes preserved; [final r3 identity](final-identity-r3.json) and [w2 identity541](postreboot-w2/identity.json) identify sources/dependencies,plans and binaries. The11candidate source snapshots match the retained live source.
- CLI SHA256`1fea100a7aaf30a9123865f1983ce9f31983c2efed6c0b9d6fdc45c10d5de436`; libtest`de3616e3d2e0c3d9507e075cf9af4dfe8b3290f0b291ac704b3e1df537f8d910`. Same CLI for A/C. CLI predates cfg(test)-only stale-timing regression additions; release tokens were unaffected, libtest was rebuilt r2. Details in identity.
- AppleM5,24GiB; MLX0.32.2; Python3.12.14; Rust1.98.1. Target`Qwen3.8-Flash-Next-exl3-205-65c89531`; draft`Qwen3.8-Flash-Next-DFlash2-a22a1744`. Full model/oracle paths and options in manifests. T17/T05/T04=0,Hot/Wide/Unique=1,lookahead0,MLX40ops/40MB,allocator64MiB,context4096,draftQ4/ring1,no prompt cache reuse.
- w1 stopped after1.5MiB new global Swapout during MTP4A warmup:0complete warmups/measurements,outer process joined−15,inner join flags unfilled. [Interrupted record](interruption-before-system-update.json) preserves the failure and the misleading stale max-swap field. w1 AR remains historical; w1 partial MTP is not pooled with w2.
- User explicitly paused for a system update/reboot. New boot31Oct14:38:43Europe/Paris,macOS27.2 build26B123,AC power; previous OS build not preserved in that checkpoint. New w2 anchor was prospectively recorded after3quiet snapshots/20s; counters reset to0by reboot. w2 kept0newSwapout,Swapin≤256MiB,RSS≤10GiB,native footer peaks<12GB,one GPU model/no concurrent compiler or inference.
- Admission10.8GB refusals remained before any model PID. After audit of actual historical native peak7.897GB,prospective w2 admission10.5GB applied to all3quality and8pilot processes before any w2 model/timing; old plans preserved. [Admission r5](postreboot-admission-r5.json):10.621/10.644/10.670GB. This heuristic is not a guarantee against swap. No further automatic lowering.

## Verification and limits

- New ordinary final CPU suite:`cargo test --locked --release --no-default-features -- --test-threads=1`:256passed,0failed,8ignored; fmt passed. [Receipts](final-cpu-plan-result.json).
- New ordinary final MLX/chat suite:`cargo test --locked --release --features mlx,chat -- --test-threads=1`,MLXL3_MLX_ROOT pinned,CARGO_BUILD_JOBS=1:293passed,0failed,187ignored. Ignored physical/model tests were not selected by this command. [Receipts](final-mlx-plan-result.json).
- w2 explicitly selected physical model tests:3/3passed,0ignored; independent logits/states,five positions×248320logits,tenGDN state arrays×786432,MTPbudgets1/2/3/17,DF36prefixes,widths1..8,64000capture values. [Quality audit](postreboot-w2/quality-inspection.md). Eight w2 pilot processes/32responses(warmups included) passed full external golden128 and mode/counter/memory/process checks. Same code's w1 storage/components/model tests17executions/13distinctselectors remain attached to original receipts,not rerun needlessly.
- Unchanged compatible CPU evidence retained:reader308cases underASan+UBSan/TSan; queue4084concurrentcases+4096finitepolicycases underboth; isolated CAS+2 mutation detected. Pythonprofile/export233tests,controller44tests,inspector17syntheticcases; strictClippyCPU/MLX/build passed before reboot with source identities unchanged. These are tests/finite checks,not a full proof.
- Source verification gaps preserved:CBMC on real C++queue harness≤8jobs/4workers fails standard-library parsing before obligations; Kani actual5job/4readerMLX FFI harness(unwind25) encounters compilerICEintrinsics243 before obligations; CrossHair unavailable. No Metal/concurrency/full-generation formal proof. No remote CI run of this local dirty state; inspected PR/main workflows cover portable queue/reader checks but not physical models.

## Startup settings and RAM diagnosis

On explicit user request, Steam's login item was removed and registered Ollama,Steamclean and both Epic launch agents disabled and unloaded. No matching processes remained. Backup:`/Users/justin/Library/Application Support/Codex/startup-backups/20261031-145011/changes.json`. Persistence after the next reboot has not been tested.

[Memory snapshot](postreboot-memory-explanation.json) measured~3.03GBfree+8.12GBfile-backed recoverable pages,0swap; approximate topMEM of ChatGPT/Codex descendants~3.80GB. Process topMEM sums are not unique physical-page totals; file-backed pages overlap active/inactive categories. Admission is a campaign heuristic,not engine minimum RAM.
