# Issue165 — Q02 MTP norm/preparation fusion

**Rejected: no measured MTP application gain.** The optional prototype preserves the numerical target and tested draft outputs, but the first32-token improvement did not survive a focused64-token control. No default promotion or installed-app update. Only this consumer-fusion subvariant of Q02 was tested; deferred affine and other Q02 math variants are not claimed complete.

## Implementation and correctness

Existing1024-thread global norm publishes oneF32factor; a64-thread H128 kernel combines norm/gain/F32-to-Half/Half-Suh preparation. Two dispatches remain. Avoids40KiB normalized intermediary and its80KiB explicit write/read traffic per draft mix, adding4bytes factor. This is not a removal of an SSD read or CPU/GPU boundary.

Independent oldnorm shader fixture from0e319439 is compiled separately;81920preparedHalfwords and81920projectedHalfwords match exactly over signedzero/seeded/mixedpowers/tiny inputs and twoeps values. Actual16fusedpreparations asserted; undersizedinput rejected before submission. Full MTPscalar/B2/EOS/accept/reject/pending/rollback fixture passes with90actualfusedpreparations. Production AR/MTP8warm+measured outputs match hashc10b2a835cb93ad6.

CPU library368passed/136ignored; strictClippy/fmt/Metal+chatCLIbuild pass. Kani verifies actual shape/FP32/u64byteextent admission with allu32shapes/allu64lengths; raw obligations/covers/bounds in q02-r3-kani.log. This is boundedCPU verification, not a Metal/allocator/concurrency proof. NewMetaldebug validation/fullignoredphysical suite/source-mutant rebuild not run. First empty-argument type failure retained; inputextent hardening rebuilt before all final GPU gates.

## Performance and decision

Initial A1/B1/B2/A2, AR+MTP32warm+32measured per mode: MTP~+4.94%, but unchanged AR controls drift4.33% and appear+1.56%faster, so attribution is inconclusive. Raw q02-speed reports preserved.

Pre-registered independent MTP-only A1/B1/B2/A2,64warm+64measured per process, same binary/checkpoint/MAIN87/context4096/greedy/cache3GiB/53slots, no process ceiling, serialquietGPU, power/thermal/pressure captured per arm:

| Arm | Decode tok/s |
|---|---:|
| A1 off | 10.440141 |
| B1 on | 10.409964 |
| B2 on | 10.322914 |
| A2 off | 10.358068 |

Mean 10.399104→10.366439tok/s (-0.314%), controls-0.786%. All64outputs identical hashd7c2f53d11d70633, same generation modes/counters, processes/readers joined. The reduction in buffer traffic does **not** establish an application gain. Remove runtime prototype; preserve exact source/binary identities and all raw negative evidence.

Next requested order: Q04, Q03, then Q05 onward. Q01 remains explicitly withdrawn. No claim that the complete issue165 is implemented.
