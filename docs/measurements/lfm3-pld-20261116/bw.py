import time, statistics, json, mlx.core as mx
GiB = 1 << 30
def bench(name, make, op, nbytes, bufs=4, warm=5, reps=30):
    xs = [make() for _ in range(bufs)]
    mx.eval(*xs)
    for i in range(warm): mx.eval(op(xs[i % bufs]))
    ts = []
    for i in range(reps):
        x = xs[i % bufs]
        t = time.perf_counter(); y = op(x); mx.eval(y); ts.append(time.perf_counter() - t)
        assert mx.all(mx.isfinite(y.astype(mx.float32))).item() if y.dtype != mx.uint32 else True
    gbs = sorted(nbytes / t / 1e9 for t in ts)
    r = {"case": name, "median_GBs": statistics.median(gbs), "p10": gbs[len(gbs)//10], "p90": gbs[9*len(gbs)//10], "median_ms": 1e3*statistics.median(ts)}
    print(json.dumps(r), flush=True); return r
out = []
out.append(bench("sum_f32_1GiB", lambda: mx.random.uniform(shape=(GiB//4,)), lambda x: mx.sum(x), GiB))
out.append(bench("max_f16_1GiB", lambda: mx.random.uniform(shape=(GiB//2,)).astype(mx.float16), lambda x: mx.max(x), GiB))
w_bytes = 128000*2048*2
v = mx.random.uniform(shape=(2048,)).astype(mx.float16); mx.eval(v)
out.append(bench("gemv_f16_head_128000x2048", lambda: (mx.random.uniform(shape=(128000,2048))*0.01).astype(mx.float16), lambda w: w @ v, w_bytes, bufs=6))
out.append(bench("copy_f16_1GiB_rw", lambda: mx.random.uniform(shape=(GiB//2,)).astype(mx.float16), lambda x: x + mx.array(0, dtype=mx.float16), 2*GiB))
json.dump(out, open("/Users/justin/.codex/worktrees/lfm3-prompt-lookup/mix-stq1_0/tmp/pld/bw-m5.json","w"), indent=2)
