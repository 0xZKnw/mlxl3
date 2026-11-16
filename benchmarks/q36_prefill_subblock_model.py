import json, math, random
E, K = 256, 8
def sample(T, sigma, rng):
    if sigma == 0: w = [1.0]*E
    else: w = [math.exp(rng.gauss(0, sigma)) for _ in range(E)]
    counts = [0]*E
    for _ in range(T):
        # 8 distinct experts, weighted without replacement
        chosen = set()
        while len(chosen) < K:
            chosen.add(rng.choices(range(E), weights=w)[0])
        for e in chosen: counts[e] += 1
    work = sum(math.ceil(c/32) for c in counts)
    launched = 2*((T*K)//64 + E + 1)
    return work, launched, sum(1 for c in counts if c)
out = {}
for sigma in (0, 0.5, 1.0):
    for T in (73, 256, 512, 1024, 2048):
        rng = random.Random(1234 + T)
        r = [sample(T, sigma, rng) for _ in range(20)]
        work = sum(x[0] for x in r)/len(r); launched = sum(x[1] for x in r)/len(r)
        out[f"sigma{sigma}_T{T}"] = {"working_subblocks": round(work,1), "launched_subblocks": launched,
            "working_per_token": round(work/T,4), "rows_used_fraction": round(T*K/(32*work),3)}
base = {s: out[f"sigma{s}_T256"]["working_per_token"] for s in (0,0.5,1.0)}
for k,v in out.items():
    s = float(k.split('_')[0][5:]); v["per_token_vs_T256"] = round(v["working_per_token"]/base[s],3)
print(json.dumps(out, indent=1))
