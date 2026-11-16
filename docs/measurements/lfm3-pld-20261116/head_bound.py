import json, struct, sys, numpy as np, glob, os
d = sys.argv[1]
def tensors():
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with open(f, "rb") as h:
            n = struct.unpack("<Q", h.read(8))[0]; hdr = json.loads(h.read(n))
        for k, v in hdr.items():
            if k != "__metadata__": yield f, 8 + n, k, v
want = {}
for f, base, k, v in tensors():
    if k.endswith("lm_head.weight") or k.endswith("embedding_norm.weight") or k.endswith("norm.weight") and "layers" not in k:
        want[k] = (f, base, v)
print({k: (v[2]["dtype"], v[2]["shape"]) for k, v in want.items()})
def load(k):
    f, base, v = want[k]; a, b = v["data_offsets"]
    raw = np.fromfile(f, dtype=np.uint8, count=b - a, offset=base + a)
    if v["dtype"] == "BF16":
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(v["shape"])
    return raw.view({"F16": np.float16, "F32": np.float32}[v["dtype"]]).reshape(v["shape"])
hk = [k for k in want if k.endswith("lm_head.weight")][0]
W = load(hk).astype(np.float32)
nk = [k for k in want if not k.endswith("lm_head.weight")]
g = load(nk[0]).astype(np.float32) if nk else np.ones(W.shape[1], np.float32)
rown = np.linalg.norm(W, axis=1)
# int8 per-row absmax
s8 = np.abs(W).max(1, keepdims=True) / 127
d8 = W - np.round(W / s8) * s8
# int4 groupwise 64 asym-free absmax
G = W.reshape(W.shape[0], -1, 64); s4 = np.abs(G).max(2, keepdims=True) / 7
d4 = (G - np.round(G / s4) * s4).reshape(W.shape)
rng = np.random.default_rng(0)
out = {"rows": W.shape[0], "row_norm_median": float(np.median(rown))}
for name, D in (("int8_row", d8), ("int4_g64", d4)):
    dn = np.linalg.norm(D, axis=1)
    out[name] = {"rel_err_median": float(np.median(dn / rown)), "rel_err_p99": float(np.quantile(dn / rown, .99))}
    counts = []
    for t in range(8):
        x = (rng.standard_normal(W.shape[1]).astype(np.float32)) * g
        # emulate a confident prediction: add a component aligned with one row
        tgt = rng.integers(W.shape[0]); x = x + 3.0 * W[tgt] / rown[tgt] * np.linalg.norm(x) / np.sqrt(W.shape[1]) * 8
        L = W @ x; A = (W - D) @ x; e = np.linalg.norm(x) * dn * 1.001 + 1e-3
        thr = np.max(A - e); counts.append(int(np.sum(A + e >= thr)))
        assert np.argmax(L) in np.nonzero(A + e >= thr)[0]
    out[name]["candidates_per_token(synthetic)"] = counts
    out[name]["logit_std_example"] = float(np.std(L)); out[name]["bound_median_example"] = float(np.median(e))
print(json.dumps(out, indent=1))
