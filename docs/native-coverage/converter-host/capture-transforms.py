import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from ponyexl3.mlx.hadamard import had_r_128_mlx, preapply_had_left_mlx, preapply_had_right_mlx

root = Path(__file__).parent / "hadamard-reference"
root.mkdir(exist_ok=False)
rng = np.random.default_rng(75)
cases = []
for shape, axis in [
    ((4, 128), 1),
    ((128, 256), 1),
    ((128, 256), 0),
    ((256, 129), 0),
    ((128, 128), 1),
]:
    x = (np.eye(128) if shape == (128, 128) else rng.normal(size=shape)).astype(np.float32)
    cases.append((f"matrix-{len(cases)}", x, axis, None, None))
tiny = np.empty((4, 128), dtype=np.float32)
tiny[0] = np.nextafter(np.float32(0), np.float32(1))
tiny[1] = -tiny[0]
tiny[2] = np.finfo(np.float32).tiny / 4
tiny[3] = -tiny[2]
cases.append(("subnormal", tiny, 1, None, None))
x = rng.normal(size=(128, 256)).astype(np.float32)
su = rng.uniform(0.5, 2, size=128).astype(np.float16).astype(np.float32)
sv = rng.uniform(0.5, 2, size=256).astype(np.float16).astype(np.float32)
cases.append(("basis", x, None, su, sv))
x = rng.normal(size=(192, 256)).astype(np.float16).astype(np.float32)
su = rng.uniform(0.5, 2, size=256).astype(np.float16).astype(np.float32)
cases.append(("activations", x, -1, su, None))
r = {"status": "in_progress", "cases": []}
(root / "report.json").write_text(json.dumps(r))
try:
    for name, x, axis, su, sv in cases:
        np.save(root / f"{name}-input.npy", x)
        a = mx.array(x, dtype=mx.float32)
        if axis is None:
            a = preapply_had_right_mlx(a / mx.array(sv).reshape(1, -1))
            a = preapply_had_left_mlx(a / mx.array(su).reshape(-1, 1))
        elif axis == -1:
            a = had_r_128_mlx(a, pre_scale=mx.array(su), r_scale=1.0)
        else:
            a = (preapply_had_right_mlx if axis == 1 else preapply_had_left_mlx)(a)
        mx.eval(a)
        values = np.asarray(a, dtype=np.float32)
        assert values.shape == x.shape and np.isfinite(values).all()
        np.save(root / f"{name}-output.npy", values)
        if su is not None:
            np.save(root / f"{name}-su.npy", su)
        if sv is not None:
            np.save(root / f"{name}-sv.npy", sv)
        r["cases"].append({"name": name, "axis": axis, "shape": list(x.shape), "count": x.size})
    rms = []
    input_pattern = [-2,-1,-.5,-.25,0,.25,.5,.75,1,1.25,1.5,1.75,2,2.25,2.5,3]
    gain_pattern = [.75,1.25,1.5,2,2.5,1.125,.875,1.75,1.5,1.25,.75,.5,2,1.5,1.25,.875]
    for width in [128,2560,10240]:
        inp = mx.array(np.tile(input_pattern,width//16).astype(np.float32).reshape(1,width),dtype=mx.bfloat16)
        gain = mx.array(np.tile(gain_pattern,width//16).astype(np.float32),dtype=mx.bfloat16)
        y = mx.fast.rms_norm(inp,gain,1e-6)
        x32,g32 = inp.astype(mx.float32),gain.astype(mx.float32)
        wrong = (x32*mx.rsqrt(mx.mean(x32*x32,axis=-1,keepdims=True)+1e-6)*g32).astype(mx.bfloat16)
        mx.eval(y,wrong)
        actual=np.asarray(y.astype(mx.float32)).reshape(-1);bad=np.asarray(wrong.astype(mx.float32)).reshape(-1)
        rms.append({'width':width,'eps':1e-6,'dtype':'bfloat16','first16_f32_bits':actual[:16].view(np.uint32).tolist(),'wrong_one_round_first16_f32_bits':bad[:16].view(np.uint32).tolist(),'unequal':int(np.count_nonzero(actual.view(np.uint32)!=bad.view(np.uint32)))})
    (root/'bf16-rms.json').write_text(json.dumps({'status':'complete','input_pattern':input_pattern,'gain_pattern':gain_pattern,'cases':rms},indent=2)+'\n')
    r["status"] = "complete"
finally:
    (root / "report.json").write_text(json.dumps(r, indent=2) + "\n")
