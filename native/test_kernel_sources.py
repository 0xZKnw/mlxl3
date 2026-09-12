"""A Python kernel edit must explicitly update the corresponding native port."""
import ast
from pathlib import Path


def test_native_qmv_shader_bodies_match_production():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "src/mlxl3/kernels/qmv.py").read_text())
    names = {"_qmv_tile_kernel", "_qmv_inner_kernel", "_qmv_mapped_tile_kernel"}
    checked = set()
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef) or function.name not in names:
            continue
        for call in ast.walk(function):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "metal_kernel"):
                continue
            body = ast.literal_eval(next(k.value for k in call.keywords if k.arg == "source"))
            native = (root / "native/shaders" / f"{function.name}.metal").read_text()
            assert native.split("\n", 2)[2].rstrip() == body.rstrip()
            checked.add(function.name)
    assert checked == names
