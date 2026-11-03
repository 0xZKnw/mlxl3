"""Check the reviewed inventory of MLX dependencies during the native migration."""

import argparse
import ast
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

BASELINE = Path("docs/native-coverage/mlx-sites.json")
SOURCE = {
    ".cpp",
    ".h",
    ".hpp",
    ".m",
    ".mm",
    ".metal",
    ".rs",
    ".swift",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yml",
    ".yaml",
    ".patch",
    ".spec",
    ".lock",
}
ROOTS = {"benchmarks", "native", "src", "scripts", "apps", "packaging", ".github"}
MARKERS = re.compile(
    r"\b(?:mx|mlx)::(?:\w+::)*\w+|\b(?:lib(?:mlx|jaccl)\.dylib|mlx\.metallib|"
    r"MLXL3_MLX_ROOT|mlx_enabled|mlx[-_]lm|ponyexl3|mlx)\b|"
    r"\bmlxl3_\w+(?=\s*\()|(?:\bArray|\barray)::\w+"
)
METHOD = re.compile(r"\bpub(?:\([^)]*\))?\s+(?:unsafe\s+)?fn\s+(\w+)\s*\(")
CONFIG_KEY = re.compile(r'"((?:MLXL3|MLX)_[A-Z0-9_]+)"')
SHADER = re.compile(r'include_str!\(\s*"([^"]+\.(?:metal|h))"\s*\)|"(mlxl3_[a-z][a-zA-Z0-9_{}:]*)"')
CALL = re.compile(r"\.\s*(\w+)\s*\(")
PYTHON_DEPS = {"mlx", "mlx_lm", "ponyexl3"}


def python_calls(source: str) -> Counter[str]:
    tree = ast.parse(source)
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                root = item.name.split(".")[0]
                if root in PYTHON_DEPS:
                    aliases[item.asname or root] = item.name if item.asname else root
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in PYTHON_DEPS:
            for item in node.names:
                aliases[item.asname or item.name] = f"{node.module}.{item.name}"
    calls: Counter[str] = Counter()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        parts = []
        target = node.func
        while isinstance(target, ast.Attribute):
            parts.append(target.attr)
            target = target.value
        if isinstance(target, ast.Name) and target.id in aliases:
            calls["call " + ".".join([aliases[target.id], *reversed(parts)])] += 1
    return calls


def inventory(root: Path) -> dict[str, dict[str, int]]:
    paths = (
        subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root
        )
        .decode()
        .split("\0")
    )
    methods = set(METHOD.findall((root / "native/src/array.rs").read_text()))
    sites = {}
    for name in sorted(filter(None, paths)):
        path = Path(name)
        if (
            path.suffix not in SOURCE
            or (len(path.parts) > 1 and path.parts[0] not in ROOTS)
            or "SwiftMath" in path.parts
            or name == "scripts/check_native_coverage.py"
        ):
            continue
        if (root / path).is_symlink():
            sites[name] = {f"source symlink -> {(root / path).readlink()}": 1}
            continue
        if not (root / path).is_file():
            continue
        source = (root / path).read_text()
        counts = Counter(MARKERS.findall(source))
        if name == "native/src/array.rs":
            counts.update("array API " + method for method in METHOD.findall(source))
        if path.suffix == ".py":
            counts.update(python_calls(source))
        if path.suffix == ".rs" and path.parts[:2] == ("native", "src"):
            counts.update("config " + key for key in CONFIG_KEY.findall(source))
            counts.update("shader " + (path or key) for path, key in SHADER.findall(source))
            counts.update("." + method for method in CALL.findall(source) if method in methods)
        if counts:
            sites[name] = dict(sorted(counts.items()))
    return sites


def differences(
    expected: dict[str, dict[str, int]], actual: dict[str, dict[str, int]]
) -> list[str]:
    """post: (len(__return__) == 0) == (expected == actual)"""
    lines = []
    for path in sorted(expected.keys() | actual.keys()):
        old, new = expected.get(path, {}), actual.get(path, {})
        if old == new and (path in expected) != (path in actual):
            lines.append(f"{path}: file inventory changed")
        for name in sorted(old.keys() | new.keys()):
            if old.get(name) != new.get(name):
                lines.append(f"{path}: {name} {old.get(name, 0)} -> {new.get(name, 0)}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--inventory", action="store_true", help="emit inventory for review")
    args = parser.parse_args()
    actual = inventory(args.root)
    if args.inventory:
        print(json.dumps(actual, indent=2, ensure_ascii=False))
        return 0
    changes = differences(json.loads((args.root / BASELINE).read_text()), actual)
    if changes:
        print("\n".join(changes), file=sys.stderr)
        return 1
    print(f"MLX source inventory matches {len(actual)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
