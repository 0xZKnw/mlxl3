"""Record the exact dependency environment and source revision in a release."""
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
info = plistlib.loads((root / 'apps/MLXL3Studio/Resources/Info.plist').read_bytes())
payload = {
    'version': info['CFBundleShortVersionString'], 'build': info['CFBundleVersion'],
    'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
    'tracked_changes': bool(subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=root)),
    'runtime': 'Rust + MLX 0.32.2',
    'signing': 'ad-hoc; not notarized',
    'swift_sdk': os.environ.get('MLXL3_MACOS_SDK') or subprocess.check_output(
        ['xcrun', '--show-sdk-path'], text=True).strip(),
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + '\n')
