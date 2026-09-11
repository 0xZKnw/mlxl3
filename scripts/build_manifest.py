"""Record the exact dependency environment and source revision in a release."""
import importlib.metadata
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys

from mlxl3 import __version__

root = Path(__file__).resolve().parents[1]
info = plistlib.loads((root / 'apps/MLXL3Studio/Resources/Info.plist').read_bytes())
assert info['CFBundleShortVersionString'] == __version__
packages = {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()}
# The bundled source version wins over stale editable-install metadata.
packages['mlxl3'] = __version__
payload = {
    'version': __version__, 'build': info['CFBundleVersion'],
    'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
    'tracked_changes': bool(subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=root)),
    'python': sys.version.split()[0], 'packages': dict(sorted(packages.items())),
    'signing': 'ad-hoc; not notarized',
    'swift_sdk': os.environ.get('MLXL3_MACOS_SDK') or subprocess.check_output(
        ['xcrun', '--show-sdk-path'], text=True).strip(),
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + '\n')
