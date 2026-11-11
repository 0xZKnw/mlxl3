"""Exercise model-card startup with incomplete, delayed server metadata."""

import runpy
import sys
import time
from pathlib import Path

Path(sys.argv[1]).write_text("{")
time.sleep(6)
sys.argv[0] = str(Path(__file__).with_name("desktop-transfer-fixture.py"))
runpy.run_path(sys.argv[0], run_name="__main__")
