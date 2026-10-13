"""Regenerate standalone examples from the one release source. Run from any directory."""
from pathlib import Path
import json, re, shutil
ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "pycombs_v16r.py").read_text()
for item in json.loads((ROOT / "examples.json").read_text()):
    target = ROOT / item["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    code, count = re.subn(r'^DEFAULT_STARTUP_PRESET = ".*"$',
                         'DEFAULT_STARTUP_PRESET = ' + json.dumps(item["preset"]),
                         source, count=1, flags=re.MULTILINE)
    assert count == 1
    target.write_text(code)
for group in {item["group"] for item in json.loads((ROOT / "examples.json").read_text())}:
    for name in ("fig2c_benchmark.py", "paper_fig2c_reference.png"):
        shutil.copy2(ROOT / name, ROOT / "Examples" / group / name)
print("Regenerated 10 standalone examples from the common v16r engine.")
