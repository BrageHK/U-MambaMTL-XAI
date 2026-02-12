"""Fix maps_dir paths in all xai_outputs results JSON files by removing ../../../ prefix."""
import json
import re
from pathlib import Path

XAI_ROOT = Path("xai_outputs")
fixed_total = 0

for json_path in sorted(XAI_ROOT.rglob("results.json")):
    with open(json_path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        continue

    fixed = 0
    for entry in data:
        if entry.get("maps_dir") and "../" in entry["maps_dir"]:
            entry["maps_dir"] = re.sub(r"(\.\./)+", "", entry["maps_dir"])
            fixed += 1

    if fixed:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Fixed {fixed} entries in {json_path}")
        fixed_total += fixed

print(f"\nDone. Fixed {fixed_total} maps_dir paths total.")
