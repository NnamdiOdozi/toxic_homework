import json
from pathlib import Path

p = Path("outputs_toxic/metrics.json")
backup = p.with_suffix(".json.bak")

data = json.loads(p.read_text())

# Keep a backup first
backup.write_text(json.dumps(data, indent=2, allow_nan=True))

# Remove poisoned RM metric entries
data.get("rm", {}).pop("metrics", None)
data.get("variables", {}).pop("rm_metrics", None)

# Optional: remove empty rm section
if data.get("rm") == {}:
    data.pop("rm", None)

p.write_text(json.dumps(data, indent=2, allow_nan=True))

print(f"Backed up old metrics to {backup}")
print("Removed rm.metrics and variables.rm_metrics")