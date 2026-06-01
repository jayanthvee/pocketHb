"""Print stream outputs of a notebook for quick inspection."""
import json
import sys
from pathlib import Path

nb_path = Path(sys.argv[1] if len(sys.argv) > 1 else "notebooks/03_train.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    cid = c.get("id", "?")
    for out in c.get("outputs", []):
        if out.get("output_type") == "stream":
            print(f"--- cell {i} (id={cid}) ---")
            print("".join(out["text"])[:3000])
        elif out.get("output_type") == "error":
            print(f"--- cell {i} (id={cid}) ERROR ---")
            print(out.get("ename"), out.get("evalue"))
