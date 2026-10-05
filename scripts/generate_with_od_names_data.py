"""Generate `_with_od_names` variants of the 3D-LLaVA train_info JSONs (EXP-OD3).

Coordinate ablation of EXP-OD1: identical full-scene injection, but the OD
block contains ONLY the deduplicated class names (comma-separated,
first-occurrence order in bbox_N), e.g.

    "<pc>\\n<od>\\nwindow, chair, door, sink\\n</od>\\n{original remainder}"

No coordinates, no bbox ids, no counts -- isolates the "scene object
inventory" signal from geometry.  Same policy as EXP-OD1 otherwise: every
conversation of every task gets the block; 2 empty-OD scenes keep their
original turn; only conversations[0]["value"] is modified.

Also writes the eval-side name lists `od_exp4_names.json` (train scenes) and
`od_exp4_names_val.json` (val scenes) consumable by the existing --od-file
mechanism with zero eval-code changes.

Usage (from repo root):
    python scripts/generate_with_od_names_data.py [--skip-audit]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_INFO = os.path.join(REPO, "playground", "data", "train_info")
OD_FILE = os.path.join(REPO, "playground", "data", "od_exp4.json")
OD_VAL_FILE = os.path.join(REPO, "playground", "data", "od_exp4_val.json")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from od_retrieval import parse_od_lines  # noqa: E402

PC = "<pc>\n"


def names_only(od_text):
    """Deduplicated class names, first-occurrence order, comma-separated."""
    seen, out = set(), []
    for _, cls, _ in parse_od_lines(od_text):
        if cls not in seen:
            seen.add(cls)
            out.append(cls)
    return ", ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-audit", action="store_true", help="skip the token audit")
    args = ap.parse_args()

    # --- name lists (train + val) for the eval-side --od-file mechanism ---
    for src, dst in [
        (OD_FILE, os.path.join(REPO, "playground", "data", "od_exp4_names.json")),
        (OD_VAL_FILE, os.path.join(REPO, "playground", "data", "od_exp4_names_val.json")),
    ]:
        od = json.load(open(src))
        names = {k: names_only(v) for k, v in od.items()}
        with open(dst, "w") as f:
            json.dump(names, f)
        nonempty = sum(1 for v in names.values() if v)
        lens = [len(v.split(", ")) for v in names.values() if v]
        print(f"{os.path.basename(dst)}: {len(names)} scenes ({nonempty} non-empty), "
              f"classes/scene p50={int(np.percentile(lens, 50))} max={max(lens)}")

    od = json.load(open(OD_FILE))

    total = n_od = n_empty = 0
    files = sorted(
        f for f in glob.glob(os.path.join(TRAIN_INFO, "*_3d_llava.json"))
        if "_with_click" not in f and "_with_od" not in f
    )
    for src in files:
        data = json.load(open(src))
        n_file_od = n_file_empty = 0
        for d in data:
            turn = d["conversations"][0]["value"]
            assert turn.startswith(PC), f"{os.path.basename(src)}: entry {d['id']} lacks <pc>\\n"
            names = names_only(od.get(d["scene_id"], ""))
            if names:
                d["conversations"][0]["value"] = (
                    "<pc>\n<od>\n" + names + "\n</od>\n" + turn[len(PC):]
                )
                n_file_od += 1
            else:
                n_file_empty += 1  # keep original turn
        dst = src.replace("_3d_llava.json", "_3d_llava_with_od_names.json")
        with open(dst, "w") as f:
            json.dump(data, f)
        total += len(data); n_od += n_file_od; n_empty += n_file_empty
        print(f"{os.path.basename(dst)}: {len(data)} entries, {n_file_od} with od, {n_file_empty} kept original")

    print(f"\nTOTAL {total} entries | with-od {n_od} | empty {n_empty}")

    if not args.skip_audit:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            os.path.join(REPO, "checkpoints", "finetune-3d-llava-lora"))
        lens = []
        for src in files:
            data = json.load(open(src.replace("_3d_llava.json", "_3d_llava_with_od_names.json")))
            texts = [d["conversations"][0]["value"] + d["conversations"][1]["value"] for d in data]
            for i in range(0, len(texts), 512):
                lens += [len(x) for x in tok(texts[i:i + 512], add_special_tokens=False)["input_ids"]]
        lens = np.array(lens)
        print(f"token audit [all 293k] p50={int(np.percentile(lens, 50))} "
              f"p95={int(np.percentile(lens, 95))} max={lens.max()}")
        assert lens.max() < 4096
        # spot-check one injected turn
        d = json.load(open(files[0].replace("_3d_llava.json", "_3d_llava_with_od_names.json")))[0]
        head = d["conversations"][0]["value"][:160].replace("\n", "\\n")
        print(f"spot-check: {head}")


if __name__ == "__main__":
    main()
