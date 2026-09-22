"""Generate `_with_od` variants of the 3D-LLaVA train_info JSONs.

Injects an object-detection text block into the human turn right after
`<pc>\n`, using SpatialLM EXP4 predictions (playground/data/od_exp4.json):

    "<pc>\n<od>\n{od_text}\n</od>\n{original remainder of the turn}"

Design notes:
- Numbers in the od text are rounded to 2 decimals: the Llama tokenizer
  splits every digit, so full-precision floats cost ~72 tokens/line vs ~47
  rounded — with model_max_length=4096 and scenes up to 53 boxes, only the
  rounded form is guaranteed to fit (right-side collator truncation would
  otherwise clip the tail of the answer).
- The OD block lives in the human turn, so training loss-masks it
  (IGNORE_INDEX): it conditions the model, it is never supervised.
- Scenes with empty OD predictions (2 of 1114) keep their original turn.
- Only conversations[0]["value"] is modified; every other field — including
  the gpt turn — is byte-identical to the source file.
- Source files are never modified. Outputs are `<name>_3d_llava_with_od.json`.

Usage (from anywhere):
    python scripts/generate_with_od_data.py
"""

import glob
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_INFO = os.path.join(REPO, "playground", "data", "train_info")
OD_FILE = os.path.join(REPO, "playground", "data", "od_exp4.json")
NUM_RE = re.compile(r"-?\d+\.\d+")


def round_od(text: str) -> str:
    return NUM_RE.sub(lambda m: f"{float(m.group()):.2f}", text)


def main():
    od = json.load(open(OD_FILE))
    empty_scenes = {k for k, v in od.items() if not v.strip()}
    total = 0
    for src in sorted(glob.glob(os.path.join(TRAIN_INFO, "*_3d_llava.json"))):
        base = os.path.basename(src)
        if "_with_click" in base or "_with_od" in base:
            continue
        data = json.load(open(src))
        n_od = n_empty = 0
        for d in data:
            turn = d["conversations"][0]["value"]
            assert turn.startswith("<pc>\n"), f"{base}: entry {d['id']} lacks <pc>\\n prefix"
            raw = od.get(d["scene_id"], "").strip()
            if raw:
                d["conversations"][0]["value"] = (
                    "<pc>\n<od>\n" + round_od(raw) + "\n</od>\n" + turn[len("<pc>\n"):]
                )
                n_od += 1
            else:
                n_empty += 1  # keep original turn
        dst = src.replace("_3d_llava.json", "_3d_llava_with_od.json")
        with open(dst, "w") as f:
            json.dump(data, f)
        total += len(data)
        print(f"{os.path.basename(dst)}: {len(data)} entries, {n_od} with od, {n_empty} kept original")
    print(f"\nTOTAL entries: {total} (expect 293582), empty-OD scenes: {sorted(empty_scenes)}")


if __name__ == "__main__":
    main()
