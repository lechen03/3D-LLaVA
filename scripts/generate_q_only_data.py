"""Generate `_textonly` variants of the VQA train_info JSONs (EXP-OD7).

Question-only ablation of EXP-OD6: the LLM sees ONLY the question -- no
point cloud (OD6's no_pc path) and no OD block. This completes the 2x2
factorial: baseline (pc only) / OD5 (pc + GT OD) / OD6 (GT OD only) /
OD7 (neither). OD7 vs OD6 is the cleanest training comparison of the
series: same two files, same 53k conversations, same task mix; the only
difference is whether the od block is present.

Data is produced by stripping the leading `<od>\n...\n</od>\n` block from
the EXP-OD6 no-pc files. Invariant (asserted in full, not sampled): the
resulting human turn equals the ORIGINAL `_3d_llava.json` turn minus its
`<pc>\n` prefix. 15 scanqa entries (scene0154_00, empty GT) already lack
the block and pass through unchanged.

Reading: OD7 ~= OD6 -> the od text is redundant given the language prior;
OD7 clearly lower -> quantifies the OD text's causal contribution, expected
to concentrate in How-many counting questions (OD6's +5.4 came from
counting the box list).

Usage (from repo root):
    python scripts/generate_q_only_data.py [--skip-audit]
"""

import argparse
import json
import os
import re

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_INFO = os.path.join(REPO, "playground", "data", "train_info")

PC = "<pc>\n"
OD_RE = re.compile(r"^<od>\n.*\n</od>\n", re.S)
SOURCES = [
    "scanqa_train_3d_llava_with_od_gt_nopc.json",
    "sqa3d_train_3d_llava_with_od_gt_nopc.json",
]
ORIGINALS = [
    "scanqa_train_3d_llava.json",
    "sqa3d_train_3d_llava.json",
]


def strip_od(value):
    return OD_RE.sub("", value, count=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-audit", action="store_true", help="skip the token audit")
    args = ap.parse_args()

    total = 0
    for name in SOURCES:
        src = os.path.join(TRAIN_INFO, name)
        dst = src.replace("_with_od_gt_nopc.json", "_textonly.json")
        data = json.load(open(src))
        for d in data:
            turn = d["conversations"][0]["value"]
            if turn.startswith("<od>\n"):
                d["conversations"][0]["value"] = strip_od(turn)
        with open(dst, "w") as f:
            json.dump(data, f)
        total += len(data)
        print(f"{os.path.basename(dst)}: {len(data)} entries")
    print(f"\nTOTAL {total} entries")
    assert total == 53138, f"expected 53,138, got {total}"

    # --- audits: byte-identity with source and original ---
    for name, orig_name in zip(SOURCES, ORIGINALS):
        src = json.load(open(os.path.join(TRAIN_INFO, name)))
        dst = json.load(open(os.path.join(TRAIN_INFO,
                   name.replace("_with_od_gt_nopc.json", "_textonly.json"))))
        orig = json.load(open(os.path.join(TRAIN_INFO, orig_name)))
        assert len(src) == len(dst) == len(orig)
        for a, b, o in zip(src, dst, orig):
            assert a["id"] == b["id"] == o["id"] and a["scene_id"] == b["scene_id"]
            assert a["conversations"][1:] == b["conversations"][1:], "gpt turn changed"
            hb = b["conversations"][0]["value"]
            assert hb == strip_od(a["conversations"][0]["value"]), "not the stripped source"
            assert hb == o["conversations"][0]["value"][len(PC):], \
                "differs from original turn minus <pc>"
            for bad in ("<od>", "<pc>", "<image>", "<loc>", "<link>"):
                assert bad not in hb, f"{bad} leaked into textonly turn"
        print(f"audit ok (incl. original-invariant): {orig_name}")

    if not args.skip_audit:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            os.path.join(REPO, "checkpoints", "finetune-3d-llava-lora"))
        lens = []
        for name in SOURCES:
            data = json.load(open(os.path.join(TRAIN_INFO,
                       name.replace("_with_od_gt_nopc.json", "_textonly.json"))))
            texts = [d["conversations"][0]["value"] + d["conversations"][1]["value"]
                     for d in data]
            for i in range(0, len(texts), 1024):
                lens += [len(x) for x in tok(texts[i:i + 1024], add_special_tokens=False)["input_ids"]]
        lens = np.array(lens)
        print(f"token audit [all 53k] p50={int(np.percentile(lens, 50))} "
              f"p95={int(np.percentile(lens, 95))} max={lens.max()}")
        assert lens.max() < 4096
        d = json.load(open(os.path.join(TRAIN_INFO,
                   SOURCES[0].replace("_with_od_gt_nopc.json", "_textonly.json"))))[0]
        print(f"spot-check: {d['conversations'][0]['value'][:120]!r}")


if __name__ == "__main__":
    main()
