"""Generate `_with_od_gt_nopc` variants of the VQA train_info JSONs (EXP-OD6).

Point-cloud-free ablation of EXP-OD5: the LLM input is ONLY the GT bbox text
plus the question -- no `<pc>` token, no point cloud tensor. The pipeline
train/eval code gets a `--no-pc` switch; the data side is produced here by
stripping the leading `<pc>\n` (5 bytes) from the EXP-OD5-injected files, so
everything else (GT block, question, gpt turn, scene_id, task_type) stays
byte-identical to EXP-OD5.

Only the two VQA tasks are eligible: the other six train files either answer
`[SEG].` (nr3d/sr3d/scanrefer/multi3drefer -- mask supervision needs pc tower
features) or designate the target object with `<loc>` (scan2cap/nr3d_caption
-- the prompt embedding comes from the pc tower's visual sampler).

Reading: OD6 ~= OD5 on scanqa/sqa3d -> the point cloud contributes nothing
for VQA beyond the GT inventory; OD6 clearly lower -> quantifies the pc's
marginal value (expect color/appearance questions to collapse: no RGB source
anywhere in the input).

Usage (from repo root):
    python scripts/generate_with_od_gt_nopc_data.py [--skip-audit]
"""

import argparse
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_INFO = os.path.join(REPO, "playground", "data", "train_info")

PC = "<pc>\n"
SOURCES = [
    "scanqa_train_3d_llava_with_od_gt.json",
    "sqa3d_train_3d_llava_with_od_gt.json",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-audit", action="store_true", help="skip the token audit")
    args = ap.parse_args()

    total = 0
    for name in SOURCES:
        src = os.path.join(TRAIN_INFO, name)
        dst = src.replace("_with_od_gt.json", "_with_od_gt_nopc.json")
        data = json.load(open(src))
        for d in data:
            turn = d["conversations"][0]["value"]
            assert turn.startswith(PC), f"{name}: entry {d['id']} lacks <pc>\\n"
            d["conversations"][0]["value"] = turn[len(PC):]
        with open(dst, "w") as f:
            json.dump(data, f)
        total += len(data)
        print(f"{os.path.basename(dst)}: {len(data)} entries")
    print(f"\nTOTAL {total} entries")
    assert total == 53138, f"expected 53,138 (26,515 scanqa + 26,623 sqa3d), got {total}"

    # --- audits against the EXP-OD5 sources ---
    for name in SOURCES:
        src = json.load(open(os.path.join(TRAIN_INFO, name)))
        dst = json.load(open(os.path.join(TRAIN_INFO,
                   name.replace("_with_od_gt.json", "_with_od_gt_nopc.json"))))
        assert len(src) == len(dst)
        for a, b in zip(src, dst):
            assert a["id"] == b["id"] and a["scene_id"] == b["scene_id"]
            assert a["task_type"] == b["task_type"]
            assert a["conversations"][1:] == b["conversations"][1:], "gpt turn changed"
            ha, hb = a["conversations"][0]["value"], b["conversations"][0]["value"]
            assert hb == ha[len(PC):], "human turn is not the stripped source turn"
            # 15 scanqa entries (scene0154_00) had empty GT in EXP-OD5 and kept
            # their question-only turn; stripping <pc>\n correctly leaves a bare
            # question there (empty inventory -> no od block at eval either).
            assert hb.startswith("<od>\n") or "<od>" not in hb, "unexpected turn head"
            for bad in ("<pc>", "<image>", "<loc>", "<link>"):
                assert bad not in hb, f"{bad} leaked into no-pc turn"
        print(f"audit ok: {name}")

    if not args.skip_audit:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            os.path.join(REPO, "checkpoints", "finetune-3d-llava-lora"))
        lens = []
        for name in SOURCES:
            data = json.load(open(os.path.join(TRAIN_INFO,
                       name.replace("_with_od_gt.json", "_with_od_gt_nopc.json"))))
            texts = [d["conversations"][0]["value"] + d["conversations"][1]["value"]
                     for d in data]
            for i in range(0, len(texts), 512):
                lens += [len(x) for x in tok(texts[i:i + 512], add_special_tokens=False)["input_ids"]]
        lens = np.array(lens)
        print(f"token audit [all 53k] p50={int(np.percentile(lens, 50))} "
              f"p95={int(np.percentile(lens, 95))} max={lens.max()}")
        assert lens.max() < 4096, f"max {lens.max()} exceeds model_max_length"
        d = json.load(open(os.path.join(TRAIN_INFO,
                   SOURCES[0].replace("_with_od_gt.json", "_with_od_gt_nopc.json"))))[0]
        head = d["conversations"][0]["value"][:200].replace("\n", "\\n")
        print(f"spot-check: {head}")


if __name__ == "__main__":
    main()
