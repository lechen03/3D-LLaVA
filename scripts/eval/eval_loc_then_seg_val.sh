#!/bin/bash
# Chunked val eval for the [LOC] grounding -> click -> seg pipeline.
# Runs N chunks in parallel (one GPU each), merges, computes metrics.
set -e
cd /home/chenle/3D-LLaVA

CKPT=checkpoints/finetune-3d-llava-lora-grounding
BASE=liuhaotian/llava-v1.5-7b
QFILE=./playground/data/eval_info/referseg_scanrefer/ScanRefer_filtered_val.json
SCAN=./playground/data/scannet/val
# number of nearest superpoints to select as the click (env-overridable, default 1)
NUM_CLICK_SP="${NUM_CLICK_SP:-1}"
OUTDIR=./playground/predictions/finetune-3d-llava-lora-grounding/loc_then_seg_k${NUM_CLICK_SP}
mkdir -p "$OUTDIR"

GPUS=(0 1 2 3)
CHUNKS=${#GPUS[@]}

echo "Launching $CHUNKS chunks on GPUs ${GPUS[*]}..."
for IDX in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[$IDX]} PYTHONPATH=$(pwd) \
    python -m llava.eval.model_scanrefer_loc_then_seg \
        --scan-folder "$SCAN" \
        --model-path "$CKPT" \
        --model-base "$BASE" \
        --question-file "$QFILE" \
        --answers-file "$OUTDIR/${CHUNKS}_${IDX}.jsonl" \
        --temperature 0 --conv-mode vicuna_v1 \
        --num-chunks $CHUNKS --chunk-idx $IDX \
        --num_click_sp "$NUM_CLICK_SP" \
        > "$OUTDIR/chunk_${IDX}.log" 2>&1 &
done
wait
echo "All chunks done. Merging..."

MERGE="$OUTDIR/merge.jsonl"
> "$MERGE"
for IDX in "${!GPUS[@]}"; do cat "$OUTDIR/${CHUNKS}_${IDX}.jsonl" >> "$MERGE"; done

echo "Computing metrics..."
python -c "
import json, numpy as np
rows=[json.loads(l) for l in open('$MERGE') if l.strip()]
ious=np.array([r['iou'] for r in rows])
print(f'=== loc_then_seg val results (num_click_sp=$NUM_CLICK_SP, {len(rows)} samples) ===')
print(f'mean IoU      : {ious.mean():.4f}')
print(f'median IoU    : {np.median(ious):.4f}')
print(f'acc@0.25      : {(ious>=0.25).mean():.2%}')
print(f'acc@0.5       : {(ious>=0.5).mean():.2%}')
print(f'(nonzero IoU samples: {(ious>0).sum()}/{len(rows)})')
"
echo "DONE"
