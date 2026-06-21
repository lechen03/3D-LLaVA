# !/bin/bash

export CUDA_VISIBLE_DEVICES=2,3
export PYTHONPATH=$(pwd)

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

EXP_NAME=finetune-3d-llava-lora-with-click

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_scan2cap \
        --scan-folder ./playground/data/scannet \
        --mask3d-inst-folder ./playground/data/eval_info/densecap_scanrefer/mask3d_inst_seg \
        --model-path checkpoints/$EXP_NAME \
        --model-base liuhaotian/llava-v1.5-7b \
        --question-file ./playground/data/eval_info/densecap_scanrefer/scan2cap_mask3d_val.json \
        --answers-file ./playground/predictions/$EXP_NAME/densecap_scanrefer/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/predictions/$EXP_NAME/densecap_scanrefer/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/predictions/$EXP_NAME/densecap_scanrefer/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python llava/eval/eval_scan2cap.py \
--pred-instance-attribute-file ./playground/data/eval_info/densecap_scanrefer/scannet_mask3d_val_attributes.pt \
--gt-instance-attribute-file ./playground/data/eval_info/densecap_scanrefer/scannet_val_attributes.pt \
--annotation-file ./playground/data/eval_info/densecap_scanrefer/scan2cap_val_corpus.json \
--result-file $output_file \
