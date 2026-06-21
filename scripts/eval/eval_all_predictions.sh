# !/bin/bash

export CUDA_VISIBLE_DEVICES=2,3
export PYTHONPATH=$(pwd)

EXP_NAME=finetune-3d-llava-lora

# print exp name
echo "Evaluating experiment: without object_id"
echo ""

echo "SQA3D:"
python llava/eval/eval_sqa3d.py \
    --annotation-file ./playground/data/eval_info/sqa3d/sqa3d_test_answer.json \
    --result-file ./playground/predictions/$EXP_NAME/sqa3d/merge.jsonl
echo ""

echo "ScanQA:"
python llava/eval/eval_scanqa.py \
    --annotation-file ./playground/data/eval_info/scanqa/scanqa_val_answer.jsonl \
    --result-file ./playground/predictions/$EXP_NAME/scanqa/merge.jsonl
echo ""

echo "Scan2Cap:"
python llava/eval/eval_scan2cap.py \
    --pred-instance-attribute-file ./playground/data/eval_info/densecap_scanrefer/scannet_mask3d_val_attributes.pt \
    --gt-instance-attribute-file ./playground/data/eval_info/densecap_scanrefer/scannet_val_attributes.pt \
    --annotation-file ./playground/data/eval_info/densecap_scanrefer/scan2cap_val_corpus.json \
    --result-file ./playground/predictions/$EXP_NAME/densecap_scanrefer/merge.jsonl
echo ""

echo "ScanRefer:"
python llava/eval/eval_refer_seg.py \
    --result-file ./playground/predictions/$EXP_NAME/referseg_scanrefer/merge.jsonl
echo ""

echo "Multi3DRefer:"
python llava/eval/eval_refer_seg.py \
    --result-file ./playground/predictions/$EXP_NAME/multi3drefer/merge.jsonl
echo ""