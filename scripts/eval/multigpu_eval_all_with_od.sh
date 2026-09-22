# !/bin/bash
# Evaluate finetune-3d-llava-lora-with-od on all 5 benchmarks (4 GPUs),
# with the EXP4 object-detection prior injected into eval prompts
# (<od>...</od> after <image>\n), matching the training-time format exactly.
# Usage: run from the repo root:  bash scripts/eval/multigpu_eval_all_with_od.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$(pwd)

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

EXP_NAME=finetune-3d-llava-lora-with-od
OD_FILE=./playground/data/od_exp4_val.json
RESULTS=./playground/predictions/$EXP_NAME/metrics_all.txt

run_task () {
    local MODULE=$1 TASK_DIR=$2 SCAN_FOLDER=$3 QUESTION_FILE=$4 EXTRA_ARGS=$5 METRIC_CMD=$6
    mkdir -p ./playground/predictions/$EXP_NAME/$TASK_DIR
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.$MODULE \
            --scan-folder $SCAN_FOLDER \
            --model-path checkpoints/$EXP_NAME \
            --model-base liuhaotian/llava-v1.5-7b \
            --question-file $QUESTION_FILE \
            --answers-file ./playground/predictions/$EXP_NAME/$TASK_DIR/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --od-file $OD_FILE \
            --conv-mode vicuna_v1 $EXTRA_ARGS &
    done
    wait
    output_file=./playground/predictions/$EXP_NAME/$TASK_DIR/merge.jsonl
    > "$output_file"
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ./playground/predictions/$EXP_NAME/$TASK_DIR/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done
    echo "===== $TASK_DIR =====" | tee -a $RESULTS
    echo "answers: $(wc -l < $output_file)" | tee -a $RESULTS
    eval "$METRIC_CMD --result-file $output_file" 2>&1 | tee -a $RESULTS
}

mkdir -p ./playground/predictions/$EXP_NAME
> $RESULTS

date | tee -a $RESULTS

run_task model_sqa3d sqa3d ./playground/data/scannet/val \
    ./playground/data/eval_info/sqa3d/sqa3d_test_question.json "" \
    "python llava/eval/eval_sqa3d.py --annotation-file ./playground/data/eval_info/sqa3d/sqa3d_test_answer.json"

run_task model_scanqa scanqa ./playground/data/scannet/val \
    ./playground/data/eval_info/scanqa/scanqa_val_question.jsonl "--temperature 0" \
    "python llava/eval/eval_scanqa.py --annotation-file ./playground/data/eval_info/scanqa/scanqa_val_answer.jsonl"

run_task model_scan2cap densecap_scanrefer ./playground/data/scannet \
    ./playground/data/eval_info/densecap_scanrefer/scan2cap_mask3d_val.json \
    "--mask3d-inst-folder ./playground/data/eval_info/densecap_scanrefer/mask3d_inst_seg" \
    "python llava/eval/eval_scan2cap.py --pred-instance-attribute-file ./playground/data/eval_info/densecap_scanrefer/scannet_mask3d_val_attributes.pt --gt-instance-attribute-file ./playground/data/eval_info/densecap_scanrefer/scannet_val_attributes.pt --annotation-file ./playground/data/eval_info/densecap_scanrefer/scan2cap_val_corpus.json"

run_task model_scanrefer referseg_scanrefer ./playground/data/scannet/val \
    ./playground/data/eval_info/referseg_scanrefer/ScanRefer_filtered_val.json "--temperature 0" \
    "python llava/eval/eval_refer_seg.py"

run_task model_multi3drefer multi3drefer ./playground/data/scannet/val \
    ./playground/data/eval_info/multi3drefer/multi3drefer_val.json "--temperature 0" \
    "python llava/eval/eval_refer_seg.py"

date | tee -a $RESULTS
echo "ALL_EVAL_DONE"
