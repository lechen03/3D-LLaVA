# !/bin/bash
# EXP-OD7: evaluate finetune-3d-llava-lora-textonly (question only -- no
# point cloud, no OD text) on the two VQA benchmarks. Completes the 2x2:
# baseline (pc) / OD5 (pc+GT OD) / OD6 (GT OD, no pc) / OD7 (neither).
# NOTE: no --od-file is passed -- od_map stays empty, no injection; the
# --no-pc flag from EXP-OD6 handles the pc-free path.
# Run with CUDA_VISIBLE_DEVICES=0,3 (GPUs 1,2 busy).
# Usage: run from the repo root:  bash scripts/eval/multigpu_eval_all_textonly.sh

# respect an inherited CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,3}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$(pwd)

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

EXP_NAME=finetune-3d-llava-lora-textonly
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
    ./playground/data/eval_info/sqa3d/sqa3d_test_question.json "--no-pc" \
    "python llava/eval/eval_sqa3d.py --annotation-file ./playground/data/eval_info/sqa3d/sqa3d_test_answer.json"

run_task model_scanqa scanqa ./playground/data/scannet/val \
    ./playground/data/eval_info/scanqa/scanqa_val_question.jsonl "--no-pc --temperature 0" \
    "python llava/eval/eval_scanqa.py --annotation-file ./playground/data/eval_info/scanqa/scanqa_val_answer.jsonl"

date | tee -a $RESULTS
echo "ALL_EVAL_DONE"
