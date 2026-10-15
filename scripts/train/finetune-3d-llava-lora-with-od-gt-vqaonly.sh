# EXP-OD8b: point cloud + GT OD text + VQA-ONLY training data. Completes the
# mix-matched 2x2: same 53,138-conversation VQA-only mix as vqaonly/textonly/
# with_od_gt_nopc, with both the pc and the GT od block (the two VQA files of
# the EXP-OD5 set). Replicates the OD marginal effect at matched mix
# (with_od_gt_vqaonly vs vqaonly; OD5-vs-baseline measured it at 293k).
# Run AFTER finetune-3d-llava-lora-vqaonly.sh (sequential, same GPUs 0,3).
export CUDA_VISIBLE_DEVICES=0,3

# vqa with GT od block after <pc>
export scanqa=./playground/data/train_info/scanqa_train_3d_llava_with_od_gt.json
export sqa3d=./playground/data/train_info/sqa3d_train_3d_llava_with_od_gt.json

EXP_NAME=finetune-3d-llava-lora-with-od-gt-vqaonly

CODE_DIR=record/${EXP_NAME}
mkdir -p "$CODE_DIR"
cp -r llava "$CODE_DIR"

PYTHONPATH=$(pwd) \
deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 32 --lora_alpha 64 \
    --deepspeed ./scripts/zero1_3d_llava.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path $scanqa $sqa3d \
    --scan_folder ./playground/data/scannet \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --pointcloud_tower ./checkpoints/pc_pretrained/ost-sa-only-llava-align-scannet200.pth \
    --pc_modules_to_finetune alignment_proj hidden_seg_fc \
    --num_pc_tokens 100 \
    --inst_prompt_encoder shared_projector \
    --freeze_pointcloud_tower True \
    --pc_use_link_token False \
    --image_aspect_ratio pad \
    --group_by_task_length_per_batch True \
    --bf16 True \
    --output_dir ./checkpoints/${EXP_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
