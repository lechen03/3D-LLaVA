# EXP-OD5: ground-truth bbox injection (scripts/generate_with_od_gt_data.py).
# GPUs 2,3 (0,1 occupied by other users); gradient_accumulation 8 keeps the
# global batch at 2*2*8=32 == baseline == EXP-OD1/OD3 optimization scale.
export CUDA_VISIBLE_DEVICES=2,3

# referring seg
export scanrefer=./playground/data/train_info/scanrefer_train_3d_llava_with_od_gt.json
export multi3drefer=./playground/data/train_info/multi3drefer_train_3d_llava_with_od_gt.json
export nr3d=./playground/data/train_info/nr3d_train_3d_llava_with_od_gt.json

# dense captioning
export scan2cap=./playground/data/train_info/scan2cap_train_3d_llava_with_od_gt.json
export nr3d_caption=./playground/data/train_info/nr3d_caption_train_3d_llava_with_od_gt.json

# vqa
export scanqa=./playground/data/train_info/scanqa_train_3d_llava_with_od_gt.json
export sqa3d=./playground/data/train_info/sqa3d_train_3d_llava_with_od_gt.json

# OD-prior variant of finetune-3d-llava-lora.sh: identical compute config,
# only the 8 data files are swapped to the _with_od versions (object-detection
# text from SpatialLM EXP4 injected after <pc>; see scripts/generate_with_od_data.py).
EXP_NAME=finetune-3d-llava-lora-with-od-gt

CODE_DIR=record/${EXP_NAME}
mkdir -p "$CODE_DIR"
cp -r llava "$CODE_DIR"

PYTHONPATH=$(pwd) \
deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 32 --lora_alpha 64 \
    --deepspeed ./scripts/zero1_3d_llava.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path $scan2cap $scanqa $sqa3d $nr3d_caption $scanrefer $scanrefer $scanrefer $multi3drefer $nr3d \
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
