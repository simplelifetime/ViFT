#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

MODEL_TYPE=qwen1.5-1.8b

PRETRAIN_DIR=bunny-$MODEL_TYPE-pretrain-siglip
OUTPUT_DIR=bunny-$MODEL_TYPE-Caption+Text-MC3

mkdir -p /nas/data/zkliu/checkpoints-sft-bunny/$OUTPUT_DIR

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 13111 bunny/train/train.py \
    --deepspeed ./script/deepspeed/zero3.json \
    --model_name_or_path /nas/data/zkliu/hf_models/qwen2_5-7b-instruct \
    --model_type $MODEL_TYPE \
    --version qwen2 \
    --data_path /home/zkliu/TMCOT/MM_pretrain_data/all_captions/all_captions.json,/home/zkliu/LLaMA-Factory/generation_results/numina_filtered_qwen2-5_bunny_72b.json \
    --image_folder /nas/data/zkliu/llava_datasets \
    --vision_tower google/siglip-so400m-patch14-384 \
    --pretrain_mm_mlp_adapter /nas/data/zkliu/checkpoints-pretrain-bunny/$PRETRAIN_DIR/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /nas/data/zkliu/checkpoints-sft-bunny/$OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --mm_projector_lr 2e-6 \
    --unfreeze_vision_tower \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none | tee 2>&1 /nas/data/zkliu/checkpoints-sft-bunny/$OUTPUT_DIR/log.txt
