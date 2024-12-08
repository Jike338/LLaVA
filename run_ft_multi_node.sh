#!/bin/bash

#SBATCH --account=kpsounis_171
#SBATCH --partition=main
#SBATCH --nodes=2                   
#SBATCH --ntasks=2                  
#SBATCH --ntasks-per-node=1          
#SBATCH --cpus-per-task=1           
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=a100:2       
#SBATCH --output=slurm_out/%x_%j.out

# Generate hostfile from SLURM_NODELIST with slots information
scontrol show hostnames $SLURM_NODELIST | awk '{print $1, "slots=2"}' > hostfile

MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
MASTER_PORT=29500    

cd /scratch1/jikezhon/LLaVA
source ~/miniconda3/bin/activate llavanew

deepspeed --launcher slurm --num_nodes=2 --num_gpus=4 --hostfile=hostfile llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower aim \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-7b-pretrain_aim_clip224preprocessor/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/test_multinode \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
