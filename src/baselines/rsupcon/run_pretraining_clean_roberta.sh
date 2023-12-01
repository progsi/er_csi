#!/bin/bash
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --export=NONE
BATCH=$1
LR=$2
TEMP=$3
AUG=$4

rsupcon_path=$(realpath ../../../../contrastive-product-matching/)

cd $rsupcon_path/src/contrastive

python run_pretraining_deepmatcher.py \
    --do_train \
	--dataset_name=shs100k2_yt \
	--clean=True \
    --train_file $rsupcon_path/data/processed/shs100k2_yt/contrastive/shs100k2_yt-train.pkl.gz \
	--id_deduction_set $rsupcon_path/data/interim/shs100k2_yt/shs100k2_yt-train.json.gz \
	--tokenizer="roberta-base" \
	--grad_checkpoint=True \
    --output_dir ../../../reports/contrastive/shs100k2_yt-$AUG$BATCH-$LR-$TEMP-roberta-base/ \
	--temperature=$TEMP \
	--per_device_train_batch_size=$BATCH \
	--learning_rate=$LR \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--lr_scheduler_type="linear" \
	--warmup_ratio=0.05 \
	--max_grad_norm=1.0 \
	# --fp16 \
	--dataloader_num_workers=4 \
	--disable_tqdm=True \
	--save_strategy="epoch" \
	--logging_strategy="epoch" \
	--augment=$AUG \