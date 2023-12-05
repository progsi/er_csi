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
PREAUG=$5

rsupcon_path=$(realpath ../../../../contrastive-product-matching/)

cd $rsupcon_path/src/contrastive

python run_finetune_siamese.py \
	--model_pretrained_checkpoint $rsupcon_path/reports/contrastive/shs100k2_yt-$AUG$BATCH-$LR-$TEMP-roberta-base/pytorch_model.bin \
    --do_train \
	--dataset_name=shs100k2_yt \
    --train_file $rsupcon_path/data/interim/shs100k2_yt/shs100k2_yt-train.json.gz \
	--validation_file $rsupcon_path/data/interim/shs100k2_yt/shs100k2_yt-train.json.gz \
	--test_file $rsupcon_path/data/interim/shs100k2_yt/shs100k2_yt-gs.json.gz \
	--evaluation_strategy=epoch \
	--tokenizer="roberta-base" \
	--grad_checkpoint=False \
    --output_dir $rsupcon_path/reports/finetune/shs100k2_yt-$AUG$BATCH-$LR-$TEMP-roberta-base/ \
	--per_device_train_batch_size=64 \
	--learning_rate=5e-05 \
	--weight_decay=0.01 \
	--num_train_epochs=5 \
	--lr_scheduler_type="linear" \
	--warmup_ratio=0.05 \
	--max_grad_norm=1.0 \
	--metric_for_best_model=loss \
	--dataloader_num_workers=4 \
	--disable_tqdm=True \
	--save_strategy="epoch" \
	--load_best_model_at_end \
	--augment=$AUG \
    #--fp16 \
    #--do_param_opt \