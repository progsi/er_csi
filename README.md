# CSI/ER on Online Video Platforms

General requirements: a file in `../data/shs100k2_yt.parquet` containing the original SHS100K metadata and YouTube metadata (video title, description, channel name).

## Benchmarks

### rSupCon: Contrastive Product Matching



1. Run `src/preprocessing/rsupcon.py`
2. activate env: `conda activate contrastive-product-matching`
3. Pretrain: `bash run_pretraining_clean_roberta.sh BATCH SIZE LEARNING RATE TEMPERATURE (AUG)` with setting the appropriate params (refer to the original repo)
4. Finetune: 

### Magellan

1. 
2. activate env: `conda activate py_entitymatching`