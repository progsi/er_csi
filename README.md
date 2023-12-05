# CSI/ER on Online Video Platforms

General requirements: a file in `../data/shs100k2_yt.parquet` containing the original SHS100K metadata and YouTube metadata (video title, description, channel name).

## Benchmarks

### rSupCon: Contrastive Product Matching

1. Clone https://github.com/progsi/contrastive-product-matching next to this repo.
2. Install and activate the env from the yaml file: `contrastive-product-matching.yml`
3. Go to `src/baselines/rsupcon` where you find scripts to preprocess and train the baseline.
1. Preprocess: `python preprocess_rsupcon.py`
3. Pretrain: `bash run_pretraining_clean_roberta.sh BATCH SIZE LEARNING RATE TEMPERATURE (AUG)` with setting the appropriate params (refer to the original repo)
4.  Pretrain: `bash run_finetune_siamese_frozen_roberta.sh BATCH SIZE LEARNING RATE TEMPERATURE (AUG)` with setting the appropriate params (refer to the original repo)

### Magellan

1. 
2. activate env: `conda activate py_entitymatching`