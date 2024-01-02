# CSI/ER on Online Video Platforms

General requirements: a file in `../data/shs100k2_yt.parquet` containing the original SHS100K metadata and YouTube metadata (video title, description, channel name).

## Benchmarks

### rSupCon: Contrastive Product Matching

1. Clone https://github.com/progsi/contrastive-product-matching next to this repo.
2. Install and activate the env from the yaml file: `contrastive-product-matching.yml`
3. Go to `src/baselines/rsupcon` where you find scripts to preprocess and train the baseline.
4. Preprocess: `python preprocess_rsupcon.py`
5. Preprocess for Ditto and HierGAT `python preprocess_ditto_hiergat.py`
6. Pretrain: `bash run_pretraining_clean_roberta.sh 256 5e-05 0.07 all-` with setting the appropriate params (refer to the original repo)
7.  Pretrain: `bash run_finetune_siamese_frozen_roberta.sh 64 5e-05 0.07 all-` with setting the appropriate params (refer to the original repo)

### Magellan

Dropped. We realized that `py_entitymatching` is not suitable for our usecase, because:
- no asymmetric matching is supported (eg. two data sources do not share the same attribute types and numbers of attributes)
- evaluating on the CSI test sets (eg. SHS100K and DaTacos) with mean average precision on N^2 pairs is incredibly slow using the provided catalogues. 


### Ditto


### HierGAT
