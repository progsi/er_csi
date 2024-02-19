# CSI/ER on Online Video Platforms

General requirements: a file in `../data/shs100k2_yt.parquet` containing the original SHS100K metadata and YouTube metadata (video_title, description, channel_name). The CSI datasets which are in this repo must be in `../data/csi_datasets/`. The repos of the ER models must be cloned:
  - https://github.com/wbsg-uni-mannheim/contrastive-product-matching
  - https://github.com/CGCL-codes/HierGAT
  - https://github.com/megagonlabs/ditto 

# Preprocessing

1. Install and activate the env from the yaml file: `contrastive-product-matching.yml`
2. Go to `src/baselines/rsupcon` where you find scripts to preprocess and train the baseline.
3. Preprocess: `python preprocess_rsupcon.py`
4. Preprocess for Ditto and HierGAT `python preprocess_ditto_hiergat.py`

# Ditto or HierGAT
1. Modify the config.json in the Ditto/HierGAT repo to contain the respective task
2. Run training as explained in the Ditto/HierGAT repository

# rSupCon
1. Pretrain: `bash run_pretraining_clean_roberta.sh 256 5e-05 0.07 all-` with setting the appropriate params (refer to the original repo)
2.  Pretrain: `bash run_finetune_siamese_frozen_roberta.sh 64 5e-05 0.07 all-` with setting the appropriate params (refer to the original repo)

# Magellan

Dropped. We realized that `py_entitymatching` is not suitable for our usecase, because:
- no asymmetric matching is supported (eg. two data sources do not share the same attribute types and numbers of attributes)
- evaluating on the CSI test sets (eg. SHS100K and DaTacos) with mean average precision on N^2 pairs is incredibly slow using the provided catalogues. 


