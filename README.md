# CSI/ER on Online Video Platforms
This is a repo to model cover song identification as a multi-modal problem combining existing audio-based approaches with entity resolution (ER).

# Requirements
## YouTube metadata file
Due to copyright reasons, we cannot share the YouTube metadata publicly. Hence, one must download the corresponding YouTube metadata using the corresponding `yt_id`. The resulting file should be put `data/yt_metadata.parquet`, with the following columns (one row is one YouTube video):
- `id`: the YouTube identifier
- `title`: the YouTube video title
- `channel_name`: the channel name 
- `description`: the video description
- `keywords`: the list of keywords
## Repositories 
- Contrastive-Product-Matching (our fork): https://anonymous.4open.science/r/contrastive-product-matching-E49B
- Ditto (our fork): https://anonymous.4open.science/r/ditto-2D11
- HierGAT (our fork): https://anonymous.4open.science/r/HierGAT-148B


# Preprocessing

1. Install and activate the env from the yaml file: `contrastive-product-matching.yml`
2. Go to `src/baselines/rsupcon` where you find scripts to preprocess and train the baseline.
3. Preprocess: `python preprocess_rsupcon.py`
4. Preprocess for Ditto and HierGAT `python preprocess_ditto_hiergat.py`

# S-BERT
To train S-BERT like described in the paper, run: `train_metric_bert.py --model_name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --task tvShort`

# Ditto or HierGAT
1. Modify the config.json in the Ditto/HierGAT repo to contain the respective task
2. Run training as explained in the Ditto/HierGAT repository.

# rSupCon
1. Pretrain: `bash run_pretraining_clean_roberta.sh 256 5e-05 0.07 all-` with setting the appropriate params (refer to the original repo)
2.  Pretrain: `bash run_finetune_siamese_frozen_roberta.sh 64 5e-05 0.07 all-` with setting the appropriate params (refer to the original repo)

# Magellan

Dropped. We realized that `py_entitymatching` is not suitable for our usecase, because:
- no asymmetric matching is supported (eg. two data sources do not share the same attribute types and numbers of attributes)
- evaluating on the CSI test sets (eg. SHS100K and DaTacos) with mean average precision on N^2 pairs is incredibly slow using the provided catalogues. 


