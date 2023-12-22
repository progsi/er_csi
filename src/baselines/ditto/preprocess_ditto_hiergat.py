import pandas as pd
import argparse
import os
import csv


COL_TOKEN = "COL"
VAL_TOKEN = "VAL"


def tokenize_df(data):
        
    def get_cols(side):
        return [col for col in data.columns if side in col and "id_" not in col]
        
    def tokenize_side(row, side):
        
        def clean_col_name(x):
            return x.replace("_left", '').replace('_right', '').replace('_', ' ')
        side_cols = get_cols(side)
        tuple_list = [(COL_TOKEN, clean_col_name(col), VAL_TOKEN, row[col]) for col in side_cols]
        side_tokenized = ' '.join([' '.join(t) for t in tuple_list])
        return side_tokenized
    
    def serialize(row):
        return '\t'.join((tokenize_side(row, "_left"), tokenize_side(row, "_right"), str(row.label))).replace('\n', ' ')

    return data.apply(serialize, axis=1)
        
    
def main(dataset_path, val_ids_path, repo_dir):
    
    data = pd.read_json(dataset_path, lines=True)
    val_ids = pd.read_csv(val_ids_path)

    data_train = data.loc[~data.pair_id.isin(val_ids.pair_id)]
    data_val = data.loc[data.pair_id.isin(val_ids.pair_id)]

    train_tokenized = tokenize_df(data_train)
    val_tokenized = tokenize_df(data_val)
    
    # create dirs in Ditto and HierGAT
    path_ditto = os.path.join(repo_dir, "ditto", "data", "shs100k2_yt")
    os.makedirs(path_ditto, exist_ok=True)
    path_hiergat = os.path.join(repo_dir, "hiergat", "data", "shs100k2_yt")
    os.makedirs(path_hiergat, exist_ok=True)
    
    train_tokenized.to_csv(path_ditto + f"{os.sep}train.txt", index=False, header=False, quoting=csv.QUOTE_NONE, quotechar='', escapechar='\\')
    val_tokenized.to_csv(path_ditto + f"{os.sep}valid.txt", index=False, header=False,  quoting=csv.QUOTE_NONE, quotechar='', escapechar='\\')
    train_tokenized.to_csv(path_hiergat + f"{os.sep}train.txt", index=False, header=False,  quoting=csv.QUOTE_NONE, quotechar='', escapechar='\\')
    val_tokenized.to_csv(path_hiergat + f"{os.sep}valid.txt", index=False, header=False,  quoting=csv.QUOTE_NONE, quotechar='', escapechar='\\')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/home/repos/contrastive-product-matching/data/interim/shs100k2_yt/shs100k2_shs_yt_1000-train.json.gz",
                        help="path to json lines source file")
    parser.add_argument("--val_ids", type=str, default="/home/repos/contrastive-product-matching/data/interim/shs100k2_yt/shs100k2_yt_1000-valid.csv",
                        help="path to csv source file with ids")
    parser.add_argument("--repo_dir", type=str, default="/home/repos",
                        help="path to parquet source file")
    args = parser.parse_args()
    main(args.dataset_path, args.val_ids, args.repo_dir)