import pandas as pd
import argparse
import os


def to_interim(output_dir):
    
    interim_path = output_dir.replace("raw", "interim")
    os.makedirs(interim_path, exist_ok=True)

    def split_to_interim(split, json=True):
        interim = pd.DataFrame(pd.read_csv(f"{output_dir}/{split}.csv").apply(
            lambda x: "left_" + x.ltable_id + "#right_" + x.rtable_id, axis=1), columns=["pair_id"])

        if json:
            split_name = "gs" if split == "test" else split 
            interim.to_json(f"{interim_path}/shs100k2_yt-{split_name}.json.gz", compression='gzip', lines=True, orient='records')
        else:
            interim.to_csv(f"{interim_path}/shs100k2_yt-{split}.csv", index=False, header=True)
    
    # interim subdir
    print("Generating interem train...")
    split_to_interim("train",  json=True)
    print("Generating interem test...")
    split_to_interim("test", json=True)
    print("Generating interem valid...")
    split_to_interim("valid", json=False)
    

    
def main(input_file: str, output_dir: str):
    
    def gen_pairs(data, n=10_000, frac_pos=0.15):

        n_cross = 5000 # number of items for crossjoin
        n_pos = int(frac_pos * n) + 1 # number of positive pairs
        n_neg = int((1 - frac_pos) * n) + 1 # number of negative pairs
        
        sample = data[["yt_id", "set_id"]].sample(n_cross)
        cross = pd.merge(sample, sample, how="cross", suffixes=["_a", "_b"])

        pos_pairs = cross.query("(yt_id_a != yt_id_b) & (set_id_a == set_id_b)").sample(n_pos)
        neg_pairs = cross.query("(yt_id_a != yt_id_b) & (set_id_a != set_id_b)").sample(n_neg)
        
        pos_pairs["label"] = 1
        neg_pairs["label"] = 0
        
        dataset = pd.concat(
            [pos_pairs, neg_pairs], 
            ignore_index=True).sample(frac=1) #.drop(["set_id_a", "set_id_b"])
        
        return dataset.rename({"yt_id_a": "ltable_id", "yt_id_b": "rtable_id"}, axis=1)

    os.makedirs(output_dir, exist_ok=True)
    
    data_raw = pd.read_parquet(input_file)    
    
    # cols used for ER task
    rel_cols = ["yt_id", "video_title", "channel_name", "description"]
    
    # save table file
    data_raw[rel_cols].to_parquet(os.path.join(output_dir, "table.parquet"))
    
    # write train pairs
    print("Generating Training Pairs...")
    train_pairs = gen_pairs(
        data_raw.query("split == 'TRAIN'"), 
        n=10_000
        )
    train_pairs.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    
    # write test pairs
    print("Generating Test Pairs...")
    test_pairs = gen_pairs(
        data_raw.query("split == 'TEST'"), 
        n=1_000
        )
    test_pairs.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    # write val pairs
    print("Generating Validation Pairs...")
    val_pairs = gen_pairs(
        data_raw.query("split == 'VAL'"), 
        n=1_000
        )
    val_pairs.to_csv(os.path.join(output_dir, "valid.csv"), index=False)
    
    # to interim subdir
    to_interim(output_dir)
    
    # to filter metadata file
    relevant_yt_ids = train_pairs.ltable_id.to_list() + train_pairs.rtable_id.to_list() + \
        test_pairs.ltable_id.to_list() + test_pairs.rtable_id.to_list() + val_pairs.ltable_id.to_list() + \
            val_pairs.rtable_id.to_list()
    # build path
    handle = output_dir.split('/')[-2]
    processed_output_dir = output_dir.replace('raw', 'processed') + 'contrastive/'
    os.makedirs(processed_output_dir, exist_ok=True)
    # write to processed
    data_raw.query(f"yt_id.isin({relevant_yt_ids})").rename(
        {"yt_id": "id", "set_id": "cluster_id"}, axis=1).to_pickle(
            processed_output_dir + f'{handle}-train.pkl.gz', compression='gzip')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="../../../data/shs100k2_yt.parquet",
                        help="path to parquet source file")
    parser.add_argument("--output_dir", type=str, default="../../../contrastive-product-matching/data/raw/shs100k2_yt/",
                        help="path to parquet source file")
    args = parser.parse_args()
    main(args.input_file, args.output_dir)