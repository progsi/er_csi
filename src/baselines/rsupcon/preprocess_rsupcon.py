import pandas as pd
import argparse
import os


def to_interim(output_dir, data_raw, train_pairs, test_pairs, val_pairs, n_pos_pairs):
    
    def gen_pair_id(x):
        return "left_" + x.ltable_id + "#right_" + x.rtable_id
        
    interim_path = output_dir.replace("raw", "interim")
    os.makedirs(interim_path, exist_ok=True)

    def to_interim_csv():
        
        interim = pd.DataFrame(
            pd.read_csv(f"{output_dir}valid.csv").apply(gen_pair_id, axis=1), columns=["pair_id"])
        interim.to_csv(f"{interim_path}/shs100k_{n_pos_pairs}-valid.csv", index=False, header=True)
    
    def to_interim_json():
        
        def hell_of_a_join(pairs, mode="shs_side"):
            
            if mode == "r":
                dataset = pd.merge(
                        pd.merge(
                        data_raw[["set_id", "yt_id", "title", "performer", "video_title", "channel_name", "keywords", "description"]].rename(
                            {"set_id": "cluster_id", "yt_id": "id"}, axis=1), 
                        pairs, 
                        how="right", 
                        left_on=["cluster_id", "id"], 
                        right_on=["set_id_a", "ltable_id"]),
                        data_raw[["set_id", "yt_id", "title", "performer", "video_title", "channel_name", "keywords", "description"]].rename(
                            {"set_id": "cluster_id", "yt_id": "id"}, axis=1),
                        how="left",
                        left_on=["set_id_b", "rtable_id"],
                        right_on=["cluster_id", "id"],
                        suffixes=["_left", "_right"]
            )
                dataset["title_right"] = '[MASK]'
                dataset["performer_right"] = '[MASK]'
            elif mode == "x":

                dataset = pd.merge(
                        pd.merge(
                        data_raw[["set_id", "yt_id", "title", "performer", "video_title", "channel_name", "keywords", "description"]].rename(
                            {"set_id": "cluster_id", "yt_id": "id"}, axis=1), 
                        pairs, 
                        how="right", 
                        left_on=["cluster_id", "id"], 
                        right_on=["set_id_a", "ltable_id"]),
                        data_raw[["set_id", "yt_id", "title", "performer", "video_title", "channel_name", "keywords", "description"]].rename(
                            {"set_id": "cluster_id", "yt_id": "id"}, axis=1),
                        how="left",
                        left_on=["set_id_b", "rtable_id"],
                        right_on=["cluster_id", "id"],
                        suffixes=["_left", "_right"]
            )
                
                dataset["title_right"] = '[MASK]'
                dataset["performer_right"] = '[MASK]'

                dataset["video_title_left"] = '[MASK]'
                dataset["channel_name_left"] = '[MASK]'
                dataset["description_left"] = '[MASK]'
                dataset["keywords_left"] = '[MASK]'

            elif mode == "sv":
                dataset = pd.merge(
                        pd.merge(
                        data_raw[["set_id", "yt_id", "title", "performer"]].rename(
                            {"set_id": "cluster_id", "yt_id": "id"}, axis=1), 
                        pairs, 
                        how="right", 
                        left_on=["cluster_id", "id"], 
                        right_on=["set_id_a", "ltable_id"]),
                        data_raw[["set_id", "yt_id", "video_title", "channel_name", "keywords", "description"]].rename(
                            {"set_id": "cluster_id", "yt_id": "id"}, axis=1),
                        how="left",
                        left_on=["set_id_b", "rtable_id"],
                        right_on=["cluster_id", "id"],
                        suffixes=["_left", "_right"]
            )
                dataset = dataset.rename({"title": "title_left", "performer": "performer_left", 
                         "video_title": "video_title_right", 
                         "channel_name": "channel_name_right", 
                         "keywords": "keywords_right",
                         "description": "description_right"}, axis=1)
            elif mode == "vv":
                dataset = pd.merge(
                        pd.merge(
                        data_raw[["set_id", "yt_id", "video_title", "channel_name", "keywords", "description"]].rename(
                            {"set_id": "cluster_id", "yt_id": "id"}, axis=1), 
                        pairs, 
                        how="right", 
                        left_on=["cluster_id", "id"], 
                        right_on=["set_id_a", "ltable_id"]),
                        data_raw[["set_id", "yt_id", "video_title", "channel_name", "keywords", "description"]].rename(
                            {"set_id": "cluster_id", "yt_id": "id"}, axis=1),
                        how="left",
                        left_on=["set_id_b", "rtable_id"],
                        right_on=["cluster_id", "id"],
                        suffixes=["_left", "_right"]
            )
            dataset["pair_id"] = dataset.apply(gen_pair_id, axis=1)
            return dataset
        
        print("Generating rich representation...")
        interim_train = hell_of_a_join( # somehow, the original repo requires train AND val here.
            pd.concat([train_pairs, val_pairs], axis=0, ignore_index=True), mode="r"
            )
        interim_train.to_json(f"{interim_path}/shs100k_rLong-train.json.gz", lines=True, 
                              compression='gzip', orient='records')
        interim_train.drop(["description_left", "description_right", "keywords_left", "keywords_right"], axis=1).to_json(f"{interim_path}/shs100k_rShort-train.json.gz", lines=True, 
                              compression='gzip', orient='records')
        interim_test = hell_of_a_join(test_pairs, mode="r")
        interim_test.to_json(f"{interim_path}/shs100k_rLong-gs.json.gz", lines=True, 
                              compression='gzip', orient='records')
        interim_test.drop(["description_left", "description_right", "keywords_left", "keywords_right"], axis=1).to_json(f"{interim_path}/shs100k_rShort-gs.json.gz", lines=True, 
                              compression='gzip', orient='records')
        
        print("Generating rich cross representation...")
        interim_train = hell_of_a_join( # somehow, the original repo requires train AND val here.
            pd.concat([train_pairs, val_pairs], axis=0, ignore_index=True), mode="x"
            )
        interim_train.to_json(f"{interim_path}/shs100k_xLong-train.json.gz", lines=True, 
                              compression='gzip', orient='records')
        interim_train.drop(["description_left", "description_right", "keywords_left", "keywords_right"], axis=1).to_json(f"{interim_path}/shs100k_xShort-train.json.gz", lines=True, 
                              compression='gzip', orient='records')
        interim_test = hell_of_a_join(test_pairs, mode="r")
        interim_test.to_json(f"{interim_path}/shs100k_xLong-gs.json.gz", lines=True, 
                              compression='gzip', orient='records')
        interim_test.drop(["description_left", "description_right", "keywords_left", "keywords_right"], axis=1).to_json(f"{interim_path}/shs100k_xShort-gs.json.gz", lines=True, 
                              compression='gzip', orient='records')
        
        print("Generating interim train video-video...")
        interim_train = hell_of_a_join( # somehow, the original repo requires train AND val here.
            pd.concat([train_pairs, val_pairs], axis=0, ignore_index=True), mode="vv"
            )
        interim_train.drop(["keywords_left", "keywords_right"], axis=1).to_json(f"{interim_path}/shs100k_vvLong-train.json.gz", lines=True, 
                              compression='gzip', orient='records')
        interim_train.drop(["description_left", "description_right"], axis=1).to_json(f"{interim_path}/shs100k_vvShort+Tags-train.json.gz", lines=True, 
                              compression='gzip', orient='records')
        interim_train.drop(["description_left", "description_right", "keywords_left", "keywords_right"], axis=1).to_json(f"{interim_path}/shs100k_vvShort-train.json.gz", lines=True, 
                              compression='gzip', orient='records')
        
        print("Generating interim test video-video...")
        interim_test = hell_of_a_join(test_pairs, mode="vv")
        interim_test.drop(["keywords_left", "keywords_right"], axis=1).to_json(f"{interim_path}/shs100k_vvLong-gs.json.gz", lines=True, 
                              compression='gzip', orient='records')
        interim_test.drop(["description_left", "description_right"], axis=1).to_json(f"{interim_path}/shs100k_vvShort+Tags-gs.json.gz", lines=True, 
                              compression='gzip', orient='records')
        interim_test.drop(["description_left", "description_right", "keywords_left", "keywords_right"], axis=1).to_json(f"{interim_path}/shs100k_vvShort-gs.json.gz", lines=True, 
                              compression='gzip', orient='records')
        
        print("Generating interim train song-video...")
        interim_train = hell_of_a_join( # somehow, the original repo requires train AND val here.
            pd.concat([train_pairs, val_pairs], axis=0, ignore_index=True), mode="sv"
            )
        interim_train.drop(["keywords_right"], axis=1).to_json(f"{interim_path}/shs100k_svLong-train.json.gz", lines=True, 
                              compression='gzip', orient='records')
        interim_train.drop(["description_right"], axis=1).to_json(f"{interim_path}/shs100k_svShort+Tags-train.json.gz", lines=True, 
                              compression='gzip', orient='records')
        interim_train.drop(["description_right", "keywords_right"], axis=1).to_json(f"{interim_path}/shs100k_svShort-train.json.gz", lines=True, 
                              compression='gzip', orient='records')
        
        print("Generating interim test with song-video...")
        interim_test = hell_of_a_join(test_pairs, mode="vv")
        interim_train.drop(["keywords_right"], axis=1).to_json(f"{interim_path}/shs100k_svLong-gs.json.gz", lines=True, 
                              compression='gzip', orient='records')
        interim_train.drop(["description_right"], axis=1).to_json(f"{interim_path}/shs100k_svShort+Tags-gs.json.gz", lines=True, 
                              compression='gzip', orient='records')
        interim_train.drop(["description_right", "keywords_right"], axis=1).to_json(f"{interim_path}/shs100k_svShort-gs.json.gz", lines=True, 
                              compression='gzip', orient='records')
            
    to_interim_json()
    
    print("Generating interim valid...")
    to_interim_csv()
        
    
def main(input_file: str, output_dir: str, n_pos_pairs=1_000, n_pos_pairs_val=1_000):
    
    output_dir = '_'.join((output_dir, str(n_pos_pairs))) + os.sep
    
    def gen_pairs(data, n_pos_pairs=1_000, factor_neg=6):

        n_max_items = min(n_pos_pairs * 2, 10_000) # heuristic for cross join items
        n_neg_pairs = n_pos_pairs * factor_neg # number of negative pairs
        
        sample = data[["yt_id", "set_id"]].sample(n_max_items)
        cross = pd.merge(sample, sample, how="cross", suffixes=["_a", "_b"])

        pos_pairs = cross.query("(yt_id_a != yt_id_b) & (set_id_a == set_id_b)").sample(n_pos_pairs)
        neg_pairs = cross.query("(yt_id_a != yt_id_b) & (set_id_a != set_id_b)").sample(n_neg_pairs)
        
        pos_pairs["label"] = 1
        neg_pairs["label"] = 0
        
        dataset = pd.concat(
            [pos_pairs, neg_pairs], 
            ignore_index=True).sample(frac=1) #.drop(["set_id_a", "set_id_b"])
        
        return dataset.rename({"yt_id_a": "ltable_id", "yt_id_b": "rtable_id"}, axis=1)

    os.makedirs(output_dir, exist_ok=True)
    
    data_raw = pd.read_parquet(input_file)  
    data_raw.keywords = data_raw.keywords.fillna('').apply(lambda x: ' '.join(map(str, x)))  
    
    # cols used for ER task
    rel_cols = ["yt_id", "video_title", "channel_name", "keywords", "description"]
    
    # save table file
    data_raw[rel_cols].to_parquet(os.path.join(output_dir, f"table.parquet"))
    
    # write train pairs
    print("Generating Training Pairs...")
    train_pairs = gen_pairs(
        data_raw.query("split == 'TRAIN'"),
        n_pos_pairs=n_pos_pairs, 
        factor_neg=6
        )
    train_pairs.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    
    # write test pairs
    print("Generating Test Pairs...")
    test_pairs = gen_pairs(
        data_raw.query("split == 'TEST'"), 
        n_pos_pairs=n_pos_pairs_val,
        factor_neg=6
        )
    test_pairs.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    # write val pairs
    print("Generating Validation Pairs...")
    val_pairs = gen_pairs(
        data_raw.query("split == 'VAL'"), 
        n_pos_pairs=n_pos_pairs_val,
        factor_neg=6
        )
    val_pairs.to_csv(os.path.join(output_dir, "valid.csv"), index=False)
    
    # to interim subdir
    to_interim(output_dir, data_raw, train_pairs, test_pairs, val_pairs, n_pos_pairs)
    
    # to filter metadata file
    relevant_yt_ids = train_pairs.ltable_id.to_list() + train_pairs.rtable_id.to_list() + \
        test_pairs.ltable_id.to_list() + test_pairs.rtable_id.to_list() + val_pairs.ltable_id.to_list() + \
            val_pairs.rtable_id.to_list()
    # build path
    handle = output_dir.split('/')[-2] 
    processed_output_dir = output_dir.replace('raw', 'processed') + 'contrastive/'
    os.makedirs(processed_output_dir, exist_ok=True)
    # write to processed
    data_processed = data_raw.loc[data_raw.yt_id.isin(relevant_yt_ids)].rename(
        {"yt_id": "id", "set_id": "cluster_id"}, axis=1)[
            ["cluster_id", "id", "title", "performer", "video_title", "description", 
             "channel_name", "keywords"]]

    data_processed.to_pickle(
            processed_output_dir + f'{handle}-train.pkl.gz', compression='gzip')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="/data/csi_datasets/shs100k2_yt.parquet",
                        help="path to parquet source file")
    parser.add_argument("--output_dir", type=str, default="/data/repos/contrastive-product-matching/data/raw/shs100k",
                        help="path to parquet source file")
    args = parser.parse_args()
    main(args.input_file, args.output_dir)