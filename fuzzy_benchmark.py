import argparse
import torch
import time
from src.evaluation import RetrievalEvaluation
from src.dataset import TestDataset
from rapidfuzz import process, fuzz, distance
import yaml
import json
from nltk.stem.snowball import SnowballStemmer
from torchmetrics.classification import BinaryRecall
from torchmetrics.retrieval import RetrievalMAP, RetrievalNormalizedDCG


def main(benchmark: str, dataset_name: str):
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with open("config.yml", "r") as f:
        config_data = yaml.safe_load(f)

    dataset = TestDataset(
        dataset_name,
        config_data["data_path"],
        config_data["yt_metadata_file"],
        tokenizer=None
    )
     
    if benchmark == 'funcs':
        result_dict = benchmark_fuzzy_functions(dataset, dataset.get_target_matrix(), device)
        with open(f"results/{dataset_name}/{benchmark}/results.json", "w") as f:
            json.dump(result_dict, f)
    elif benchmark == 'attrs':
        result_dict = benchmark_fuzzy_attributes(dataset, dataset.get_target_matrix(), device, scorer="fuzz.token_ratio")
        with open(f"results/{dataset_name}/{benchmark}/results.json", "w") as f:
            json.dump(result_dict, f)
    

def benchmark_blocking_threshold(dataset: torch.utils.data.Dataset, 
                                 target_matrix: torch.Tensor, device: str, scorer):
    
    result_list = []
    
    data = dataset.get_df()
    
    items_shs = data.apply(lambda x: x.title_shs + ' ' + x.performer, axis=1).to_list()
    items_shs_short = data.title_shs.to_list()
    items_yt_short = data.apply(lambda x: x.title_yt + ' ' + x.channel_name, axis=1).to_list()
    
    thresholds = [i for i in range(0.1, 0.9)]

    start_time = time.monotonic()

    for thr in thresholds:

        recall = BinaryRecall(threshold=thr).to(device)
        
        try:
            print(f"Scoring {scorer}")
            prediction_matrix = torch.tensor(process.cdist(shs_side, items_yt_short, scorer=eval(scorer), workers=64))

            if torch.max(prediction_matrix) > 1:
                prediction_matrix = (prediction_matrix - prediction_matrix.min()) / (prediction_matrix.max() - prediction_matrix.min())
        
        except:
            print(f"Skipped {scorer}")
            continue
    
        test_time = time.monotonic() - start_time
            

def benchmark_fuzzy_attributes(dataset: torch.utils.data.Dataset, target_matrix: torch.Tensor, 
                               device: str, scorer):
    
    result_list = []
    
    data = dataset.get_df()
        
    items_shs = data.apply(lambda x: x.title_shs + ' ' + x.performer, axis=1).to_list()
    items_shs_short = data.title_shs.to_list()
    items_yt = data.apply(lambda x: x.title_yt + ' ' + x.channel_name + ' ' + x.description, axis=1).to_list()
    items_yt_short = data.apply(lambda x: x.title_yt + ' ' + x.channel_name, axis=1).to_list()
    items_yt_keywords = data.apply(lambda x: ' '.join(x.keywords), axis=1)
    items_yt_short_keyword = [' '.join(t) for t in zip(items_yt_short, items_yt_keywords)]
    
    evals = [
        ("short", items_shs_short, items_yt_short),
        ("short_shs", items_shs_short, items_yt),
        ("short_yt", items_shs, items_yt_short),
        ("long", items_shs, items_yt),
        ("keyword_short", items_shs_short, items_yt_keywords),
        ("keyword", items_shs, items_yt_keywords),
        ("short+keyword_short", items_shs_short, items_yt_short_keyword),
        ("short+keyword", items_shs, items_yt_short_keyword)
    ]
    
    stemmer = SnowballStemmer(language="english")
    
    recall = BinaryRecall(threshold=0.5).to(device)
    mAP = RetrievalMAP(empty_target_action="skip").to(device)
    nDCG = RetrievalNormalizedDCG(empty_target_action="skip").to(device)

    for (eval_type, shs_side, yt_side) in evals:
        for stemming in [True, False]:
            
            start_time = time.monotonic()

            try:
                print(f"Scoring {eval_type}")
                processor = stemmer.stem if stemming else None
                prediction_matrix =  torch.tensor(process.cdist(shs_side, yt_side, scorer=scorer, 
                                                  workers=64, processor=processor))
                if torch.max(prediction_matrix) > 1:
                    prediction_matrix = (prediction_matrix - prediction_matrix.min()) / (prediction_matrix.max() - prediction_matrix.min())

            except:
                print(f"Skipped {eval_type}")
                continue
        
            test_time = time.monotonic() - start_time

            preds, target = prediction_matrix.to(device), target_matrix.to(device)
            m, n = target.shape
            indexes = torch.arange(m).view(-1, 1).expand(-1, n).to(device)

            try:
                map = mAP(preds, target, indexes=indexes)
                ndcg = nDCG(preds, target, indexes=indexes)
            except torch.cuda.OutOfMemoryError:
                print(f"Out of Memory for {eval_type}")
                continue
            
            rc = recall(preds, target).item()
            print(f"Recall: {round(rc, 4)}, inference time: {test_time}, stemming: {stemming}")
            print(f"mAP: {round(map.item(), 4)}, inference time: {test_time}, stemming: {stemming}")
            print(f"nDCG: {round(ndcg.item(), 4)}, inference time: {test_time}, stemming: {stemming}")

            results = {"scorer": scorer, "stemming": stemming, "eval": eval_type, "time": test_time, 
                       "Recall": round(rc, 4), "map": round(map.item(), 4), "nDCG": round(ndcg.item(), 4)}
            result_list.append(results)             
            
    return result_list

    
def benchmark_fuzzy_functions(dataset: torch.utils.data.Dataset, target_matrix: torch.Tensor, device: str):
    """Benchmarks all fuzzy functions with YouTube title and channel attributes and SHS attributes.
    """
    # getting all fuzzy functions
    scorers = ["fuzz.ratio", "fuzz.token_ratio", "fuzz.partial_ratio", 
               "fuzz.token_set_ratio", "fuzz.partial_token_set_ratio", 
               "fuzz.token_sort_ratio", "fuzz.partial_token_sort_ratio",  
               "distance.DamerauLevenshtein.normalized_similarity", 
               "distance.JaroWinkler.normalized_similarity", 
               "distance.LCSseq.normalized_similarity", 
               "distance.Hamming.normalized_similarity", "fuzz.WRatio", "fuzz.QRatio"]
    
    result_list = []
    
    data = dataset.get_df()
    
    items_shs = data.apply(lambda x: x.title_shs + ' ' + x.performer, axis=1).to_list()
    items_shs_short = data.title_shs.to_list()
    items_yt_short = data.apply(lambda x: x.title_yt + ' ' + x.channel_name, axis=1).to_list()
    
    recall = BinaryRecall(threshold=0.5).to(device)
    mAP = RetrievalMAP(empty_target_action="skip").to(device)
    nDCG = RetrievalNormalizedDCG(empty_target_action="skip").to(device)

    for scorer in scorers:
        
        for (eval_type, shs_side) in [("title", items_shs_short), ("performer+title", items_shs)]:
            
            start_time = time.monotonic()

            try:
                print(f"Scoring {scorer}")
                prediction_matrix = torch.tensor(process.cdist(shs_side, items_yt_short, scorer=eval(scorer), workers=64))

                if torch.max(prediction_matrix) > 1:
                    prediction_matrix = (prediction_matrix - prediction_matrix.min()) / (prediction_matrix.max() - prediction_matrix.min())
            
            except:
                print(f"Skipped {scorer}")
                continue
        
            test_time = time.monotonic() - start_time
            
            preds, target = prediction_matrix.to(device), target_matrix.to(device)
            m, n = target.shape
            indexes = torch.arange(m).view(-1, 1).expand(-1, n).to(device)
            
            try:
                map = mAP(preds, target, indexes=indexes)
                ndcg = nDCG(preds, target, indexes=indexes)
            except torch.cuda.OutOfMemoryError:
                print(f"Out of Memory for {scorer.__name__}")
                continue
            
            rc = round(recall(preds, target).item(), 4)
            print(f"Recall: {round(rc, 4)}, inference time: {test_time}")
            print(f"mAP: {round(map.item(), 4)}, inference time: {test_time}")
            print(f"mAP: {round(ndcg.item(), 4)}, inference time: {test_time}")

            results = {"scorer": scorer, "eval": eval_type, "time": test_time, 
                       "Recall": round(rc, 4), "map": round(map.item(), 4), "nDCG": round(ndcg.item(), 4)}
            result_list.append(results)             
            
    return result_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark fuzzy functions.")
    parser.add_argument("--benchmark", type=str, default="funcs", 
                        choices=["funcs", "attrs"], help="Benchmark type")
    parser.add_argument("--dataset_name", type=str, default="shs100k2_val", 
                        choices=["shs100k2_test", "shs100k2_val", "shs-yt", "da-tacos"], 
                        help="Dataset name")

    args = parser.parse_args()

    main(args.benchmark, args.dataset_name)    
