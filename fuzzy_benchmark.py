import argparse
import torch
import time
from src.evaluation import RetrievalEvaluation
from src.dataset import TestDataset
from rapidfuzz import process, fuzz, distance
import yaml
import json
from nltk.stem.snowball import SnowballStemmer


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

    ir_eval = RetrievalEvaluation(top_k=10, device=device)
     
    if benchmark == 'funcs':
        result_dict = benchmark_fuzzy_functions(dataset, ir_eval, dataset.get_target_matrix(), device)
        with open(f"results/{dataset_name}/{benchmark}/results.json", "w") as f:
            json.dump(result_dict, f)
    elif benchmark == 'attrs':
        result_dict = benchmark_fuzzy_attributes(dataset, ir_eval, dataset.get_target_matrix(), device, scorer=fuzz.token_ratio)
        with open(f"results/{dataset_name}/{benchmark}/results.json", "w") as f:
            json.dump(result_dict, f)
    
        
def benchmark_fuzzy_attributes(dataset: torch.utils.data.Dataset, ir_eval: RetrievalEvaluation, 
                   target_matrix: torch.Tensor, device: str, scorer):
    
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
    
    for (eval_type, shs_side, yt_side) in evals:
        for stemming in [True, False]:
            
            start_time = time.monotonic()

            try:
                print(f"Scoring {eval_type}")
                processor = stemmer.stem if stemming else None
                prediction_matrix = process.cdist(shs_side, yt_side, scorer=scorer, 
                                                  workers=64, processor=processor)
            except:
                print(f"Skipped {eval_type}")
                continue
        
            test_time = time.monotonic() - start_time

            try:
                results = ir_eval.compute_metrics(torch.tensor(prediction_matrix).to(device), target_matrix.to(device))
            except torch.cuda.OutOfMemoryError:
                print(f"Out of Memory for {scorer.__name__}")
                continue
            
            print(f"{results['mAP']}, inference time: {test_time}, stemming: {stemming}")

            results = {"scorer": scorer.__name__, "stemming": stemming, "eval": eval_type, "time": test_time, **results}
            result_list.append(results)             
            
    return result_list

    
def benchmark_fuzzy_functions(dataset: torch.utils.data.Dataset, ir_eval: RetrievalEvaluation, 
                   target_matrix: torch.Tensor, device: str):
    """Benchmarks all fuzzy functions with YouTube title and channel attributes and SHS attributes.
    """
    # getting all fuzzy functions
    scorers = [fuzz.ratio, fuzz.token_ratio, fuzz.partial_ratio, 
               fuzz.token_set_ratio, fuzz.partial_token_set_ratio, 
               fuzz.token_sort_ratio, fuzz.partial_token_sort_ratio,  
               distance.DamerauLevenshtein.normalized_similarity, 
               distance.JaroWinkler.normalized_similarity, 
               distance.LCSseq.normalized_similarity, 
               distance.Hamming.normalized_similarity, fuzz.WRatio, fuzz.QRatio]
    
    result_list = []
    
    data = dataset.get_df()
    
    items_shs = data.apply(lambda x: x.title_shs + ' ' + x.performer, axis=1).to_list()
    items_shs_short = data.title_shs.to_list()
    items_yt_short = data.apply(lambda x: x.title_yt + ' ' + x.channel_name, axis=1).to_list()
    
    for scorer in scorers:
        
        for (eval_type, shs_side) in [("title", items_shs_short), ("performer+title", items_shs)]:
            
            start_time = time.monotonic()

            func_name = scorer.__module__ + '_' + scorer.__name__
            try:
                print(f"Scoring {func_name}")
                prediction_matrix = process.cdist(shs_side, items_yt_short, scorer=scorer, workers=64)
            except:
                print(f"Skipped {func_name}")
                continue
        
            test_time = time.monotonic() - start_time
        
            try:
                results = ir_eval.compute_metrics(torch.tensor(prediction_matrix).to(device), target_matrix.to(device))
            except torch.cuda.OutOfMemoryError:
                print(f"Out of Memory for {scorer.__name__}")
                continue
            
            print(f"{results['mAP']}, inference time: {test_time}")

            results = {"scorer": func_name, "eval": eval_type, "time": test_time, **results}
            result_list.append(results)             
            
    return result_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark fuzzy functions.")
    parser.add_argument("--benchmark", type=str, default="attrs", 
                        choices=["funcs", "attrs"], help="Benchmark type")
    parser.add_argument("--dataset_name", type=str, default="shs100k2_val", 
                        choices=["shs100k2_test", "shs100k2_val", "shs-yt", "da-tacos"], 
                        help="Dataset name")

    args = parser.parse_args()

    main(args.benchmark, args.dataset_name)    
