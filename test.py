import argparse
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from src.baselines.rsupcon.model import ContrastiveClassifierModel
from src.baselines.blocking import Blocker
from src.dataset import TestDataset
from src.evaluation import RetrievalEvaluation
import json
import rapidfuzz
from rapidfuzz import fuzz, process, distance
import yaml


def test(model: torch.nn.Module, dataset: torch.utils.data.Dataset, target_matrix,
         device: str, ir_eval: RetrievalEvaluation, print_all=False):

    target_matrix = target_matrix.to(device)
    
    start_time = time.monotonic()
    
    # compute metrics
    metrics_dict, metrics_dict2 = __test_model(model, dataset, device, ir_eval, target_matrix)
        
    test_time = time.monotonic() - start_time
        
    print('Total time: {:.0f}m{:.0f}s.'.format(test_time // 60, test_time % 60))

    if print_all:
        print(json.dumps(metrics_dict, indent=4))
        
    model.train()

    return metrics_dict, metrics_dict2


def __test_model(model: torch.nn.Module, dataset: torch.utils.data.Dataset, 
                 device: str, ir_eval: RetrievalEvaluation, target_matrix: torch.Tensor):
    
    model.eval()

    test_loader = DataLoader(dataset, batch_size=16, collate_fn=dataset.collate_fn)

    # A tensor to store all embeddings
    emb_all = torch.tensor([], device=device)
    emb_all2 = torch.tensor([], device=device)

    with torch.no_grad():
        
        for i, batch in tqdm(enumerate(test_loader), desc="Computing embeddings: "):
            
            embs = model(batch["yt_title"]) 
            embs_target = model(batch["shs_title"])
            
            emb_all = torch.cat((emb_all, embs))
            emb_all2 = torch.cat((emb_all2, embs_target))

        print("Evaluation YouTube -- YouTube")
        metrics_dict = ir_eval.eval(emb_all, target_matrix)
        print("Evaluation SHS -- YouTube")
        metrics_dict2 = ir_eval.eval(emb_all, target_matrix, emb_all2=emb_all2)
    return metrics_dict, metrics_dict2
   

    
def main(model_name: str, blocking_func: str, dataset_name: str, task: str):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with open("config.yml", "r") as f:
        config_data = yaml.safe_load(f)
    
    ir_eval = RetrievalEvaluation(top_k=10, device=device)
    
    if blocking_func is not None:
        blocker = Blocker(blocking_func=blocking_func)
    if model_name == 'rsupcon':
        model = ContrastiveClassifierModel(
            model='roberta-base',
            len_tokenizer=512, # FIXME: fixed length
            checkpoint_path="../contrastive-product-matching/reports/contrastive-ft-siamese/shs100k-shs100k_svS-train-all-256-all-5e-05-0.07-frozen-roberta-base",
            frozen=True,
            pos_neg=False)
    elif model_name == 'magellan':
        pass
    else:
        print(f"Model {model_name} is not implemented!")
        raise NotImplementedError
        
    model.to(device)

    dataset = TestDataset(
    dataset_name,
    config_data["data_path"],
    config_data["yt_metadata_file"],
    tokenizer=model.tokenizer
    )

    test(model, model_name, dataset, device, ir_eval)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the ER models.")
    parser.add_argument("--config_file", type=str, default="config.yml", 
                        help="Path to the configuration file")
    parser.add_argument("--model_name", type=str, default="rsupcon", 
                        choices=["ditto", "hiergat", "sbert", "rsupcon", "magellan", "blocking"], help="Model name")
    parser.add_argument("--blocking_func", type=str, default="token_ratio")
    parser.add_argument("--dataset_name", type=str, default="shs100k2_test", 
                        choices=["shs100k2_test", "shs100k2_val", "shs-yt", "da-tacos"], 
                        help="Dataset name")
    parser.add_argument("--task", type=str, default="svS", 
                        choices=["svS", "vvS", "svL", "vvL"])

    args = parser.parse_args()

    assert not (args.blocking_func is None and args.model_name == "blocker"), "Cannot use blocker as model without defined blocking function"
    
    main(args.model_name, args.blocking_func, args.dataset_name, args.task)    
    
        
        