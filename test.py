import argparse
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from src.baselines.rsupcon.model import ContrastiveClassifierModel
from src.dataset import TestDataset
from src.evaluation import RetrievalEvaluation
import json


def test(model: torch.nn.Module, dataset: torch.utils.data.Dataset, target_matrix,
         device: str):

    model.eval()
    
    test_loader = DataLoader(dataset, batch_size=16, collate_fn=dataset.collate_fn)
    target_matrix = target_matrix.to(device)
    
    start_time = time.monotonic()

    # A tensor to store all embeddings
    emb_all = torch.tensor([], device=device)
    emb_all2 = torch.tensor([], device=device)


    with torch.no_grad():
        
        for i, batch in tqdm(enumerate(test_loader), desc="Computing embeddings: "):
            
            embs = model(batch["yt_title"])
            embs_target = model(batch["shs_title"])
            # compute and collect embeddings
            emb_all = torch.cat((emb_all, embs))
            emb_all2 = torch.cat((emb_all2, embs_target))

        test_time = time.monotonic() - start_time
        
        # compute metrics
        ir_eval = RetrievalEvaluation(top_k=10, device=device)
        print("Evaluation YouTube -- YouTube")
        metrics_dict = ir_eval.eval(emb_all, target_matrix)
        print("Evaluation SHS -- YouTube")
        metrics_dict2 = ir_eval.eval(emb_all, target_matrix, emb_all2=emb_all2)

    model.train()
    
    print(json.dumps(metrics_dict, indent=4))
    print('Total time: {:.0f}m{:.0f}s.'.format(test_time // 60, test_time % 60))
    return metrics_dict, metrics_dict2


def main(model_name: str, dataset_name: str):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = 'roberta-base'
    
    dataset = TestDataset(
        dataset_name,
        '../data/',
        '../data/yt_metadata.parquet',
        tokenizer=tokenizer
    )
    
    if model_name == 'rsupcon':
        model = ContrastiveClassifierModel(
            model=tokenizer,
            len_tokenizer=len(dataset.tokenizer),
            checkpoint_path="../contrastive-product-matching/reports/finetune/shs100k2_yt-all-256-5e-05-0.07-roberta-base/2/pytorch_model.bin",
            frozen=True,
            pos_neg=False)
    elif model_name == 'magellan':
        pass
    else:
        print(f"Model {model_name} is not implemented!")
        raise NotImplementedError
        
    model.to(device)

    test(model, model_name, dataset, device)
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the ER models.")
    parser.add_argument("--config_file", type=str, default="config.yml", 
                        help="Path to the configuration file")
    parser.add_argument("--model_name", type=str, default="rsupcon", 
                        choices=["rsupcon", "magellan"], help="Model name")
    parser.add_argument("--dataset_name", type=str, default="shs100k2_test", 
                        choices=["shs100k2_test", "shs100k2_val", "shs-yt", "da-tacos"], 
                        help="Dataset name")

    args = parser.parse_args()

    main(args.model_name, args.dataset_name)    
    
        
        