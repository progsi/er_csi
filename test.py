import argparse
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from src.dataset import TestDataset
from src.utils import Config
from src.evaluation import RetrievalEvaluation
import json


def test(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, 
         target_matrix: torch.tensor):

    model.eval()
    
    start_time = time.monotonic()

    # A tensor to store all embeddings
    device = 'cuda'
    emb_all = torch.tensor([], device=device)

    with torch.no_grad():
        
        for i, batch in tqdm(enumerate(test_loader), desc="Computing embeddings: "):

            # get feature
            feats = batch["feat"].to(device)
            
            # compute and collect embeddings
            embs, __ = model.inference(feats)
            emb_all = torch.cat((emb_all, embs))

        test_time = time.monotonic() - start_time
        
        # compute metrics
        ir_eval = RetrievalEvaluation(top_k=10, device=device)
        metrics_dict = ir_eval.eval(emb_all, target_matrix)

    model.train()
    
    print(json.dumps(metrics_dict, indent=4))
    print('Total time: {:.0f}m{:.0f}s.'.format(test_time // 60, test_time % 60))
    return metrics_dict


def main(config_file: str, model_name: str, dataset_name: str):
    
    config = Config(model_name, config_file=config_file)

    if model_name == 'rsupcon':
        pass
    elif model_name == 'magellan':
        pass
    else:
        print(f"Model {model_name} is not implemented!")
        raise NotImplementedError
        
    model.load_weights()
    model.to(config.device)
    
    dataset = TestDataset(
        dataset_name,
        config.data_path,
        config.yt_metadata_file
    )

    test_loader = DataLoader(dataset, batch_size=16, collate_fn=dataset.collate_fn)
    
    test(model, test_loader, dataset.get_target_matrix().to(config.device))
    
        
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

    main(args.config_file, args.model_name, args.dataset_name)    
    
        
        