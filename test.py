import argparse
import pandas as pd
import os
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from src.baselines.rsupcon.model import ContrastiveClassifierModel
from src.baselines.ditto.model import DittoModel
from src.baselines.hiergat.model import TranHGAT
from src.baselines.blocking import Blocker
from src.dataset import TestDataset
from src.evaluation import RetrievalEvaluation
import json
from rapidfuzz import fuzz, process, distance
import yaml
from transformers import AutoTokenizer, AutoConfig


def test(model: torch.nn.Module, dataset: torch.utils.data.Dataset,
         device: str, ir_eval: RetrievalEvaluation, itemwise_embeddings: bool, 
         blocker: Blocker, task: str, print_all=False):
    
    start_time = time.monotonic()
    
    # compute metrics
    if itemwise_embeddings:
        metrics_dict, metrics_dict2 = __test_model_itemwise(model, dataset, ir_eval, device)
    else:
        metrics_dict, _ = __test_model_pairwise(model, blocker, dataset, task, ir_eval, device)            
        
    test_time = time.monotonic() - start_time
        
    print('Total time: {:.0f}m{:.0f}s.'.format(test_time // 60, test_time % 60))

    if print_all:
        print(json.dumps(metrics_dict, indent=4))
        
    model.train()

    return metrics_dict, metrics_dict2


def __test_model_pairwise(model: torch.nn.Module, blocker: Blocker, 
                          dataset: torch.utils.data.Dataset, task: str,
                            ir_eval: RetrievalEvaluation, device: str):
    """Embeddings on pair level (slow!), thus with blocking recommended.
    """
    model.eval()
    model_name = type(model).__name__
    
    left_df, right_df = dataset.get_dfs_by_task(task)
    x_length, y_length = len(left_df), len(right_df)
    
    target_matrix = dataset.get_target_matrix().to(device)
    blocking_mask = blocker.block(left_df, right_df) if blocker is not None else None    
    preds = torch.where(blocking_mask > 0, torch.ones_like(blocking_mask).to(dtype=torch.float32), torch.zeros_like(blocking_mask).to(dtype=torch.float32))
    
    # iterating square matrix
    for i in tqdm(range(x_length), desc="Generating pairs embeddings..."):
        for j in range(y_length):
            # check if result is blocked
            if preds[i,j].item() > 0:
                
                if model_name == "ContrastiveClassifierModel":
                
                    input_ids, attention_mask = dataset.getitem_tokenized(i, "left", task)
                    input_ids_right, attention_mask_right = dataset.getitem_tokenized(j, "right", task)
                    
                    labels = target_matrix[i,j].unsqueeze(0)
                    
                    (loss, pred) = model.forward(input_ids, attention_mask, labels, input_ids_right, attention_mask_right)
                    
                elif model_name == "DittoModel":
                    
                    input_ids, labels = dataset.getitem_pair_tokenized(i, j, task)
                    logits = model(input_ids)
                    pred = logits.softmax(dim=1)[:, 1]

                elif model_name == 'TranHGAT':
                    
                    _, x, y, masks, _ = dataset.getitem_hiergat(
                        i, j, task, True ) # FIXME: split param is fixed!
                    logits, y1, y_hat = model(x, y, masks)
                    logits = logits.view(-1, logits.shape[-1])
                    pred = y_hat.view(-1)
                
                preds[i,j] = pred.item()
                
    print(f"Evaluation task {task}")
    metrics_dict = ir_eval.eval(target_matrix, preds=preds)
    return metrics_dict, _
           
                    
def __test_model_itemwise(model: torch.nn.Module, dataset: torch.utils.data.Dataset, 
                            ir_eval: RetrievalEvaluation, device: str ):
    """Embeddings on item level (fast!)
    """
    model.eval()

    target_matrix = dataset.get_target_matrix().to(device)
    
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
        metrics_dict = ir_eval.eval(target_matrix, emb_all=emb_all)
        print("Evaluation SHS -- YouTube")
        metrics_dict2 = ir_eval.eval(target_matrix, emb_all=emb_all, emb_all2=emb_all2)
    return metrics_dict, metrics_dict2
   
    
def main(model_name: str, tokenizer_name: str, blocking_func: str, dataset_name: str, task: str):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with open("config.yml", "r") as f:
        config_data = yaml.safe_load(f)
    
    ir_eval = RetrievalEvaluation(top_k=10, device=device)
    
    checkpoint_dir = os.path.join("checkpoints", model_name, tokenizer_name, task)
    if model_name == "rsupcon":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, additional_special_tokens=('[COL]', '[VAL]'))
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    
    if blocking_func is not None:
        blocker = Blocker(blocking_func=blocking_func, threshold=0.5)
    if model_name == 'rsupcon':
        attr_num = None
        model = ContrastiveClassifierModel(
            model=tokenizer_name,
            len_tokenizer=len(tokenizer), 
            checkpoint_path=checkpoint_dir + os.sep + "pytorch_model.bin",
            frozen=True,
            pos_neg=False)
    elif model_name == 'ditto':
        attr_num = None
        model = DittoModel(device=device)
        checkpoint = checkpoint_dir + os.sep + "model.pt"
        if os.path.isfile(checkpoint):
            saved_state = torch.load(checkpoint, map_location=lambda storage, loc: storage)
            model.load_state_dict(saved_state['model'])
        model.cuda()
    elif model_name == 'hiergat':
        split = True # FIXME: this should go to a yaml config file
        if split:
            attr_num = 3 if "Long" in task or "+Tags" in task else 2
        else:
            attr_num = 1
        model = TranHGAT(attr_num=attr_num, lm=tokenizer_name, device=device)
        checkpoint = checkpoint_dir + os.sep + "model.pt"
        if os.path.isfile(checkpoint):
            saved_state = torch.load(checkpoint, map_location=lambda storage, loc: storage)
            model.load_state_dict(saved_state)
        model.cuda()
    else:
        print(f"Model {model_name} is not implemented!")
        raise NotImplementedError
        
    model.to(device)

    dataset = TestDataset(
    dataset_name,
    config_data["data_path"],
    config_data["yt_metadata_file"],
    tokenizer=tokenizer,
    attr_num=attr_num
    )
    
    itemwise_embeddings = model_name == "sentence-transformer"
    
    test(model, dataset, device, ir_eval, itemwise_embeddings, blocker, task)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the ER models.")
    parser.add_argument("--model_name", type=str, default="hiergat", 
                        choices=["ditto", "hiergat", "sentence-transformers", "rsupcon", "magellan", "blocking"], help="Model name")
    parser.add_argument("--tokenizer_name", type=str, default="roberta-base",
                        choices=["roberta-base", "paraphrase-multilingual-MiniLM-L12-v2"])
    parser.add_argument("--blocking_func", type=str, default="token_ratio")
    parser.add_argument("--dataset_name", type=str, default="shs100k2_test", 
                        choices=["shs100k2_test", "shs100k2_val", "shs-yt", "da-tacos"], 
                        help="Dataset name")
    parser.add_argument("--task", type=str, default="svShort", 
                        choices=["svShort", "vvShort", "svShort+Tags", "vvShort+Tags", "svLong", "vvLong"])

    args = parser.parse_args()

    assert not (args.blocking_func is None and args.model_name == "blocker"), "Cannot use blocker as model without defined blocking function"
    
    main(args.model_name, args.tokenizer_name, args.blocking_func, args.dataset_name, args.task)    
    
        
        