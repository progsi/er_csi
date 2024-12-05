import argparse
import pandas as pd
import os
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from src.baselines.ditto.model import DittoModel
from src.baselines.blocking import Blocker
from src.baselines.sentence_transformers.model import SBert
from src.dataset import TestDataset, OnlineCoverSongDataset
from src.evaluation import RetrievalEvaluation
import json
from rapidfuzz import fuzz, process, distance
import yaml
from transformers import AutoTokenizer, AutoConfig


def test(model: torch.nn.Module, dataset: torch.utils.data.Dataset,
         device: str, ir_eval: RetrievalEvaluation, itemwise_embeddings: bool, 
         blocker: Blocker, task: str, with_audio: bool, only_original: bool, print_all=False):
    
    start_time = time.monotonic()

    # compute metrics
    if model is not None:
        if itemwise_embeddings:
            audio_weight = 0.8 # found with grid search on validation set 
            metrics_dict, metrics_dict2 = __test_model_itemwise(model, dataset, ir_eval, device, with_audio, weight2=audio_weight)
            preds = None
        else:
            audio_weight = 0.8 # found with grid search on validation set but with SBERT and Fuzzy
            preds = __pred_model_pairwise(model, blocker, dataset, task, device)
            metrics_dict, metrics_dict2 = __test_model(preds, dataset, task, ir_eval, device, weight2=audio_weight)
        model.train()
    else:
        left_df, right_df = dataset.get_dfs_by_task(task)
        preds = blocker.predict(left_df, right_df) / 100
        audio_weight = 0.8 # found with grid search on validation set
        metrics_dict, metrics_dict2 = __test_model(preds, dataset, task, ir_eval, device, weight2=audio_weight)

    test_time = time.monotonic() - start_time
        
    print('Total time: {:.0f}m{:.0f}s.'.format(test_time // 60, test_time % 60))

    if print_all:
        print(json.dumps(metrics_dict, indent=4))
        
    return metrics_dict, metrics_dict2, preds


def __pred_model_pairwise(model: torch.nn.Module, blocker: Blocker, 
                          dataset: torch.utils.data.Dataset, task: str,
                            device: str):
    """Embeddings on pair level (slow!), thus with blocking recommended.
    """
    model.eval()
    model_name = type(model).__name__
        
    target_matrix = dataset.get_target_matrix().to(device)
    
    # get indices to predict on
    if blocker is not None:
        if blocker.blocking_func == "sentence-transformers":
            preds = dataset.get_csi_pred_matrix("sentence-transformers").to(device)
            pred_indices = blocker.block_topk(preds)
        else:
            left_df, right_df = dataset.get_dfs_by_task("tvShort")
            blocking_mask, preds = blocker.block_fuzzy(left_df, right_df) if blocker is not None else None    
            pred_indices = torch.nonzero(blocking_mask.triu(1))

            preds = torch.where(blocking_mask > 0, torch.ones_like(blocking_mask).to(dtype=torch.float32), torch.zeros_like(blocking_mask).to(dtype=torch.float32))
    else:
        preds = torch.zeros_like(target_matrix).to(dtype=torch.float32)
        rows, cols = preds.size()
        i_indices, j_indices = torch.meshgrid(torch.arange(rows), torch.arange(cols))
        pred_indices = torch.stack((i_indices, j_indices), dim=-1).reshape(-1, 2)
        pred_indices = pred_indices[pred_indices[:, 0] <= pred_indices[:, 1]]

    # iterating square matrix
    for i, j in tqdm(pred_indices, desc="Generating pairs embeddings..."):
        
        i, j = i.item(), j.item()
        
        if model_name == "ContrastiveClassifierModel":
        
            input_ids, attention_mask = dataset.getitem_tokenized(i, "left", task)
            input_ids_right, attention_mask_right = dataset.getitem_tokenized(j, "right", task)
            
            labels = target_matrix[i,j].unsqueeze(0)

            (loss, pred) = model.forward(input_ids, attention_mask, labels, input_ids_right, attention_mask_right)
            
        elif model_name == "DittoModel":
            
            input_ids, labels = dataset.getitem_pair_tokenized(i, j, task)
            logits = model(input_ids)
            pred = logits.softmax(dim=1)[:, 1]

        
        preds[i,j] = pred.item()
        preds[j,i] = pred.item()
    
    return preds


def __test_model(preds_text: torch.Tensor, dataset: torch.utils.data.Dataset, task: str, 
                          ir_eval: RetrievalEvaluation, device: str, weight2 = 0.5):

    print(f"Evaluation task {task}")
    target_matrix = dataset.get_target_matrix().to(device)

    results = {}
    for audio_model in [None, "coverhunter", "cqtnet"]:
        
        print("\n")

        if audio_model is not None:

            try:
                audio_preds = dataset.get_csi_pred_matrix(audio_model) 
                            
                eval_type = audio_model + "+text"

                print(f"audio_only: {audio_model}")
                metrics_dict_audio = ir_eval.eval(target_matrix, preds1=audio_preds)
                print(f"mAP: {metrics_dict_audio['mAP']}, MR1: {metrics_dict_audio['MR1']}, MRR: {metrics_dict_audio['MRR']}")
            
            except FileNotFoundError:

                print(f"Given dataset not predicted for model {audio_model}")
                continue

        else:
            audio_preds = None
            eval_type = "text_only"

        print(eval_type)
        metrics_dict = ir_eval.eval(target_matrix, preds1=preds_text, preds2=audio_preds, weight2=weight2)

        results[eval_type] = metrics_dict
        print(f"mAP: {metrics_dict['mAP']}, MR1: {metrics_dict['MR1']}, MRR: {metrics_dict['MRR']}")
        
    return results, None
           
                    
def __test_model_itemwise(model: torch.nn.Module, dataset: torch.utils.data.Dataset, 
                            ir_eval: RetrievalEvaluation, device: str, with_audio: bool, weight2: float = 0.5):
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
            
            yt_id = batch["yt_id"]
            
            embs = model(batch["left_side"]) 
            embs_target = model(batch["right_side"])
            
            emb_all = torch.cat((emb_all, embs))
            emb_all2 = torch.cat((emb_all2, embs_target))

        results_sym, results_asym = {}, {}

        text_preds_ll = ir_eval.pairwise_cosine_similarities(emb_all, emb_all)
        text_preds_lr = ir_eval.pairwise_cosine_similarities(emb_all, emb_all2)
        preds_path = os.path.join("preds", "sentence-transformers", dataset.dataset_name)
        os.makedirs(preds_path, exist_ok=True)
        torch.save(text_preds_lr, preds_path + os.sep + "preds.pt")
        torch.save(emb_all, preds_path + os.sep + "emb_all.pt")
        torch.save(emb_all2, preds_path + os.sep + "emb_all2.pt")

        # free GPU memory
        del emb_all, emb_all2
        torch.cuda.empty_cache()
        
        for audio_model in [None, "coverhunter", "cqtnet"]:
            
            if audio_model is not None:
                audio_preds = dataset.get_csi_pred_matrix(audio_model) 
                eval_type = "text+" + audio_model
                print(f"audio_only: {audio_model}")
             
                metrics_dict_audio = ir_eval.eval(target_matrix, preds1=audio_preds)
                print(f"mAP: {metrics_dict_audio['mAP']}, MR1: {metrics_dict_audio['MR1']}, MRR: {metrics_dict_audio['MRR']}")
            
            else:
                audio_preds = None
                eval_type = "text_only"

            print(eval_type)
           
            print("Evaluation left side - left side")
            metrics_dict_sym = ir_eval.eval(target_matrix, preds1=text_preds_ll, preds2=audio_preds, weight2=weight2)
            print(f"mAP: {metrics_dict_sym['mAP']}, MR1: {metrics_dict_sym['MR1']}, MRR: {metrics_dict_sym['MRR']}")
            print("Evaluation left side - right side")
            metrics_dict_asym = ir_eval.eval(target_matrix, preds1=text_preds_lr, preds2=audio_preds, weight2=weight2)
            print(f"mAP: {metrics_dict_asym['mAP']}, MR1: {metrics_dict_asym['MR1']}, MRR: {metrics_dict_asym['MRR']}")

            results_sym[eval_type] = metrics_dict_sym
            results_asym[eval_type] = metrics_dict_asym

            # free CUDA memory
            del audio_preds
            torch.cuda.empty_cache()

            if not with_audio:
                break
            
        return results_sym, results_asym


def main(model_name: str, checkpoint: str, tokenizer_name: str, blocking_func: str, dataset_name: str, task: str,
         nsample: int = None, only_original: bool = False):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with open("config.yml", "r") as f:
        config_data = yaml.safe_load(f)
    
    ir_eval = RetrievalEvaluation(top_k=10, device=device)
    
    if model_name == "sentence-transformers":
        tokenizer = AutoTokenizer.from_pretrained('/'.join((model_name, tokenizer_name)))
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    
    if model_name != "sentence-transformers":

        dataset = TestDataset(
        dataset_name,
        config_data["data_path"],
        config_data["yt_metadata_file"],
        tokenizer=tokenizer,
        nsample=nsample,
        only_original=only_original
        )

        if nsample is None and blocking_func is not None:
            if blocking_func == "sentence-transformers":
                _blocking_func = blocking_func
                _blocking_strategy = "top"
                k = 100 if len(dataset.data) > 100 else 10
                blocker = Blocker(blocking_func=_blocking_func, threshold=None, k=k, strategy=_blocking_strategy) # for > 0.95 recall (empirical)

            else:
                _blocking_func = eval(blocking_func)
                _blocking_strategy = "above"
                blocker = Blocker(blocking_func=_blocking_func, threshold=0.8, strategy=_blocking_strategy) # for > 0.95 recall (empirical)
        else:
            blocker = None
    else:

        # init validation iterator
        dataset = OnlineCoverSongDataset(
            dataset_name,
            config_data["data_path"],
            config_data["yt_metadata_file"],
            task,
            only_original=only_original
        )  

        blocker = None

    if model_name == 'fuzzy':
        model = None

    elif model_name == 'ditto':
        model = DittoModel(device=device, lm=tokenizer_name)

        saved_state = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(saved_state['model'])
        model.to(device)

    elif model_name == "sentence-transformers":
        blocker = None
        model = SBert('/'.join((model_name, tokenizer_name)), pooling='mean') 
        saved_state = torch.load(checkpoint)
        model.load_state_dict(saved_state["model"])
        model.to(device)
    else:
        print(f"Model {model_name} is not implemented!")
        raise NotImplementedError
        
    
    itemwise_embeddings = model_name == "sentence-transformers"
    with_audio = "_one" not in dataset_name

    results1, results2, preds = test(model, dataset, device, ir_eval, itemwise_embeddings, blocker, task, with_audio, only_original)
    
    if preds is not None:
        preds_path = os.path.join("preds", model_name, dataset_name)
        print(f"Saving preds to {preds_path}")
        os.makedirs(preds_path, exist_ok=True)
        torch.save(preds, preds_path + os.sep + "preds.pt")
        dataset.data.to_csv(preds_path + os.sep +"data.csv", sep=";")

    results_path = os.path.join("results", dataset_name, tokenizer_name, model_name, task)
    os.makedirs(results_path, exist_ok=True)

    with open(results_path + os.sep + "sym.json", "w") as f:
        json.dump(results1, f, indent=4)
    
    if results2 is not None:
        with open(results_path + os.sep + "asym.json", "w") as f:
            json.dump(results2, f, indent=4)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the ER models.")
    parser.add_argument("--model_name", type=str, default="ditto",
                        choices=["ditto", "sentence-transformers", "fuzzy"], help="Model name")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model')
    parser.add_argument("--tokenizer_name", type=str, default="roberta-base",
                        choices=["roberta-base", "paraphrase-multilingual-MiniLM-L12-v2", "bert-base-multilingual-cased"])
    parser.add_argument("--blocking_func", type=str, default=None) # default="fuzz.token_set_ratio")
    parser.add_argument("--dataset_name", type=str, default="shs100k2_test_balanced", 
                        help="Dataset name")
    parser.add_argument("--task", type=str, default="rLong", 
                        choices=["svShort", "vvShort", "svShort+Tags", "vvShort+Tags", "svLong", 
                                 "vvLong", "tvShort", "tvLong", "tvShort+Tags", "rLong", "rShort", "xLong",
                                 "xShort"])
    parser.add_argument("--nsample",  type=int, default=None)
    parser.add_argument('-o', '--only_original', action='store_true')
    args = parser.parse_args()

    assert not (args.blocking_func is None and args.model_name == "fuzzy"), "Cannot use blocker as model without defined blocking function"
    
    main(args.model_name, args.checkpoint, args.tokenizer_name, args.blocking_func, args.dataset_name, args.task, args.nsample, args.only_original)    
    
        
        