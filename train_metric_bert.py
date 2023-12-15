import argparse
from datetime import datetime
import os
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from torch.nn import MultiLabelSoftMarginLoss
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from src.dataset import OnlineCoverSongDataset
from src.utils import Config
from src.evaluation import RetrievalEvaluation
from test import test
import json
from pytorch_metric_learning import miners, losses, distances, samplers, regularizers
import wandb


MODELS = ["sentence-transformers/distiluse-base-multilingual-cased-v2", 
          "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"]
LOSSES = {
    'contrastive_loss': losses.ContrastiveLoss,
    'triplet_loss': losses.TripletMarginLoss,
    'ntxent_loss': losses.NTXentLoss,
    'multisimilarity_loss': losses.MultiSimilarityLoss,
    'arcface_loss': losses.ArcFaceLoss,
}


class SBert(nn.Module):
    def __init__(self, base_name, pooling='max', device='cuda'):
        super(SBert, self).__init__()
        self.base_name = base_name    
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_name)
        self.model = AutoModel.from_pretrained(self.base_name)
        self.pooling = self.__max_pooling if pooling == 'max' else self.__mean_pooling
        
    def forward(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
        model_output = self.model(**encoded_input.to(self.device))
        pooled_output = self.pooling(model_output, encoded_input['attention_mask'])
        return pooled_output
        
    def __max_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return torch.max(token_embeddings, 1)[0]

    def __mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask    
      

def train(config_file: str, model_name: str, train_dataset_name: str, 
          val_dataset_name: str, loss_name: str, attr_pairs: str, mining_strategy: str, epochs: int, 
          batch_size: int, val_every: int, m_per_class: int, sbert: bool):
    
    torch.autograd.set_detect_anomaly(True)
    
    config = Config(model_name, train_dataset_name, val_dataset_name,
                    epochs, batch_size, config_file=config_file)
    
    checkpoint_path = os.path.join("checkpoints", model_name)
    os.makedirs(checkpoint_path, exist_ok=True)
    
    if sbert:
        model = SBert(model_name, pooling='mean')
    else:
        pass

    model.to(config.device)
    model.train()
    
    # init training iterator
    dataset_train = OnlineCoverSongDataset(
        train_dataset_name,
        config.data_path,
        config.yt_metadata_file
    )
    
    if m_per_class > 0:
        # sampler to unsure certain number of ites per class per batch
        sampler = samplers.MPerClassSampler(labels=dataset_train.data.set_id_norm.to_list(), 
                                        m=m_per_class)
        train_loader = DataLoader(dataset_train, batch_size=batch_size, 
                              collate_fn=dataset_train.collate_fn, sampler=sampler)
    else:
        train_loader = DataLoader(dataset_train, batch_size=batch_size, 
                              collate_fn=dataset_train.collate_fn, shuffle=True)
        
    # init validation iterator
    dataset_val = OnlineCoverSongDataset(
        val_dataset_name,
        config.data_path,
        config.yt_metadata_file
    )  
    
    if mining_strategy == "hard":
        miner = miners.BatchHardMiner()
    elif mining_strategy == "semihard":
        miner = miners.BatchEasyHardMiner(neg_strategy="semihard")
    
    # Losses
    # https://github.com/KevinMusgrave/pytorch-metric-learning/issues/512
    # advice: " I would use ContrastiveLoss, MultiSimilarityLoss, or NTXentLoss (also known as InfoNCE) because they tend to perform better than TripletMarginLoss."
    

    loss_func = LOSSES[loss_name](**config.__getattribute__(loss_name))

    params = list(model.parameters()) + list(loss_func.parameters()) if loss_name == "arcface_loss" else model.parameters()
    
    optimizer = AdamW(params=params, lr=config.learning_rate)
    
    # scheduler = ExponentialLR(optimizer, gamma=0.95)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    if hasattr(model, "hp"):
        config.set_model_hps(model.hp)
        
    wandb.init(
    # set the wandb project where this run will be logged
        project=model_name.split('/')[-1],
        # settings=wandb.Settings(_service_wait=300), # more patience
        # track hyperparameters and run metadata
        config=vars(config))
    
    best_val_mAP = None
    val_mAP = None
    val_mAP_shs = None
    step = 0
    
    for epoch in range(epochs):
        for epoch_step, batch in tqdm(enumerate(train_loader), desc=f"Training epoch {epoch}"):
            
            labels = batch["set_id_norm"]
                
            optimizer.zero_grad()
            
            embs = model(batch["yt_title"])

            # metric learning
            if attr_pairs == "yt-yt":
                embs_target = embs
            else:
                embs_target = model(batch["shs_title"])
            
            if loss_name == "triplet_loss":
                hard_triplets = miner(embeddings=embs, labels=labels, 
                                        ref_emb=embs_target, ref_labels=labels)
                loss = loss_func(embeddings=embs, labels=labels, 
                                        ref_emb=embs_target, ref_labels=labels, indices_tuple=hard_triplets)
            elif loss_name == "arcface_loss":
                loss = loss_func(embeddings=embs, labels=labels) 
            else:
                loss = loss_func(embeddings=embs, labels=labels, 
                                        ref_emb=embs_target, ref_labels=labels) 
            loss.backward()
            optimizer.step()
            # scheduler.step(loss)
            
            if step != 0 and step % val_every == 0:

                val_metrics, val_metrics_shs = test(model, dataset_val, dataset_val.get_target_matrix().to(config.device), 
                                                    config.device)
                
                val_mAP = val_metrics['mAP']
                val_mAP_shs = val_metrics_shs['mAP']

                print(f"Val. mAP (yt-yt) at epoch {epoch} and step {step}: {val_mAP}")
                print(f"Val. mAP (shs-yt) at epoch {epoch} and step {step}: {val_mAP_shs}")

                if best_val_mAP is not None and val_mAP > best_val_mAP:
                    
                    best_val_mAP = val_mAP
                    
                    torch.save({"model": model.state_dict()}, 
                               os.path.join(checkpoint_path, "best.pt"))  
                
            loss_dict = {"loss": loss, 
                       "val_mAP_yt": val_mAP, "best_val_mAP_yt": best_val_mAP, 
                       "val_mAP_shs": val_mAP_shs}
            
            wandb.log(loss_dict)
            
            step += 1
              
            
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a CSI model.")
    parser.add_argument("--config_file", type=str, default="config.yml", 
                        help="Path to the configuration file")
    parser.add_argument("--model_name", type=str, default=MODELS[0], help="Model name")
    parser.add_argument("--train_dataset_name", type=str, default="shs100k2_train", 
                        choices=["shs100k2_train"], 
                        help="Training Dataset name")
    parser.add_argument("--val_dataset_name", type=str, default="shs100k2_val", 
                        choices=["shs100k2_val", "shs-yt", "shs100k2_test"], 
                        help="Test Dataset name")
    parser.add_argument("--attr_pairs", type=str, default="yt-yt",
                        choices=["yt-yt", "shs-yt"])
    parser.add_argument("--mining_strategy", type=str, default="hard", 
                        choices=["hard", "semihard"], 
                        help="Triplet Mining Strategy")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128, help="Training Dataset name")
    parser.add_argument("--loss", type=str, choices=LOSSES.keys(), default="arcface_loss", help="loss function")
    parser.add_argument("--val_every", type=int, default=100, help="how often to perform validation")
    parser.add_argument("--m_per_class", type=int, default=4,
                        help="Number of samples per class for each batch.-1 indicates random sampling instead.")
    parser.add_argument("--sbert", action="store_true", 
                        help="Whether to load from sentence transformers. If not, from transformers.")
    
    args = parser.parse_args()
    
    train(args.config_file, args.model_name, args.train_dataset_name, 
          args.val_dataset_name, args.loss, args.attr_pairs, args.mining_strategy, args.epochs, args.batch_size, args.val_every, 
          args.m_per_class, args.sbert)    

