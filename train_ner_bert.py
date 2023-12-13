import argparse
from datetime import datetime
import os
import time
from tqdm import tqdm
import torch
from torch.nn import MultiLabelSoftMarginLoss
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from src.dataset.dataset import OnlineCoverSongDataset
from src.dataset.tagset import TagSet
from src.utils.config import Config
from src.evaluation import RetrievalEvaluation
from test import test
import json
from pytorch_metric_learning import miners, losses, distances, samplers
import wandb


def train(config_file: str, model_name: str, train_dataset_name: str, 
          val_dataset_name: str, mining_strategy: str, epochs: int, batch_size: int, val_every: int, 
          tagset_transfer: bool, m_per_class: int):
    
    torch.autograd.set_detect_anomaly(True)
    
    config = Config(model_name, train_dataset_name, val_dataset_name,
                    epochs, batch_size, tagset_transfer, m_per_class, 
                    config_file=config_file)
    
    checkpoint_path = os.path.join("checkpoints", model_name)
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # the Bottleneck for multiloss.
    bnlayer = BottleNeck(config.embed_dim, config.bn_classes) # 30k song classes in CoverHunter
    print(f"Setting up model {model_name} with embedding size {config.embed_dim} and {config.bn_classes} bn classes")
    
    model = BERT

    model.to(config.device)
    model.train()
    # the feature key
    feat_key = model.get_feat_key()
    
    
    # init training iterator
    dataset_train = OnlineCoverSongDataset(
        train_dataset_name,
        config.data_path,
        config.yt_metadata_file,
        config.audio_feat_path,
        feat_key
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
        config.yt_metadata_file,
        config.audio_feat_path,
        model.get_feat_key(),
        audio_aug_transform=None
    )  
    
    val_loader = DataLoader(dataset_val, batch_size=16, 
                            collate_fn=dataset_val.collate_fn)
    
    if mining_strategy == "hard":
        miner = miners.BatchHardMiner()
    elif mining_strategy == "semihard":
        miner = miners.BatchEasyHardMiner(neg_strategy="semihard")
    
    # Losses
    triplet_loss = losses.TripletMarginLoss()

    
    # optim
    optimizer = AdamW(params=params, lr=config.learning_rate, 
                      betas=(config.adam_b1, config.adam_b2))
    
    # scheduler = ExponentialLR(optimizer, gamma=0.95)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    if hasattr(model, "hp"):
        config.set_model_hps(model.hp)
        
    wandb.init(
    # set the wandb project where this run will be logged
        project=model_name,
        # settings=wandb.Settings(_service_wait=300), # more patience
        # track hyperparameters and run metadata
        config=vars(config))
    
    best_val_mAP = None
    val_mAP = None
    step = 0
    
    for epoch in range(epochs):
        for epoch_step, batch in tqdm(enumerate(train_loader), desc=f"Training epoch {epoch}"):
            
            feats, labels = batch["feat"], batch["set_id_norm"]
                
            optimizer.zero_grad()
            
            # CSI
            embs, preds = model.inference(feats)
            
            hard_triplets = miner(embs, labels)
            
            loss_triplet = triplet_loss(embs, labels, hard_triplets)
            loss_center = center_loss(embs, labels)
            loss_focal = focal_loss(preds, labels)
            
            loss = triplet_weight*loss_triplet + center_weight*loss_center + focal_weight*loss_focal
            
            # tag matching
            if tagset_transfer:
                labels_tag = tagset.match(batch["yt_keywords"])
                preds_tag = model_tags(embs)
                tag_loss = loss_tags(preds_tag, labels_tag.to(config.device))                

                loss += tag_loss

            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            if step != 0 and step % val_every == 0:

                val_metrics = test(model, val_loader, dataset_val.get_target_matrix().to(config.device))
                
                val_mAP = val_metrics['mAP']
                print(f"Val. mAP at epoch {epoch} and step {step}: {val_mAP}")
                
                if best_val_mAP is not None and val_mAP > best_val_mAP:
                    
                    best_val_mAP = val_mAP
                    
                    torch.save({"model": model.state_dict(), 
                                "tag_bn": model_tags.state_dict()}, os.path.join(checkpoint_path, "best.pt"))  
            
            # if step != 0 and step % 1000 == 0:
                
            #    scheduler.step()
                
            loss_dict = {"loss_triplet": loss_triplet, "loss_center": loss_center, 
                       "loss_focal": loss_focal, "loss": loss, 
                       "val_mAP": val_mAP, "best_val_mAP": best_val_mAP}
            
            if tagset_transfer:
                loss_dict["tag_loss"] = tag_loss
            
            wandb.log(loss_dict)
            
            step += 1
              
            
if __name__ == "__main__":
    # for debugging:
    import socket
    internet_available = lambda: True if socket.create_connection(("8.8.8.8", 53), timeout=3) is not None else False
    print(f"{datetime.now()} Internet available: {internet_available()} ")

    parser = argparse.ArgumentParser(description="Train a CSI model.")
    parser.add_argument("--config_file", type=str, default="config.yml", 
                        help="Path to the configuration file")
    parser.add_argument("--model_name", type=str, default="roberta", help="Model name")
    parser.add_argument("--train_dataset_name", type=str, default="shs100k2_train", 
                        choices=["shs100k2_train"], 
                        help="Training Dataset name")
    parser.add_argument("--val_dataset_name", type=str, default="shs100k2_val", 
                        choices=["shs100k2_val", "shs-yt", "shs100k2_test"], 
                        help="Test Dataset name")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256, help="Training Dataset name")
    parser.add_argument("--val_every", type=int, default=100, help="how often to perform validation")
    
    args = parser.parse_args()
    
    train(args.config_file, args.model_name, args.train_dataset_name, 
          args.val_dataset_name, args.mining_strategy, args.epochs, args.batch_size, args.val_every, 
          args.tagset_transfer, args.m_per_class)    
    