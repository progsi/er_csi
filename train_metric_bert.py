import argparse
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from src.baselines.sentence_transformers.model import SBert
from src.dataset import OnlineCoverSongDataset
from src.utils import Config
from src.evaluation import RetrievalEvaluation
from test import test
from pytorch_metric_learning import miners, losses, samplers
import wandb


MODELS = [
    "sentence-transformers/all-mpnet-base-v2", 
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # dim. 384, 50 languages
          ]
LOSSES = {
    'contrastive_loss': losses.ContrastiveLoss,
    'triplet_loss': losses.TripletMarginLoss,
    'ntxent_loss': losses.NTXentLoss,
    'multisimilarity_loss': losses.MultiSimilarityLoss,
    'arcface_loss': losses.ArcFaceLoss,
}

# FIXME: Ideas for better performance:
# - hier ganz unten bzgl. Learning Rate: https://github.com/KevinMusgrave/pytorch-metric-learning/issues/534  
      

def train(config_file: str, model_name: str, train_dataset_name: str, 
          val_dataset_name: str, loss_name: str, task: str, mining_strategy: str, epochs: int, 
          batch_size: int, val_every: int, m_per_class: int):
    
    torch.autograd.set_detect_anomaly(True)
    
    config = Config(model_name, train_dataset_name, val_dataset_name,
                    epochs, batch_size, config_file=config_file)
    
    checkpoint_path = os.path.join("checkpoints", model_name)
    os.makedirs(checkpoint_path, exist_ok=True)
    
    model = SBert(model_name, pooling='mean') 

    # init training iterator
    dataset_train = OnlineCoverSongDataset(
        train_dataset_name,
        config.data_path,
        config.yt_metadata_file,
        task
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
        task
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
        project=f"{model_name.split('/')[-1]}_{task}",
        # settings=wandb.Settings(_service_wait=300), # more patience
        # track hyperparameters and run metadata
        config=vars(config))
    
    best_val_mAP = None
    val_mAP = None
    val_mAP_shs = None
    step = 0
    
    model.to(config.device)
    model.train()
    
    ir_eval = RetrievalEvaluation(top_k=10, device=model.device)

    for epoch in range(epochs):
        for epoch_step, batch in tqdm(enumerate(train_loader), desc=f"Training epoch {epoch}"):
            
            labels = batch["set_id_norm"]
                
            optimizer.zero_grad()
            
            embs = model(batch["left_side"])

            # metric learning
            if "vv" not in task:
                embs_target = model(batch["right_side"])
            else:
                embs_target = embs
            
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
            
            if step % val_every == 0:

                val_metrics, val_metrics_shs = test(model, dataset_val, config.device, ir_eval, True, None, task, False)

                val_mAP = val_metrics["text_only"]['mAP']
                val_mAP_shs = val_metrics_shs["text_only"]['mAP']

                print(f"Val. mAP (left-left) at epoch {epoch} and step {step}: {val_mAP}")
                print(f"Val. mAP (left-right) at epoch {epoch} and step {step}: {val_mAP_shs}")

                if best_val_mAP is None or val_mAP > best_val_mAP:
                    
                    best_val_mAP = val_mAP
                    
                    torch.save({"model": model.state_dict()}, 
                               os.path.join(checkpoint_path, f"{task}_{loss_name}_{batch_size}_best.pt"))  
                
            loss_dict = {"loss": loss, 
                       "val_mAP_yt": val_mAP, "best_val_mAP_yt": best_val_mAP, 
                       "val_mAP_shs": val_mAP_shs}
            
            wandb.log(loss_dict)
            
            step += 1
              
            
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a CSI model.")
    parser.add_argument("--config_file", type=str, default="config.yml", 
                        help="Path to the configuration file")
    parser.add_argument("--model_name", type=str, default=MODELS[2], help="Model name")
    parser.add_argument("--train_dataset_name", type=str, default="shs100k_1000", 
                        choices=["shs100k2_train", "shs100k_1000"], 
                        help="Training Dataset name")
    parser.add_argument("--val_dataset_name", type=str, default="shs100k2_val", 
                        choices=["shs100k2_val500", "shs100k2_val", "shs-yt", "shs100k2_test"], 
                        help="Test Dataset name")
    parser.add_argument("--task", type=str, default="svShort",
                        choices=["svShort", "vvShort", "svShort+Tags", "vvShort+Tags", 
                                 "svLong", "vvLong", "tvShort", "tvShort+Tags", "tvLong"])
    parser.add_argument("--mining_strategy", type=str, default="hard", 
                        choices=["hard", "semihard"], 
                        help="Triplet Mining Strategy")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16, help="Training Dataset name")
    parser.add_argument("--loss", type=str, choices=LOSSES.keys(), default="triplet_loss", help="loss function")
    parser.add_argument("--val_every", type=int, default=50, help="how often to perform validation")
    parser.add_argument("--m_per_class", type=int, default=4,
                        help="Number of samples per class for each batch.-1 indicates random sampling instead.")
    #parser.add_argument("--sbert", action="store_true", 
    #                    help="Whether to load from sentence transformers. If not, from transformers.")
    
    args = parser.parse_args()
    
    train(args.config_file, args.model_name, args.train_dataset_name, 
          args.val_dataset_name, args.loss, args.task, args.mining_strategy, args.epochs, args.batch_size, args.val_every, 
          args.m_per_class)    

