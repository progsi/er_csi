import yaml
import os

class Config(object):
    def __init__(self, model_name: str, train_dataset_name: str, val_dataset_name: str, 
                 epochs: int, batch_size: int, config_file: str = "config.yml"):

        self.train_dataset_name = train_dataset_name
        self.val_dataset_name = val_dataset_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_hps = None
        self.model_name = model_name
        
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)
        # paths
        for key, value in config_data.items():
            setattr(self, key, value)
            
