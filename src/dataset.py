import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.transforms import UnicodeNormalize
from transformers import AutoTokenizer, AutoConfig


class TestDataset(Dataset):
    """This class iterates on sample level:
    one iteration is one video or candidate version
    Args:
        OnlineCoverDataset (_type_): _description_
    """
    def __init__(
            self,
            dataset_name: str, # shs-yt, shs100k2_test, shs100k2_train, ...
            data_path: str, # path with datasets metadata
            metadata_file_path: str, # path to youtube metadata in parquet file
            tokenizer: str = 'roberta-base',
            device: str = "cuda"
        ) -> None:
            
        super().__init__()
        
        self.dataset_name = dataset_name
        
        if tokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('[COL]', '[VAL]'))
        else:
            self.tokenizer = None
        
        self.data_path = data_path
        self.metadata_file_path = metadata_file_path
        
        # the youtube metadata
        self.metadata = pd.read_parquet(self.metadata_file_path).reset_index()
        # cleanup NoneType keywords
        self.metadata.keywords = self.metadata.keywords.apply(lambda x: [] if x is None else x)
        
        # the dataset
        self.data = pd.read_csv(
            os.path.join(self.data_path, dataset_name + '.csv'), sep=";")
        self.data = self.data.query("has_file")
        if "nlabel" not in self.data.columns:
            """If no nlabel is specified (as in datasets SHS100K, Da-Tacos, etc.)
            then we assign a column with 2s indicating that are items are versions.
            """            
            self.data["nlabel"] = 2
        self.data["set_id_norm"] = self.data.set_id.rank(method='dense').astype(int) - 1
        
        # transforms
        self.text_transform = UnicodeNormalize()
        self.device = device
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        
        item = dict(self.data.iloc[idx])
            
        yt_id = item["yt_id"]    
        
        # get youtube metadata 
        item_yt = self.metadata.loc[self.metadata.yt_id == yt_id]
        item["yt_title"] = self.text_transform(item_yt.title.item()) if not item_yt.title.empty else ''
        item["yt_channel"] = self.text_transform(item_yt.channel_name.item()) if not item_yt.channel_name.empty else ''
        item["yt_description"] = self.text_transform(item_yt.description.item()) if not item_yt.description.empty else ''
        item["yt_keywords"] = [
                self.text_transform(keyword) for keyword in item_yt.keywords.item()
            ] if len(item_yt.keywords) > 0 else []
        
        if self.tokenizer is not None:
            tokenized = self.tokenizer(self.serialize(item))
            item["input_ids"], item["attention_mask"] = tokenized["input_ids"], tokenized["attention_mask"]
        
        return item
    
    def get_target_matrix(self):
        """Generates the binary square matrix of cover song relationships between all
        elements in the dataset.
        Returns:
            _type_: _description_
        """
        labels = torch.from_numpy(self.data["set_id"].values) 
        nlabels = torch.from_numpy(self.data["nlabel"].values)
        relevance = nlabels >= 2  

        mask = torch.logical_and(relevance[:, None], relevance)
        mask.fill_diagonal_(False)  

        target = torch.zeros(len(self.data), len(self.data), dtype=torch.int32)
        target[mask] = (labels[:, None] == labels).int()[mask]

        return target
    
    def collate_fn(self, batch):
        
        # Filter out None values (rows with missing audio feature)
        batch = [item for item in batch if item is not None]
        
        # Prepare the rest of the data
        yt_ids = [item['yt_id'] for item in batch]
        yt_titles = [item['yt_title'] for item in batch]
        yt_channels = [item['yt_channel'] for item in batch]
        yt_descriptions = [item['yt_description'] for item in batch]
        yt_keywords = [item['yt_keywords'] for item in batch]
        set_ids = [item['set_id'] for item in batch]
        set_ids_norm = [item['set_id_norm'] for item in batch]
        ver_ids = [item['ver_id'] for item in batch]

        return {
            'yt_id': yt_ids,
            'yt_title': yt_titles,
            'yt_channel': yt_channels,
            'yt_description': yt_descriptions,
            'yt_keywords': yt_keywords,
            'set_id': torch.tensor(set_ids),
            'set_id_norm': torch.tensor(set_ids_norm),
            'ver_id': torch.tensor(ver_ids)        
            }
        
    def serialize(self, item):
        string = ''
        string = f'{string}[COL] video title [VAL] {" ".join(item[f"yt_title"].split())}'.strip()
        string = f'{string} [COL] description [VAL] {" ".join(item[f"yt_description"].split())}'.strip()
        string = f'{string} [COL] channel [VAL] {" ".join(str(item[f"yt_channel"]).split())}'.strip()
        return string
    
    def get_df(self):
        df = pd.merge(self.data, self.metadata, on='yt_id', how='left', suffixes=['_shs', '_yt'])
        df.title_yt = df.title_yt.fillna('').apply(self.text_transform)        
        df.channel_name = df.channel_name.fillna('').apply(self.text_transform)
        df.description = df.description.fillna('').apply(self.text_transform)
        df.keywords = df.keywords.fillna('')
        return df
    

class TrainingDataset(Dataset):
    """This class iterates on sample level:
    one iteration is one video or candidate version
    Args:
        OnlineCoverDataset (_type_): _description_
    """
    def __init__(
            self,
            dataset_name: str, # shs-yt, shs100k2_test, shs100k2_train, ...
            data_path: str, # path with datasets metadata
            metadata_file_path: str, # path to youtube metadata in parquet file
            neg_ratio: int = 3, # ratio of negative items per positive item
            device: str = "cuda"
        ) -> None:
            
        super().__init__()
        
        self.dataset_name = dataset_name
        self.neg_ratio = neg_ratio
        
        self.data_path = data_path
        self.metadata_file_path = metadata_file_path
        
        # the youtube metadata
        self.metadata = pd.read_parquet(self.metadata_file_path).reset_index()
        # cleanup NoneType keywords
        self.metadata.keywords = self.metadata.keywords.apply(lambda x: [] if x is None else x)
        
        # the dataset
        self.data = pd.read_csv(
            os.path.join(self.data_path, dataset_name + '.csv'), sep=";")
        self.data = self.data.query("has_file")
        if "nlabel" not in self.data.columns:
            """If no nlabel is specified (as in datasets SHS100K, Da-Tacos, etc.)
            then we assign a column with 2s indicating that are items are versions.
            """            
            self.data["nlabel"] = 2
        self.data["set_id_norm"] = self.data.set_id.rank(method='dense').astype(int) - 1
        
        # transforms
        self.text_transform = UnicodeNormalize()
        self.device = device
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        
        ref_item = self.__get_item_repr(self.data.iloc[idx])
        
        pos_set = self.data.query(f"set_id == {ref_item['set_id']} & yt_id != '{ref_item['yt_id']}'")
        neg_set = self.data.query(f"set_id != {ref_item['set_id']}")
        
        # sampling positive item
        pos_item = self.__get_item_repr(pos_set.sample(1).iloc[0])
        
        # sampling the negative items
        neg_items = []
        for i in range(self.neg_ratio):
            neg_items.append(self.__get_item_repr(neg_set.sample(1).iloc[0]))
            
        return ref_item, pos_item, neg_items
        
    def __get_item_repr(self, item):
        
        item = item.to_dict()
                    
        yt_id = item["yt_id"]    
        
        # get youtube metadata 
        item_yt = self.metadata.loc[self.metadata.yt_id == yt_id]
        item["yt_title"] = self.text_transform(item_yt.title.item()) if not item_yt.title.empty else ''
        item["yt_channel"] = self.text_transform(item_yt.channel_name.item()) if not item_yt.channel_name.empty else ''
        item["yt_description"] = self.text_transform(item_yt.description.item()) if not item_yt.description.empty else ''
        item["yt_keywords"] = [
                self.text_transform(keyword) for keyword in item_yt.keywords.item()
            ] if len(item_yt.keywords) > 0 else []
        
        return item
    
    def get_target_matrix(self):
        """Generates the binary square matrix of cover song relationships between all
        elements in the dataset.
        Returns:
            _type_: _description_
        """
        labels = torch.from_numpy(self.data["set_id"].values) 
        nlabels = torch.from_numpy(self.data["nlabel"].values)
        relevance = nlabels >= 2  

        mask = torch.logical_and(relevance[:, None], relevance)
        mask.fill_diagonal_(False)  

        target = torch.zeros(len(self.data), len(self.data), dtype=torch.int32)
        target[mask] = (labels[:, None] == labels).int()[mask]

        return target
    


import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.transforms import UnicodeNormalize


class OnlineCoverSongDataset(Dataset):
    """This class iterates on sample level:
    one iteration is one video or candidate version
    Args:
        OnlineCoverDataset (_type_): _description_
    """
    def __init__(
            self,
            dataset_name: str, # shs-yt, shs100k2_test, shs100k2_train, ...
            data_path: str, # path with datasets metadata
            metadata_file_path: str, # path to youtube metadata in parquet file
            device: str = "cuda"
        ) -> None:
            
        super().__init__()
        
        self.dataset_name = dataset_name
        
        self.data_path = data_path
        self.metadata_file_path = metadata_file_path
        
        # the youtube metadata
        self.metadata = pd.read_parquet(self.metadata_file_path).reset_index()
        # cleanup NoneType keywords
        self.metadata.keywords = self.metadata.keywords.apply(lambda x: [] if x is None else x)
        
        # the dataset
        self.data = pd.read_csv(
            os.path.join(self.data_path, dataset_name + '.csv'), sep=";")
        self.data = self.data.query("has_file")
        if "nlabel" not in self.data.columns:
            """If no nlabel is specified (as in datasets SHS100K, Da-Tacos, etc.)
            then we assign a column with 2s indicating that are items are versions.
            """            
            self.data["nlabel"] = 2
        self.data["set_id_norm"] = self.data.set_id.rank(method='dense').astype(int) - 1
        
        # transforms
        self.text_transform = UnicodeNormalize()
        self.device = device
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        
        item = dict(self.data.iloc[idx])
            
        yt_id = item["yt_id"]    
        
        # get youtube metadata 
        item_yt = self.metadata.loc[self.metadata.yt_id == yt_id]
        item["yt_title"] = self.text_transform(item_yt.title.item()) if not item_yt.title.empty else ''
        item["yt_channel"] = self.text_transform(item_yt.channel_name.item()) if not item_yt.channel_name.empty else ''
        item["yt_description"] = self.text_transform(item_yt.description.item()) if not item_yt.description.empty else ''
        item["yt_keywords"] = [
                self.text_transform(keyword) for keyword in item_yt.keywords.item()
            ] if len(item_yt.keywords) > 0 else []
        
        return item
    
    def get_target_matrix(self):
        """Generates the binary square matrix of cover song relationships between all
        elements in the dataset.
        Returns:
            _type_: _description_
        """
        labels = torch.from_numpy(self.data["set_id"].values) 
        nlabels = torch.from_numpy(self.data["nlabel"].values)
        relevance = nlabels >= 2  

        mask = torch.logical_and(relevance[:, None], relevance)
        mask.fill_diagonal_(False)  

        target = torch.zeros(len(self.data), len(self.data), dtype=torch.int32)
        target[mask] = (labels[:, None] == labels).int()[mask]

        return target
    
    def collate_fn(self, batch):
        
        # Filter out None values (rows with missing audio feature)
        batch = [item for item in batch if item is not None]
        
        # Prepare the rest of the data
        set_ids = [item['set_id'] for item in batch]
        set_ids_norm = [item['set_id_norm'] for item in batch]
        ver_ids = [item['ver_id'] for item in batch]
        yt_ids = [item['yt_id'] for item in batch]
        shs_titles = [item['title'] for item in batch]
        shs_performers = [item['performer'] for item in batch]
        yt_titles = [item['yt_title'] for item in batch]
        yt_channels = [item['yt_channel'] for item in batch]
        yt_descriptions = [item['yt_description'] for item in batch]
        yt_keywords = [item['yt_keywords'] for item in batch]


        return {
            'set_id': torch.tensor(set_ids),
            'set_id_norm': torch.tensor(set_ids_norm),
            'ver_id': torch.tensor(ver_ids),
            'shs_title': shs_titles,
            'shs_performer': shs_performers,
            'yt_id': yt_ids,
            'yt_title': yt_titles,
            'yt_channel': yt_channels,
            'yt_description': yt_descriptions,
            'yt_keywords': yt_keywords
        }
        
