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
    
