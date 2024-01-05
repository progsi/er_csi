import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.transforms import UnicodeNormalize
from transformers import AutoTokenizer, AutoConfig
import re


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
            tokenizer: AutoTokenizer,
            device: str = "cuda",
            attr_num: str = None,
            max_len: int = 256
        ) -> None:
            
        super().__init__()
        
        self.dataset_name = dataset_name
        
        self.tokenizer = tokenizer
        
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
        self.attr_num = attr_num
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        
        item = dict(self.data.iloc[idx])
            
        yt_id = item["yt_id"]    
        
        # get youtube metadata 
        item_yt = self.metadata.loc[self.metadata.yt_id == yt_id]
        item["video_title"] = self.text_transform(item_yt.title.item()) if not item_yt.title.empty else ''
        item["channel_name"] = self.text_transform(item_yt.channel_name.item()) if not item_yt.channel_name.empty else ''
        item["description"] = self.text_transform(item_yt.description.item()) if not item_yt.description.empty else ''
        item["keywords"] = ' '.join([
                self.text_transform(keyword) for keyword in item_yt.keywords.item()
            ]) if len(item_yt.keywords) > 0 else ''
        
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
        yt_titles = [item['video_title'] for item in batch]
        yt_channels = [item['channel_name'] for item in batch]
        yt_descriptions = [item['description'] for item in batch]
        yt_keywords = [item['keywords'] for item in batch]
        set_ids = [item['set_id'] for item in batch]
        set_ids_norm = [item['set_id_norm'] for item in batch]
        ver_ids = [item['ver_id'] for item in batch]

        return {
            'yt_id': yt_ids,
            'video_title': yt_titles,
            'channel_name': yt_channels,
            'description': yt_descriptions,
            'keywords': yt_keywords,
            'set_id': torch.tensor(set_ids),
            'set_id_norm': torch.tensor(set_ids_norm),
            'ver_id': torch.tensor(ver_ids)        
            }
        
    def serialize_item(self, item, cols):    
        
        COL_TOKEN = "[COL]"
        VAL_TOKEN = "[VAL]"
                
        tuple_list = [(COL_TOKEN, col, VAL_TOKEN, item[col]) for col in cols]
        side_tokenized = ' '.join([' '.join(t) for t in tuple_list])
        return side_tokenized

    def get_df(self):
        df = pd.merge(self.data, self.metadata, on='yt_id', how='left', suffixes=['_shs', '_yt'])
        df.title_yt = df.title_yt.fillna('').apply(self.text_transform)        
        df.channel_name = df.channel_name.fillna('').apply(self.text_transform)
        df.description = df.description.fillna('').apply(self.text_transform)
        df.keywords = df.keywords.fillna('')
        return df
    
    def get_dfs_by_task(self, task: str):
        
        left_cols, right_cols = self.__get_cols_by_task(task)
        df = self.get_df().rename({"title_yt": "video_title", "title_shs": "title"}, axis=1)
        return df[left_cols], df[right_cols]
    
    def getitem_tokenized(self, idx, side, task):
        
        item = self[idx]
        left_cols, right_cols = self.__get_cols_by_task(task)
        rel_cols = left_cols if side == "left" else right_cols
        
        serialized_text = self.serialize_item(item, rel_cols)
        tokenized = self.tokenizer.encode_plus(
                        serialized_text, 
                        add_special_tokens=True,
                        return_tensors="pt")
        return tokenized["input_ids"].to(self.device), tokenized["attention_mask"].to(self.device)

    def getitem_pair_tokenized(self, idx_left, idx_right, task):
        
        item_left = self[idx_left]
        item_right = self[idx_right]
        labels = torch.tensor(1 if item_left["set_id"] == item_right["set_id"] else 0)

        left_cols, right_cols = self.__get_cols_by_task(task)
        
        serialized_text_left = self.serialize_item(item_left, left_cols)
        serialized_text_right = self.serialize_item(item_right, right_cols)

        tokenized = self.tokenizer.encode(text=serialized_text_left,
                                  text_pair=serialized_text_right,
                                  max_length=256,
                                  truncation=True,
                                  return_tensors="pt")
        return tokenized, labels  
    
    def getitem_hiergat(self, idx_left, idx_right, task, split):
        
        item_left = self[idx_left]
        item_right = self[idx_right]

        left_cols, right_cols = self.__get_cols_by_task(task)
        
        serialized_text_left = self.serialize_item(item_left, left_cols).replace("[", "").replace("]", "")
        serialized_text_right = self.serialize_item(item_right, right_cols).replace("[", "").replace("]", "")

        sent = serialized_text_left + ' [SEP] ' + serialized_text_right
        label = torch.tensor(1 if item_left["set_id"] == item_right["set_id"] else 0).unsqueeze(0)
        
        items = [serialized_text_left, serialized_text_right, label]
        if split:
            attr_items = [item + ' COL' for item in items[0:-1]]
            attrs = []
            for attr_item in attr_items:
                attrs.append([f"COL {attr_str}" for attr_str in re.findall(r"(?<=COL ).*?(?= COL)", attr_item)])
            assert len(attrs[0]) == len(attrs[1])
        else:
            attrs = [[item] for item in items[0:-1]]

        xs = [self.tokenizer.encode(text=attrs[0][i], text_pair=attrs[1][i],
                                    add_special_tokens=True, truncation="longest_first", 
                                    max_length=self.max_len)
              for i in range(self.attr_num)]
        
        max_length = max(len(lst) for lst in xs)
        padded_lists = [lst + [0] * (max_length - len(lst)) for lst in xs]
        x =  torch.tensor(padded_lists)
        
        #left_zs = [self.tokenizer.encode(text=attrs[0][i], add_special_tokens=True,
        #                                 truncation="longest_first", max_length=self.max_len)
        #           for i in range(self.attr_num)]
        #right_zs = [self.tokenizer.encode(text=attrs[1][i], add_special_tokens=True,
        #                                  truncation="longest_first", max_length=self.max_len)
        #            for i in range(self.attr_num)]

        masks = [torch.zeros(self.tokenizer.vocab_size, dtype=torch.int)
                 for _ in range(self.attr_num)]
        for i in range(self.attr_num):
            masks[i][xs[i]] = 1
        masks = torch.stack(masks)

        #seqlens = [len(x) for x in xs]
        #left_zslens = [len(left_z) for left_z in left_zs]
        #right_zslens = [len(right_z) for right_z in right_zs]

        return sent, x.unsqueeze(0), label, masks.unsqueeze(0), attrs
      
    def __get_cols_by_task(self, task: str):
        
        if task == "svShort":
            right_cols = ["title", "performer"]
            left_cols = ["video_title", "channel_name"]
        elif task == "svShort+Tags":
            right_cols = ["title", "performer"]
            left_cols = ["video_title", "channel_name", "keywords"]
        elif task == "svLong":
            right_cols = ["title", "performer"]
            left_cols = ["video_title", "channel_name", "description"]
        elif task == "vvShort":
            left_cols = ["video_title", "channel_name"]
            right_cols = left_cols
        elif task == "vvShort+Tags":
            left_cols = ["video_title", "channel_name", "keywords"]
            right_cols = left_cols
        elif task == "vvLong":
            left_cols = ["video_title", "channel_name", "description"]
            right_cols = left_cols
        return left_cols, right_cols
        
    
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
        
