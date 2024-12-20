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
            max_len: int = 256,
            nsample: int = None,
            only_original: bool = False
        ) -> None:
            
        super().__init__()
        
        self.dataset_name = dataset_name
        self.parent_dataset_name = self.__get_parent_dataset_name(self.dataset_name)
        self.only_original = only_original
        
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
        
        if not self.dataset_name in ["shs100k2_test_one", "shs100k2_test_one2"]: 
            self.data = self.data.query("has_file")
        
        if nsample is not None:
            self.data = self.data.tail(nsample)
            
        if "nlabel" not in self.data.columns:
            """If no nlabel is specified (as in datasets SHS100K, Da-Tacos, etc.)
            then we assign a column with 2s indicating that are items are versions.
            """            
            self.data["nlabel"] = 2
        self.data["set_id_norm"] = self.data.set_id.rank(method='dense').astype(int) - 1
        
        # for DaTacos
        if "perf_title" in self.data:
            self.data["title"] = self.data["perf_title"]
            self.data.drop("perf_title", axis=1)
        if "perf_artist" in self.data:
            self.data["performer"] = self.data["perf_artist"]
            self.data.drop("perf_artist", axis=1)

        if self.only_original:
            self.data = pd.merge(self.data, self.__get_originals_df()[["set_id", "title", "performer"]], on="set_id", suffixes=["_indiv", ""])
        # transforms
        self.text_transform = UnicodeNormalize()
        self.device = device
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        
        item = dict(self.data.iloc[idx])

        yt_id = item["yt_id"]    
        
        # get youtube metadata 
        if type(item["set_id"]) == str or item["set_id"] >= 0: # normal dataset ID
            item_yt = self.metadata.loc[self.metadata.yt_id == yt_id]
            item["video_title"] = self.text_transform(item_yt.title.item()) if not item_yt.title.empty else ''
            item["channel_name"] = self.text_transform(item_yt.channel_name.item()) if not item_yt.channel_name.empty else ''
            item["description"] = self.text_transform(item_yt.description.item()) if not item_yt.description.empty else ''
            item["keywords"] = ' '.join([
                    self.text_transform(keyword) for keyword in item_yt.keywords.item()
                ]) if len(item_yt.keywords) > 0 else ''
        else:
            item["video_title"] = item["title_generated"]
            item["channel_name"] = self.metadata.sample(n=1).channel_name.item() # sample random channel name
            item["description"] = ''
            item["keywords"] = ''

        return item
    
    def __get_originals_df(self):
        def __update_columns(group):
            first_ver_id = group['ver_id'].iloc[0]
            group['ver_id'] = first_ver_id
            group['title'] = group['title'].iloc[0]
            group['performer'] = group['performer'].iloc[0]
            return group

        # Apply the function to each group of set_id
        data_originals = self.data.groupby('set_id').apply(__update_columns)

        # Drop duplicate rows (keeping only the first ver_id in each set_id)
        data_originals = data_originals.drop_duplicates(subset='set_id')

        # Reset index if needed
        data_originals = data_originals.reset_index(drop=True)
        return data_originals
    
    def get_target_matrix(self):
        """Generates the binary square matrix of cover song relationships between all
        elements in the dataset.
        Returns:
            _type_: _description_
        """
        labels = torch.from_numpy(self.data["set_id_norm"].values) 
        
        try: # FIXME: help
            ids = torch.from_numpy(self.data["set_id"].str.replace("W_", "").astype(int).values)
        except:
            ids = torch.from_numpy(self.data["set_id"].values)
        
        nlabels = torch.from_numpy(self.data["nlabel"].values)
        relevance = nlabels >= 2  

        mask = torch.logical_and(relevance[:, None], relevance)
        mask.fill_diagonal_(False)  

        target = torch.zeros(len(self.data), len(self.data), dtype=torch.int32)
        target[mask] = (labels[:, None] == labels).int()[mask]

        # non negative mask for generated titles
        non_negative_mask = (ids > 0).int().unsqueeze(0) * (ids > 0).int().unsqueeze(1)
        target = torch.where(non_negative_mask > 0, target, 0)

        return target
    
    def __get_parent_dataset_name(self, dataset_name):
        parent_datasets = ["shs100k2_test", "shs100k2_val", "da-tacos"]
        for d in parent_datasets:
            if d in dataset_name:
                return d
        return dataset_name
            
    def get_csi_pred_matrix(self, model_name: str):
        
        dataset_name = self.parent_dataset_name
        
        preds_yt_ids = pd.read_csv(
            os.path.join("preds", model_name, dataset_name, "data.csv"), 
            sep=";").yt_id.to_list()
        
        preds_tensor = torch.load(os.path.join("preds", model_name, dataset_name, "ypred.pt")).cpu()

        preds = pd.merge(
            self.data[["yt_id"]], 
            pd.DataFrame(
                preds_tensor, 
                index=preds_yt_ids, 
                columns=preds_yt_ids).reset_index(names="yt_id").drop_duplicates(subset="yt_id"),
            how="left").set_index("yt_id")
        if len(preds.columns) > len(preds):
            preds = preds[preds.index]
        preds = preds.loc[:, ~preds.columns.duplicated()]
        preds = preds.reindex(index=preds.index, columns=preds.index)
        
        return torch.tensor(preds.values).to(self.device)
    
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
        
    def serialize_item(self, item, cols, mask_shs=False, mask_yt=False):    
        
        COL_TOKEN = "[COL]"
        VAL_TOKEN = "[VAL]"
                
        tuple_list = [(COL_TOKEN, col, VAL_TOKEN, item[col]) for col in cols]
        # mask shs info
        if mask_shs:
            for i, (col_tok, col, val_tok, val) in enumerate(tuple_list):
                if col in ["performer", "title"]:
                    val = "[MASK]"
                tuple_list[i] = (col_tok, col, val_tok, val)
        # mask shs info
        if mask_yt:
            for i, (col_tok, col, val_tok, val) in enumerate(tuple_list):
                if col in ["video_title", "channel_name", "description", "keywords"]:
                    val = "[MASK]"
                tuple_list[i] = (col_tok, col, val_tok, val)

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
        
        mask_shs = side == "right"
        serialized_text = self.serialize_item(item, rel_cols, mask_shs=mask_shs)
        tokenized = self.tokenizer.encode_plus(
                        serialized_text, 
                        add_special_tokens=True,
                        return_tensors="pt",
                        max_length=256)
        return tokenized["input_ids"].to(self.device), tokenized["attention_mask"].to(self.device)

    def getitem_pair_tokenized(self, idx_left, idx_right, task):
        
        item_left = self[idx_left]
        item_right = self[idx_right]
        labels = torch.tensor(1 if item_left["set_id"] == item_right["set_id"] else 0)

        left_cols, right_cols = self.__get_cols_by_task(task)
        
        mask_yt = "xLong" in task or "xShort" in task
        serialized_text_left = self.serialize_item(item_left, left_cols, mask_yt=mask_yt)
        mask_shs =  "xLong" in task or "xShort" in task or  "rLong" in task or "rShort" in task
        serialized_text_right = self.serialize_item(item_right, right_cols, mask_shs=mask_shs)

        tokenized = self.tokenizer.encode(text=serialized_text_left,
                                  text_pair=serialized_text_right,
                                  max_length=256,
                                  truncation=True,
                                  return_tensors="pt")
        return tokenized, labels  
      
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
        elif task == "rShort" or task == "xShort":
            left_cols = ["title", "performer", "video_title", "channel_name"]
            right_cols = left_cols
        elif task == "rLong" or task == "xLong":
            left_cols = ["title", "performer", "video_title", "channel_name", "keywords", "description"]
            right_cols = left_cols
        elif task == "tvShort":
            left_cols = ["title"]
            right_cols = ["video_title"]
        elif task == "tvLong":
            left_cols = ["title"]
            right_cols = ["video_title", "channel_name", "description"]
        elif task == "tvShort+Tags":
            left_cols = ["title"]
            right_cols = ["video_title", "channel_name", "keywords"]
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

        if not self.dataset_name in ["shs100k2_test_one", "shs100k2_test_one2"]: 
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
        labels = torch.from_numpy(self.data["set_id_norm"].values) 
        try: # FIXME: help
            ids = torch.from_numpy(self.data["set_id"].str.replace("W_", "").astype(int).values)
        except:
            ids = torch.from_numpy(self.data["set_id"].values)

        nlabels = torch.from_numpy(self.data["nlabel"].values)
        relevance = nlabels >= 2  

        mask = torch.logical_and(relevance[:, None], relevance)
        mask.fill_diagonal_(False)  

        target = torch.zeros(len(self.data), len(self.data), dtype=torch.int32)
        target[mask] = (labels[:, None] == labels).int()[mask]

        # non negative mask for generated titles
        non_negative_mask = (ids > 0).int().unsqueeze(0) * (ids > 0).int().unsqueeze(1)
        target = torch.where(non_negative_mask > 0, target, 0)

        return target
    

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
            task: str, # determines attr pairs
            device: str = "cuda",
            only_original: bool = False # whether to use only item of original on the query side.
        ) -> None:
            
        super().__init__()
        
        self.dataset_name = dataset_name
        self.only_original = only_original

        self.data_path = data_path
        self.metadata_file_path = metadata_file_path
        self.task = task
        self.left_cols, self.right_cols = self.get_cols_by_task()

        # the youtube metadata
        self.metadata = pd.read_parquet(self.metadata_file_path).reset_index()
        # cleanup NoneType keywords
        self.metadata.keywords = self.metadata.keywords.apply(lambda x: [] if x is None else x)
        
        # the dataset
        self.data = pd.read_csv(
            os.path.join(self.data_path, dataset_name + '.csv'), sep=";")
                
        if not self.dataset_name in ["shs100k2_test_one", "shs100k2_test_one2"]: 
            self.data = self.data.query("has_file")

        if "nlabel" not in self.data.columns:
            """If no nlabel is specified (as in datasets SHS100K, Da-Tacos, etc.)
            then we assign a column with 2s indicating that are items are versions.
            """            
            self.data["nlabel"] = 2
        self.data["set_id_norm"] = self.data.set_id.rank(method='dense').astype(int) - 1
        
                # for DaTacos
        if "perf_title" in self.data:
            self.data["title"] = self.data["perf_title"]
            self.data.drop("perf_title", axis=1)
        if "perf_artist" in self.data:
            self.data["performer"] = self.data["perf_artist"]
            self.data.drop("perf_artist", axis=1)
        if self.data['set_id'].apply(lambda x: isinstance(x, str)).any():
            self.data['set_id'] = self.data['set_id'].str.replace("W_", "").astype(int)
        if self.data['ver_id'].apply(lambda x: isinstance(x, str)).any():
            self.data['ver_id'] = self.data['ver_id'].str.replace("P_", "").astype(int)

        if self.only_original:
            self.data = pd.merge(self.data, self.__get_originals_df()[["set_id", "title", "performer"]], on="set_id", suffixes=["_indiv", ""])      
        
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
        item["video_title"] = self.text_transform(item_yt.title.item()) if not item_yt.title.empty else ''
        item["channel_name"] = self.text_transform(item_yt.channel_name.item()) if not item_yt.channel_name.empty else ''
        item["description"] = self.text_transform(item_yt.description.item()) if not item_yt.description.empty else ''
        item["keywords"] = item["keywords"] = ' '.join([
                self.text_transform(keyword) for keyword in item_yt.keywords.item()
            ]) if len(item_yt.keywords) > 0 else ''
        
        item["left_side"] = ' '.join([' '.join(("[COL]", col, "[VAL]", item[col])) for col in self.left_cols])
        item["right_side"] = ' '.join([' '.join(("[COL]", col, "[VAL]", item[col])) for col in self.right_cols])

        return item
     
    def __get_originals_df(self):
        def __update_columns(group):
            first_ver_id = group['ver_id'].iloc[0]
            group['ver_id'] = first_ver_id
            group['title'] = group['title'].iloc[0]
            group['performer'] = group['performer'].iloc[0]
            return group

        # Apply the function to each group of set_id
        data_originals = self.data.groupby('set_id').apply(__update_columns)

        # Drop duplicate rows (keeping only the first ver_id in each set_id)
        data_originals = data_originals.drop_duplicates(subset='set_id')

        # Reset index if needed
        data_originals = data_originals.reset_index(drop=True)
        return data_originals
    

    def get_target_matrix(self):
        """Generates the binary square matrix of cover song relationships between all
        elements in the dataset.
        Returns:
            _type_: _description_
        """
        labels = torch.from_numpy(self.data["set_id_norm"].values) 
        
        try: # FIXME: help
            ids = torch.from_numpy(self.data["set_id"].str.replace("W_", "").astype(int).values)
        except:
            ids = torch.from_numpy(self.data["set_id"].values)
        
        nlabels = torch.from_numpy(self.data["nlabel"].values)
        relevance = nlabels >= 2  

        mask = torch.logical_and(relevance[:, None], relevance)
        mask.fill_diagonal_(False)  

        target = torch.zeros(len(self.data), len(self.data), dtype=torch.int32)
        target[mask] = (labels[:, None] == labels).int()[mask]

        # non negative mask for generated titles
        non_negative_mask = (ids > 0).int().unsqueeze(0) * (ids > 0).int().unsqueeze(1)
        target = torch.where(non_negative_mask > 0, target, 0)

        return target

    def __get_parent_dataset_name(self, dataset_name):
        parent_datasets = ["shs100k2_test", "shs100k2_val", "da-tacos"]
        for d in parent_datasets:
            if d in dataset_name:
                return d
        return dataset_name
            
    def get_csi_pred_matrix(self, model_name: str):
        
        dataset_name = self.__get_parent_dataset_name(self.dataset_name)
        
        preds_yt_ids = pd.read_csv(
            os.path.join("preds", model_name, dataset_name, "data.csv"), 
            sep=";").yt_id.to_list()
        
        preds_tensor = torch.load(os.path.join("preds", model_name, dataset_name, "ypred.pt")).cpu()

        preds = pd.merge(
            self.data[["yt_id"]], 
            pd.DataFrame(
                preds_tensor, 
                index=preds_yt_ids, 
                columns=preds_yt_ids).reset_index(names="yt_id").drop_duplicates(subset="yt_id"),
            how="left").set_index("yt_id")
        if len(preds.columns) > len(preds):
            preds = preds[preds.index]
        preds = preds.loc[:, ~preds.columns.duplicated()]
        preds = preds.reindex(index=preds.index, columns=preds.index)
        
        return torch.tensor(preds.values).to(self.device)
    
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
        yt_titles = [item['video_title'] for item in batch]
        yt_channels = [item['channel_name'] for item in batch]
        yt_descriptions = [item['description'] for item in batch]
        yt_keywords = [item['keywords'] for item in batch]
        left_sides = [item['left_side'] for item in batch]
        right_sides = [item['right_side'] for item in batch]

        return {
            'set_id': torch.tensor(set_ids),
            'set_id_norm': torch.tensor(set_ids_norm),
            'ver_id': torch.tensor(ver_ids),
            'title': shs_titles,
            'performer': shs_performers,
            'yt_id': yt_ids,
            'video_title': yt_titles,
            'channel_name': yt_channels,
            'description': yt_descriptions,
            'keywords': yt_keywords,
            'left_side': left_sides,
            'right_side': right_sides
        }

    def get_cols_by_task(self):
        
        if self.task == "svShort":
            right_cols = ["title", "performer"]
            left_cols = ["video_title", "channel_name"]
        elif self.task == "svShort+Tags":
            right_cols = ["title", "performer"]
            left_cols = ["video_title", "channel_name", "keywords"]
        elif self.task == "svLong":
            right_cols = ["title", "performer"]
            left_cols = ["video_title", "channel_name", "description"]
        elif self.task == "tvShort":
            right_cols = ["title"]
            left_cols = ["video_title", "channel_name"]
        elif self.task == "tvShort+Tags":
            right_cols = ["title"]
            left_cols = ["video_title", "channel_name", "keywords"]
        elif self.task == "tvLong":
            right_cols = ["title"]
            left_cols = ["video_title", "channel_name", "description"]
        elif self.task == "vvShort":
            left_cols = ["video_title", "channel_name"]
            right_cols = left_cols
        elif self.task == "vvShort+Tags":
            left_cols = ["video_title", "channel_name", "keywords"]
            right_cols = left_cols
        elif self.task == "vvLong":
            left_cols = ["video_title", "channel_name", "description"]
            right_cols = left_cols
        return left_cols, right_cols
    
        
