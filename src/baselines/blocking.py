from rapidfuzz import process, fuzz, distance
import torch

class Blocker(object):
    def __init__(self, blocker, workers=64, device="cuda") -> None:
        self.blocker = blocker
        self.workers = workers
        self.device = device
    
    def block(self, left_df, right_df):
        
        left_data = left_df.apply(lambda row: ' '.join(map(str, row)), axis=1)
        right_data = right_df.apply(lambda row: ' '.join(map(str, row)), axis=1)
        
        y = process.cdist(left_data, right_data, scorer=self.blocker, 
                                                  workers=self.workers)
        return torch.tensor(y).to(self.device)
        
