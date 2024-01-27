from rapidfuzz import process, fuzz, distance
import torch

# Total masking strategies
# strategies: above (traditional), below, inner    


class Blocker(object):
    def __init__(self, blocking_func: str, threshold: float, workers: int = 64, device: str = "cuda", 
                 strategy: str = "above") -> None:
        self.blocking_func = blocking_func
        self.threshold = threshold
        self.strategy = strategy
        self.workers = workers
        self.device = device
        
    def predict(self, left_df, right_df):
        
        left_data = left_df.apply(lambda row: ' '.join(map(str, row)), axis=1)
        right_data = right_df.apply(lambda row: ' '.join(map(str, row)), axis=1)
        
        y = process.cdist(left_data, right_data, scorer=self.blocking_func, 
                                                workers=self.workers)
        return torch.tensor(y).to(self.device)
            
    def block(self, left_df, right_df):
        
        y = self.predict(left_df, right_df)
        
        # normalize if scores are between 0 and 100
        if torch.max(y).item() > 1:
            y = y / 100
        
        if self.strategy == "above":
            mask = torch.where(y > self.threshold, torch.tensor(1), torch.tensor(0))
        elif self.strategy == "below":
            mask = torch.where(y < self.threshold, torch.tensor(1), torch.tensor(0))
        else:
            raise NotImplementedError

        return mask, y
        
