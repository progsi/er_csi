from rapidfuzz import process, fuzz, distance
import torch

# Total masking strategies
# strategies: above (traditional), below, inner    


class Blocker(object):
    def __init__(self, blocking_func: str, threshold: float, workers: int = 64, device: str = "cuda", 
                 strategy: str = "above", k: int = None) -> None:
        self.blocking_func = blocking_func
        self.threshold = threshold
        self.k = k
        self.strategy = strategy
        self.workers = workers
        self.device = device
        
    def predict(self, left_df, right_df):
        
        left_data = left_df.apply(lambda row: ' '.join(map(str, row)), axis=1)
        right_data = right_df.apply(lambda row: ' '.join(map(str, row)), axis=1)
        
        y = process.cdist(left_data, right_data, scorer=self.blocking_func, 
                                                workers=self.workers)
        return torch.tensor(y).to(self.device)
    
    def block_fuzzy(self, left_df, right_df):
        
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
        
    def block_topk(self, preds):

        if self.strategy not in ["top", "lowest"]:
            raise ValueError("Invalid strategy. Please choose 'top' or 'lowest'.")

        # Get the indices of the top or lowest k values per row
        largest = self.strategy == "top"
        _, indices = torch.topk(preds, self.k, dim=1, largest=largest, sorted=True)

        row_indices = torch.arange(indices.size(0)).unsqueeze(1).expand_as(indices).to(indices.device)

        # Create the 2D matrix with row indices in the first column
        result_matrix = torch.stack([row_indices.flatten(), indices.flatten()], dim=1)

        return result_matrix.view(-1, 2)
