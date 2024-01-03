from rapidfuzz import process, fuzz, distance
import torch


SCORERS_DICT = {
    'ratio': fuzz.ratio,
    'token_ratio': fuzz.token_ratio,
    'partial_ratio': fuzz.partial_ratio,
    'token_set_ratio': fuzz.token_set_ratio,
    'partial_token_set_ratio': fuzz.partial_token_set_ratio,
    'token_sort_ratio': fuzz.token_sort_ratio,
    'partial_token_sort_ratio': fuzz.partial_token_sort_ratio,
    'normalized_similarity_DamerauLevenshtein': distance.DamerauLevenshtein.normalized_similarity,
    'normalized_similarity_JaroWinkler': distance.JaroWinkler.normalized_similarity,
    'normalized_similarity_LCSseq': distance.LCSseq.normalized_similarity,
    'normalized_similarity_Hamming': distance.Hamming.normalized_similarity,
    'WRatio': fuzz.WRatio,
    'QRatio': fuzz.QRatio
}
    
    
    
class Blocker(object):
    def __init__(self, blocking_func, workers=64, device="cuda") -> None:
        self.blocking_func = SCORERS_DICT[blocking_func]
        self.workers = workers
        self.device = device
    
    def block(self, left_df, right_df):
        
        left_data = left_df.apply(lambda row: ' '.join(map(str, row)), axis=1)
        right_data = right_df.apply(lambda row: ' '.join(map(str, row)), axis=1)
        
        y = process.cdist(left_data, right_data, scorer=self.blocker, 
                                                  workers=self.workers)
        return torch.tensor(y).to(self.device)
        
