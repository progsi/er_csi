import torch
from torchmetrics.retrieval import RetrievalNormalizedDCG, RetrievalMAP
from torchmetrics.retrieval import RetrievalHitRate, RetrievalMRR
from torchmetrics.retrieval import RetrievalPrecision, RetrievalRPrecision


class RetrievalEvaluation(object):
    """A class to ensure reproducibility of IR evaluation.
    Args:
        object (_type_): _description_
    """
    def __init__(self, top_k = 10, empty_target_action = 'skip', 
                 fill_diagonal_value: int = 0, device: str = 'cuda') -> None:
        
        self.top_k = top_k
        self.empty_target_action = empty_target_action # what to do with 0 relevance queries
        self.fill_diagonal_value = fill_diagonal_value # whether to fill similarity matrix
        self.device = device 
        
        self.nDCG = RetrievalNormalizedDCG(empty_target_action=self.empty_target_action)
        self.mAP = RetrievalMAP(empty_target_action=self.empty_target_action)
        self.H10 = RetrievalHitRate(top_k=self.top_k, empty_target_action=self.empty_target_action)
        self.MRR = RetrievalMRR(empty_target_action=self.empty_target_action)
        self.P10 = RetrievalPrecision(top_k=self.top_k, empty_target_action=self.empty_target_action)
        self.rP = RetrievalRPrecision(empty_target_action=self.empty_target_action)
        
    def pairwise_cosine_similarities(self, x, y=None):
        """
        This function is based on the implementation here:
        https://github.com/furkanyesiler/re-move
        Computing cosine similarity between the elements of two tensors
        :param x: first tensor
        :param y: second tensor (optional)
        :return: pairwise similarity matrix
        """
        if y is None:
            y = x
        preds = (x @ y.t()).div(y.norm(dim=1)).div(x.norm(dim=1).unsqueeze(-1))
        if type(self.fill_diagonal_value) == int:
            preds.fill_diagonal_(self.fill_diagonal_value)
        return preds

    def eval(self, target, preds1=None, emb_all1=None, emb_all2=None, preds2=None, weight2=0.5):
        """Compute all the matrix based on the embeddings of the dataset and the target.
        Args:
            emb_all (_type_): N embeddings
            target (_type_): N x N matrix with true binary relationships
        Returns:
            dict: result dict with metric names mapping to values
        """
        # N x N matrix
        if preds1 == None:
            preds1 = self.pairwise_cosine_similarities(emb_all1, emb_all2)
            # free cuda memory
            del emb_all1, emb_all2
            torch.cuda.empty_cache()

        if preds2 != None:
            # linear combination of preds
            preds = weight2 * preds2 + (1 - weight2) * preds1
        else:
            preds = preds1

        return self.compute_metrics(preds, target)
    
    def compute_metrics(self, preds, target, cls_based=False):
        """Computes various information retrieval metrics using torchmetrics.
        Args:
            preds (torch.tensor): similarity matrix MxN
            target (torch.tensor): true relationships matrix MxN
            cls_based: whether the evaluation is class-based. Then, we omit some
            metrics, since the results are note reasonable. We just retain results
            which are based on the first rank, such as MR1 and MRR.
        """
                # indexes for input structure for torchmetrics
        m, n = target.shape
        indexes = torch.arange(m).view(-1, 1).expand(-1, n).to(self.device)
        
        mrr_result = self.MRR(preds, target, indexes).item()
        mr1_result = self.__mean_rank_1(preds, target).item()
        
        # metrics which only refer to the first rank
        ir_dict = {
            "Queries": int(len(target)),
            "Relevant Items": int(torch.sum(target).item()),
            "MRR": mrr_result, 
            "MR1": mr1_result
        }
        
        del mr1_result, mrr_result
        torch.cuda.empty_cache()

        # metrics which concern the top 10 or whole ranking
        if not cls_based:
            # Create the dictionary with stored results
            non_cls_evals = {}

            map_result = self.mAP(preds, target, indexes).item()
            non_cls_evals["mAP"] = map_result
            # to free cuda memory
            del map_result 
            torch.cuda.empty_cache()

            # to free cuda memory
            del preds, target, indexes 
            torch.cuda.empty_cache()

            ir_dict.update(non_cls_evals)
        
        return dict(sorted(ir_dict.items()))
    
    def __mean_rank_1(self, preds, target):
        """
        Compute the mean rank for relevant items in the predictions.
        Args:
            preds (torch.Tensor): A tensor of predicted scores (higher scores indicate more relevant items).
            target (torch.Tensor): A tensor of true relationships (0 for irrelevant, 1 for relevant).
        Returns:
            float: The mean rank of relevant items for each query.
        """
        has_positives = torch.sum(target, 1) > 0
        
        _, spred = torch.topk(preds, preds.size(1), dim=1)
        found = torch.gather(target, 1, spred)
        temp = torch.arange(preds.size(1)).to(self.device).float() * 1e-6
        _, sel = torch.topk(found - temp, 1, dim=1)
        
        sel = sel.float()
        sel[~has_positives] = torch.nan
        
        mr1 = torch.nanmean((sel+1).float())

        del sel, found, temp, spred, has_positives
        torch.cuda.empty_cache()
        return mr1

