import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


class SBert(nn.Module):
    def __init__(self, base_name, pooling='max', device='cuda', add_tokens=["[COL]", "[VAL]"]):
        super(SBert, self).__init__()
        self.base_name = base_name    
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_name)
        self.tokenizer.add_tokens(add_tokens, special_tokens=True)

        self.model = AutoModel.from_pretrained(self.base_name)
        self.pooling = self.__max_pooling if pooling == 'max' else self.__mean_pooling
        
    def forward(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        model_output = self.model(**encoded_input.to(self.device))
        pooled_output = self.pooling(model_output, encoded_input['attention_mask'])
        return pooled_output
        
    def __max_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return torch.max(token_embeddings, 1)[0]
    
    def __mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
