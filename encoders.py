from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

import warnings
warnings.filterwarnings("ignore")

class PromptEncoder(nn.Module):
    def __init__(self, args, mask_token_id) -> None:
        super().__init__()
        self.device = args.device
        self.mask_token_id = mask_token_id
        self.encoder = AutoModel.from_pretrained(args.model_name)
        

    def forward(self, input_ids, attention_masks=None, task_type="mean"):
        if task_type == "mean":
            return self.get_mean_vector(input_ids, attention_masks)
        
        elif task_type == "weighted_mean":
            return self.get_weighted_mean_vector(input_ids, attention_masks)
        
        elif task_type == "cls":
            return self.get_cls_vector(input_ids, attention_masks)
        
        elif task_type == "prompt":
            return self.get_mask_vector(input_ids)
        
        else:
            raise Exception("TRANSFORMER ENCODING TYPE ERROR! OPTIONS: [mean, prompt]")
        

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    

    def attentive_mean_pooling(self, model_output, attention_mask, attention_weights):
        token_embeddings = model_output[0] 
        token_embeddings = token_embeddings[:, 1:, :]
        attention_mask = attention_mask[:, 1:]
        attention_weights = attention_weights[:, 1:]
        attention_weights = torch.softmax(attention_weights, dim=-1)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        input_weight_expanded = attention_weights.unsqueeze(-1).expand(token_embeddings.size()).float()
        input_mw_expanded = input_mask_expanded * input_weight_expanded
        #print(f'mask_size: {input_mask_expanded.size()} weight_size: {input_weight_expanded.size()} mw_size: {input_mw_expanded.size()}')
        return torch.sum(token_embeddings * input_mw_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        

    def get_mean_vector(self, input_ids, attention_masks):
        model_output = self.encoder(input_ids)
        sentence_embeddings = self.mean_pooling(model_output, attention_masks)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings
    

    def get_weighted_mean_vector(self, input_ids, attention_masks):
        model_output = self.encoder(input_ids, output_attentions=True)
        attention_weights = torch.mean(torch.sum(model_output[-1][-1].detach(), dim=-2), dim=1)
        sentence_embeddings = self.attentive_mean_pooling(model_output, attention_masks, attention_weights)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings
    
    def get_cls_vector(self, input_ids, attention_masks):
        model_output = self.encoder(input_ids)
        sentence_embeddings = model_output[0][:, 0, :]
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

      
    def get_mask_vector(self, input_ids):
        output = self.encoder(input_ids)
        mask_token_index = (input_ids == self.mask_token_id).nonzero()[:,1]
        mask_hidden_state = output[0][torch.arange(len(input_ids)), mask_token_index]
        mask_hidden_state = F.normalize(mask_hidden_state, p=2, dim=1)
        return mask_hidden_state
    

    def contrast_logits(self, embd1, embd2=None):
        feat1 = F.normalize(self.contrast_head(embd1), p=2, dim=1)
        if embd2 != None:
            feat2 = F.normalize(self.contrast_head(embd2), p=2, dim=1)
            return feat1, feat2
        else: 
            return feat1