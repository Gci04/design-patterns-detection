
from transformers import AutoTokenizer, TFAutoModel, AutoModel
from pathlib import Path
from typing import List
import numpy as np
import torch
import tensorflow as tf


DEVICE =  'cuda' if torch.cuda.is_available() else 'cpu'



tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base").to(DEVICE)

PAD = tokenizer.pad_token
CLS = tokenizer.cls_token
SEP = tokenizer.sep_token
MAX_LEN = tokenizer.model_max_length
assert MAX_LEN == 512
BASE = Path('./coach_repos_zip')


def read_file(path):
    with open(path) as f:
        return f.readlines()


def pad_max_len(tok_ids: List[int], max_len = MAX_LEN) -> List[int]:
    """Pad sequence to max length i.e., 512"""
    pad_len = max_len - len(tok_ids) 
    padding = [tokenizer.convert_tokens_to_ids(PAD)] * pad_len
    return tok_ids + padding

def get_input_mask(toks_padded: List[int]):
    """Calculate attention mask
     - 1 for tokens that are **not masked**,
     - 0 for tokens that are **masked**."""

    return np.where(np.array(toks_padded) == tokenizer.convert_tokens_to_ids(PAD), 0, 1).tolist()

tokenized_seq = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("def max(a,b): if a>b: return a else return b"))
padded_seq = pad_max_len(tokenized_seq)
assert MAX_LEN == len(padded_seq)


def tokenize_sequence(code: str):
    
    code_tokens = tokenizer.tokenize(code)
    tokens = [tokenizer.cls_token] + [tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    if len(tokens_ids) >= MAX_LEN:
        tokens_ids = tokens_ids[:MAX_LEN]
    
    padded_token_ids = pad_max_len(tokens_ids)
    mask_ids = get_input_mask(padded_token_ids)
    return padded_token_ids, mask_ids



def embed(code: str):
    tok_ids, att_mask = tokenize_sequence(code)
    with torch.no_grad():
        context_embeddings = model(input_ids=torch.Tensor(tok_ids)[None, :].to(DEVICE).long(),\
                            attention_mask=torch.Tensor(att_mask)[None, :].to(DEVICE).long())[0]  #  [0] refers to last_hidden_states
    return context_embeddings[:,0, :]