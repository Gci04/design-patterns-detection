"""Use this file to generate embeddings against projects 
embed_tf: Use this function to encode codes
"""

from typing import List
import numpy as np
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf


class EmbeddingsGeneration:
    
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = TFAutoModel.from_pretrained("microsoft/codebert-base")

        self.PAD = self.tokenizer.pad_token
        self.CLS = self.tokenizer.cls_token
        self.SEP = self.tokenizer.sep_token
        self.MAX_LEN = self.tokenizer.model_max_length
        assert self.MAX_LEN == 512
        self._assert_embeddings()

    def _pad_max_len(self, tok_ids: List[int]) -> List[int]:
        """Pad sequence to max length i.e., 512"""
        pad_len = self.MAX_LEN - len(tok_ids) 
        padding = [self.tokenizer.convert_tokens_to_ids(self.PAD)] * pad_len
        return tok_ids + padding

    def _get_input_mask(self, toks_padded: List[int]):
        """Calculate attention mask
        - 1 for tokens that are **not masked**,
        - 0 for tokens that are **masked**."""
        return np.where(np.array(toks_padded) == self.tokenizer.convert_tokens_to_ids(self.PAD), 0, 1).tolist()
    
    def _assert_embeddings(self):
        """Assert if embeddings are correct"""
        tokenized_seq = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("def max(a,b): if a>b: return a else return b"))
        padded_seq = self._pad_max_len(tokenized_seq)
        assert self.MAX_LEN == len(padded_seq)


    def _tokenize_sequence(self, code: str):
        """tokenize code sequence and return token-idxs and mask-idxs"""
        code_tokens = self.tokenizer.tokenize(code)
        tokens = [self.tokenizer.cls_token] + [self.tokenizer.sep_token] + code_tokens + [self.tokenizer.sep_token]
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        if len(tokens_ids) >= self.MAX_LEN:
            tokens_ids = tokens_ids[:self.MAX_LEN]
        
        padded_token_ids = self.pad_max_len(tokens_ids)
        mask_ids = self._get_input_mask(padded_token_ids)
        return padded_token_ids, mask_ids


    def embed_tf(self, codes: List[str]):
        """encodes the input list of codes (typically a single android project) as an embedding vector"""
        embeddings = []
        for code in codes:
            tok_ids, att_mask = self._tokenize_sequence(code)
            context_embeddings = self.model(input_ids=tf.convert_to_tensor(tok_ids)[None, :],\
                            attention_mask=tf.convert_to_tensor(att_mask)[None, :])[0]  #  [0] refers to last_hidden_states
            
            if not embeddings:
                embeddings.append(context_embeddings[:,0, :])
            else:
                embeddings.append(context_embeddings[:,0, :])
                # TODO: implement max-pooling strategy
                emb_mean = tf.reduce_sum(tf.stack(embeddings),  axis=0)  # sum-pooling
                embeddings = [emb_mean]
        if len(embeddings) == 1:
            return tf.convert_to_tensor(embeddings[0])
        return tf.squeeze(embeddings[0], axis=0)
