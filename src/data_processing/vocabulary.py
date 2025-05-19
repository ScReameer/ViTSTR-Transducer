import torch
import re

class Vocabulary:
    def __init__(self, labels: list):
        """
        Initializes a Vocabulary object with the given list of labels.
        
        Args:
            `labels` (`list`): A list of labels to be used in the vocabulary.
        """
        self.idx2token = {0: "<PAD>", 1: '<START>',  2: "<END>"}
        self.token2idx = {"<PAD>": 0, "<START>": 1, "<END>": 2}
        self.labels = labels
        self._build()
        del self.labels

    def __len__(self):
        return len(self.idx2token)

    def _build(self):
        """Builds vocabulary"""
        start_idx = len(self.idx2token)
        self.idx2token.update({idx: char for idx, char in enumerate(self.labels, start_idx)})
        self.token2idx.update({char: idx for idx, char in enumerate(self.labels, start_idx)})
    
    def encode(self, text: str):
        text = text.lower()
        for char in text:
            if char not in self.token2idx:
                text = text.replace(char, '')
        # <START> ... <END>
        output = [self.token2idx['<START>']] + [self.token2idx[digit]for digit in text] + [self.token2idx['<END>']]
        return torch.tensor(output)
    
    def decode(self, tensor: torch.Tensor):
        return ''.join([self.idx2token[idx] for idx in tensor.tolist()])
    
    @property
    def pad_token_idx(self):
        return self.token2idx['<PAD>']
    
    @property
    def start_token_idx(self):
        return self.token2idx['<START>']
    
    @property
    def end_token_idx(self):
        return self.token2idx['<END>']
