import torch


class Vocabulary:
    def __init__(self, labels: list[str] | str):
        """
        Args:
            labels (list[str] | str): List of labels or a single label string.
        """
        self.idx2token = {0: "<PAD>", 1: '<START>',  2: "<END>"}
        self.token2idx = {"<PAD>": 0, "<START>": 1, "<END>": 2}
        self.service_tokens = self.idx2token.copy()
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
    
    def encode(self, text: str) -> torch.Tensor:
        """Encodes a string into a sequence of tokens"""
        text = text.lower()
        for char in text:
            if char not in self.token2idx:
                text = text.replace(char, '')
        # <START> ... <END>
        output = [self.start_token_idx] + [self.token2idx[token] for token in text] + [self.end_token_idx]
        return torch.tensor(output)
    
    def decode(self, tensor: torch.Tensor) -> list[str]:
        """Decodes a sequence of tokens into a list of characters"""
        return [self.idx2token[idx] for idx in tensor.tolist()]
    
    @property
    def pad_token_idx(self) -> int:
        """Returns the index of the padding token"""
        return self.token2idx['<PAD>']
    
    @property
    def start_token_idx(self) -> int:
        """Returns the index of the start token"""
        return self.token2idx['<START>']
    
    @property
    def end_token_idx(self) -> int:
        """Returns the index of the end token"""
        return self.token2idx['<END>']
