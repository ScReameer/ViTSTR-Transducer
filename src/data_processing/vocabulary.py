import torch
import pandas as pd

class Vocabulary:
    def __init__(self, text: pd.Series):
        """Creates vocabulary for corpus of text

        Args:
            `text` (`pd.Series`): column from dataframe with captions
            `freq_threshold` (`int`, optional): write words in vocabulary if word frequency >= `freq_threshold`, else word becomes `<UNK>`. Defaults to `5`.
        """
        # self.tokenizer = get_tokenizer("basic_english")
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>"}
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
        self.text: pd.Series = text.copy().astype(str).apply(lambda x: x.lower())
        self._build()
        del self.text

    def __len__(self):
        return len(self.idx2word)

    def _build(self):
        """Builds vocabulary"""
        start_idx = 3
        for sentence in self.text.tolist():
            for word in sentence:
                if word not in self.word2idx:
                    self.word2idx[word] = start_idx
                    self.idx2word[start_idx] = word
                    start_idx += 1
    
    def numericalize(self, text: str):
        """Processing raw text into tensor

        Args:
            `text` (`str`): raw string of sentence

        Returns:
            `output` (`torch.Tensor`): tensor representation of sentence with shape `[sequence_length]`
        """
        tokenized_text = list(text.lower())
        # <SOS> ... <EOS>
        output = [self.word2idx['<SOS>']] + [
            self.word2idx[token] for token in tokenized_text
        ] + [self.word2idx['<EOS>']]
        return torch.tensor(output)