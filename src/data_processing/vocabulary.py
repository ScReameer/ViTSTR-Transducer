import torch
import string

class Vocabulary:
    def __init__(self):
        """Creates vocabulary for corpus of text"""
        # self.tokenizer = get_tokenizer("basic_english")
        self.idx2char = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: '<UNK>'}
        self.char2idx = {"<PAD>": 0, "<START>": 1, "<END>": 2, '<UNK>': 3}
        self.vocab = string.digits + string.ascii_lowercase + string.punctuation + ' '
        self._build()
        del self.vocab

    def __len__(self):
        return len(self.idx2char)

    def _build(self):
        """Builds vocabulary"""
        start_idx = len(self.idx2char)
        self.idx2char.update({idx: char for idx, char in enumerate(self.vocab, start_idx)})
        self.char2idx.update({char: idx for idx, char in enumerate(self.vocab, start_idx)})
    
    def numericalize(self, word: str):
        """Processing raw word into tensor

        Args:
            `word` (`str`): raw word

        Returns:
            `output` (`torch.Tensor`): tensor representation of word with shape `[word_length]`
        """
        lower_word = word.lower()
        # <START> ... <END>
        output = [self.char2idx['<START>']] + [
            self.char2idx[char] if char in self.char2idx else self.char2idx['<UNK>']
            for char in lower_word
        ] + [self.char2idx['<END>']]
        return torch.tensor(output)