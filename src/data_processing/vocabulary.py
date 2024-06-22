import torch
import string

class Vocabulary:
    def __init__(self):
        """Creates vocabulary for corpus of text"""
        # self.tokenizer = get_tokenizer("basic_english")
        self.idx2char = {0: "<PAD>", 1: "<START>", 2: "<END>"}
        self.char2idx = {"<PAD>": 0, "<START>": 1, "<END>": 2}
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
            self.char2idx[char]
            for char in lower_word
        ] + [self.char2idx['<END>']]
        return torch.tensor(output)
    
class TokenLabelConverter():
    """ Convert between text-label and text-index """

    def __init__(self):
        characters = string.digits + string.ascii_lowercase + string.punctuation
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        self.SPACE = '[s]'
        self.GO = '[GO]'
        batch_max_length = 68
        #self.MASK = '[MASK]'

        #self.list_token = [self.GO, self.SPACE, self.MASK]
        self.list_token = [self.GO, self.SPACE]
        self.character = self.list_token + list(characters)

        self.dict = {word: i for i, word in enumerate(self.character)}
        self.batch_max_length = batch_max_length + len(self.list_token)

    def __len__(self):
        return len(self.dict)
    
    def encode(self, text):
        """ convert text-label into text-index.
        """
        text = text.lower()
        length = [len(s) + len(self.list_token) for s in text]  # +2 for [GO] and [s] at end of sentence.
        batch_text = torch.LongTensor(len(text), self.batch_max_length).fill_(self.dict[self.GO])
        for i, t in enumerate(text):
            txt = [self.GO] + list(t) + [self.SPACE]
            txt = [self.dict[char] for char in txt]
            #prob = np.random.uniform()
            #mask_len = round(len(list(t)) * 0.15)
            #if is_train and mask_len > 0:
            #    for m in range(mask_len):
            #        index = np.random.randint(1, len(t) + 1)
            #        prob = np.random.uniform()
            #        if prob > 0.2:
            #            text[index] = self.dict[self.MASK]
            #            batch_weights[i][index] = 1.
            #        elif prob > 0.1: 
            #            char_index = np.random.randint(len(self.list_token), len(self.character))
            #            text[index] = self.dict[self.character[char_index]]
            #            batch_weights[i][index] = 1.
            batch_text[i][:len(txt)] = torch.LongTensor(txt)  # batch_text[:, 0] = [GO] token
        return batch_text

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i]for i in text_index[index, :]])
            texts.append(text)
        return texts