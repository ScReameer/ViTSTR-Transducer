from pathlib import Path
import torch
import numpy as np
from ..data import Vocabulary


class TorchscriptViTSTRTransducer:
    def __init__(
            self, 
            weights_path: str | Path,
            vocab: Vocabulary,
            dtype: torch.dtype,
            device: str | torch.device
        ) -> None:
        """
        Initialize the model with weights and vocabulary.

        Args:
            weights_path (str | Path): The path to the Torchscript model weights.
            vocab (Vocabulary): The vocabulary object containing character-to-index mappings.
            dtype (torch.dtype): The data type for the model inputs and outputs.
            device (str | torch.device): The device to run the model on, e.g., 'cpu' or 'cuda'.
        """
        self.model = torch.jit.load(weights_path, map_location=device)
        self.vocab = vocab
        self.device = device
        self.dtype = dtype

    @torch.inference_mode()
    def predict(self, x: torch.Tensor, max_length: int = 30) -> tuple[list[str], list[torch.Tensor]]:
        """
        Predict the text and confidence scores for a batch of input tensors.
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length).
            max_length (int): The maximum length of the predicted text.
        Returns:
            tuple[list[str], list[torch.Tensor]]: A tuple containing the predicted text and confidence scores.
        """
        x = x.to(self.device).to(self.dtype) # Autocast inputs
        y_input = torch.tensor([[self.vocab.start_token_idx] for _ in range(len(x))], dtype=torch.int, device=self.device)
        for _ in range(max_length):
            pred: torch.Tensor = self.model(x, y_input)
            next_item = pred.argmax(-1)[..., -1:]
            if torch.all(next_item.reshape(-1) == self.vocab.end_token_idx):
                break
            y_input = torch.cat((y_input, next_item), dim=1)
        confidence = torch.softmax(pred, dim=-1).max(-1).values
        output_prediction = []
        output_confidence = []
        for i in range(y_input.shape[0]):
            del_indexes = []
            # ['<START>', 'something', '...', '<END>', '<END>', '<END>']
            result = self.vocab.decode(y_input[i]) # [N, ]
            conf = confidence[i] # [N, ]
            end_idx = next((i for i, s in enumerate(result) if self.vocab.idx2token[self.vocab.end_token_idx] in s), None)
            if end_idx:
                result = result[:end_idx]
                conf = conf[:end_idx]
            for token in self.vocab.service_tokens.values():
                for j in range(len(result)):
                    if result[j] == token:
                        del_indexes.append(j)
            result = ''.join(np.delete(result, del_indexes))
            conf = np.delete(conf.half().cpu(), del_indexes)
            output_prediction.append(result)
            output_confidence.append(conf)
        return output_prediction, output_confidence