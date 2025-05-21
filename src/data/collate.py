import torch
from torch.nn.utils.rnn import pad_sequence


class Collate:
    def __init__(self, pad_idx) -> None:
        """
        Initializes a Collate object with the given pad index.

        Args:
            pad_idx (int): The index to use for padding.
        """
        self.pad_idx = pad_idx
    
    def __call__(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Collates a batch of data into tensors.
        Args:
            batch (list[tuple]): A list of tuples where each tuple contains an image and its corresponding target.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the collated images and targets.
        """
        imgs = [item[0][None, ...] for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, padding_value=self.pad_idx, batch_first=True)
        return imgs, targets