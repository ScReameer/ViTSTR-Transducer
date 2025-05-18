from typing import Union
import torch
import torch.nn.functional as F
from torch import nn, Tensor

class CrossEntropyLossSequence(nn.Module):
    def __init__(self, ignore_index: int, reduction: str = 'mean') -> None:
        """
        Initializes the CrossEntropyLossSequence instance.

        Args:
            ignore_index (int): Specifies a target value that is ignored and
                does not contribute to the input gradient.
            reduction (str, optional): Specifies the reduction to apply to the output.
                it should be one of the following 'none', 'mean', or 'sum'.
                default 'mean'.
        """
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError(
                'Reduction {} not implemented.'.format(reduction)
                )
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(
            pred.contiguous().view(-1, pred.shape[-1]), 
            target.contiguous().view(-1), 
            ignore_index=self.ignore_index, 
            reduction=self.reduction
        )


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma,
        weights: Union[None, Tensor] = None,
        reduction: str = 'mean',
        ignore_index=-100,
        eps=1e-16
    ) -> None:
        """
        Initializes the FocalLoss instance.

        Args:
            gamma (float): The focal loss focusing parameter.
            weights (Union[None, Tensor], optional): Rescaling weight given to each class.
                If given, has to be a Tensor of size C. Defaults to None.
            reduction (str, optional): Specifies the reduction to apply to the output.
                it should be one of the following 'none', 'mean', or 'sum'.
                default 'mean'.
            ignore_index (int, optional): Specifies a target value that is ignored and
                does not contribute to the input gradient. Defaults to -100.
            eps (float, optional): smoothing to prevent log from returning inf.
                Defaults to 1e-16.
        """
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError(
                'Reduction {} not implemented.'.format(reduction)
                )
        assert weights is None or isinstance(weights, Tensor), \
            'weights should be of type Tensor or None, but {} given'.format(
                type(weights))
        self.reduction = reduction
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eps = eps
        self.weights = weights

    def _get_weights(self, target: Tensor) -> Tensor:
        if self.weights is None:
            return torch.ones(target.shape[0])
        weights = target * self.weights
        return weights.sum(dim=-1)

    def _process_target(
            self, target: Tensor, num_classes: int, mask: Tensor
            ) -> Tensor:
        
        #convert all ignore_index elements to zero to avoid error in one_hot
        #note - the choice of value 0 is arbitrary, but it should not matter as these elements will be ignored in the loss calculation
        target = target * (target!=self.ignore_index) 
        target = target.view(-1)
        return F.one_hot(target, num_classes=num_classes)

    def _process_preds(self, x: Tensor) -> Tensor:
        x = x.softmax(dim=-1)
        if x.dim() == 1:
            x = torch.vstack([1 - x, x])
            x = x.permute(1, 0)
            return x
        return x.view(-1, x.shape[-1])

    def _calc_pt(
            self, target: Tensor, x: Tensor, mask: Tensor
            ) -> Tensor:
        p = target * x
        p = p.sum(dim=-1)
        p = p * ~mask
        return p

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        mask = target == self.ignore_index
        mask = mask.view(-1)
        x = self._process_preds(x)
        num_classes = x.shape[-1]
        target = self._process_target(target, num_classes, mask)
        weights = self._get_weights(target).to(x.device)
        pt = self._calc_pt(target, x, mask)
        focal = 1 - pt
        nll = -torch.log(self.eps + pt)
        nll = nll.masked_fill(mask, 0)
        loss = weights * (focal ** self.gamma) * nll
        return self._reduce(loss, mask, weights)

    def _reduce(self, x: Tensor, mask: Tensor, weights: Tensor) -> Tensor:
        if self.reduction == 'mean':
            return x.sum() / (~mask * weights).sum()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x