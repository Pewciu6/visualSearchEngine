import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates Triplet Loss for batch.
        Args:
            anchor: [Batch, Embedding_Size]
            positive: [Batch, Embedding_Size]
            negative: [Batch, Embedding_Size]
        """
        distance_positive = (anchor - positive).pow(2).sum(dim=1)
        distance_negative = (anchor - negative).pow(2).sum(dim=1)

        losses = F.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()
