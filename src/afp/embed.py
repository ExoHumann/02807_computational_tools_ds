"""Fingerprint model: pretrained music backbone followed by a non-linear projection head (paper, Section 2.2).

The projection head is implemented. Loading a backbone and contrastive training are not.
"""

import torch
from torch import nn
from torch.nn import functional

BACKBONE_DIM = 1024
HIDDEN_DIM = 4096
EMBED_DIM = 256


class ProjectionHead(nn.Module):
    """Two-layer MLP ``z = W2 elu(W1 x + b1) + b2`` that maps backbone embeddings to fingerprints.

    Args:
        in_dim: Backbone embedding size.
        hidden_dim: Hidden layer width.
        out_dim: Fingerprint size.
    """

    def __init__(self, in_dim: int = BACKBONE_DIM, hidden_dim: int = HIDDEN_DIM, out_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project backbone embeddings to L2-normalised fingerprints.

        Args:
            x: Backbone embeddings of shape ``[batch, in_dim]``.

        Returns:
            Unit-norm fingerprints of shape ``[batch, out_dim]``.
        """
        return functional.normalize(self.net(x), dim=-1)


def load_backbone(name: str) -> nn.Module:
    """Load a pretrained music foundation model (e.g. MuQ or MERT) that maps audio to ``BACKBONE_DIM`` embeddings.

    Args:
        name: Identifier of the backbone.

    Returns:
        The backbone module.
    """
    raise NotImplementedError
