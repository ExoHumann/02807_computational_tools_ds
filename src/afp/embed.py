"""Fingerprint model: pretrained MuQ backbone followed by a non-linear projection head (paper, Section 2.2).

The backbone is frozen and used only to extract features. The projection head is trained contrastively so that a
window and a distorted copy of it get nearby fingerprints.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional

BACKBONE_DIM = 1024
HIDDEN_DIM = 4096
EMBED_DIM = 256
MUQ_REPO = "OpenMuQ/MuQ-large-msd-iter"


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


class MuQBackbone:
    """Frozen MuQ model that turns 24 kHz audio windows into one embedding each.

    The last hidden layer is averaged over time. The paper does not say which layer or pooling it uses.

    Args:
        device: ``"cuda"`` or ``"cpu"``. Defaults to CUDA when available.
        repo: Hugging Face id of the MuQ checkpoint.
    """

    def __init__(self, device: str | None = None, repo: str = MUQ_REPO) -> None:
        from muq import MuQ

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MuQ.from_pretrained(repo).to(self.device).eval()

    @torch.no_grad()
    def encode(self, waveforms: np.ndarray) -> np.ndarray:
        """Embed a batch of audio windows.

        Args:
            waveforms: ``float32`` array of shape ``[n, samples]`` sampled at 24 kHz.

        Returns:
            ``float32`` array of shape ``[n, BACKBONE_DIM]``.
        """
        output = self.model(torch.from_numpy(waveforms).to(self.device), output_hidden_states=True)
        return output.last_hidden_state.mean(dim=1).float().cpu().numpy()


def info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric contrastive loss: row ``i`` of ``a`` must match row ``i`` of ``b`` against every other row.

    Args:
        a: Unit-norm fingerprints of shape ``[batch, dim]``.
        b: Unit-norm fingerprints of the distorted copies, same shape.
        temperature: Softmax temperature. Smaller values punish near-misses harder.

    Returns:
        A scalar loss.
    """
    logits = a @ b.T / temperature
    labels = torch.arange(len(a), device=a.device)
    return (functional.cross_entropy(logits, labels) + functional.cross_entropy(logits.T, labels)) / 2


def train_head(
    head: ProjectionHead,
    clean: np.ndarray,
    distorted: np.ndarray,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
    temperature: float = 0.1,
    val_frac: float = 0.1,
    seed: int = 0,
    device: str | None = None,
) -> list[tuple[float, float]]:
    """Train the projection head so a window and its distorted copy get nearby fingerprints.

    Every other window in a batch acts as a negative. The last ``val_frac`` of the pairs are held out to monitor
    the loss.

    Args:
        head: Head to train in place.
        clean: Backbone embeddings of clean windows, shape ``[n, in_dim]``.
        distorted: Backbone embeddings of the distorted copies, same shape and order.
        epochs: Number of passes over the training pairs.
        batch_size: Pairs per step. Larger batches give more negatives.
        lr: AdamW learning rate.
        temperature: Contrastive loss temperature.
        val_frac: Fraction of pairs held out for validation.
        seed: Seed for shuffling.
        device: Training device. Defaults to CUDA when available.

    Returns:
        A list of ``(train_loss, val_loss)`` per epoch.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    head.to(device).train()
    clean_t = torch.from_numpy(clean.astype(np.float32)).to(device)
    distorted_t = torch.from_numpy(distorted.astype(np.float32)).to(device)
    n_val = max(int(len(clean) * val_frac), 2)
    n_train = len(clean) - n_val
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)

    history = []
    for _ in range(epochs):
        order = torch.randperm(n_train, device=device)
        total = 0.0
        for start in range(0, n_train - 1, batch_size):
            idx = order[start : start + batch_size]
            loss = info_nce(head(clean_t[idx]), head(distorted_t[idx]), temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        head.eval()
        with torch.no_grad():
            val = info_nce(head(clean_t[n_train:]), head(distorted_t[n_train:]), temperature).item()
        head.train()
        history.append((total / n_train, val))
    head.eval()
    return history
