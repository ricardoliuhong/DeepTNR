import torch
import torch.nn as nn

class Contrast(nn.Module):
    def __init__(self, hidden_dim, output_dim, tau=0.01, dropout=0.5):
        super(Contrast, self).__init__()
        # Projection head to map embeddings to a new space
        self.proj = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),  # First linear layer
            nn.ELU(),  # Exponential Linear Unit activation
            nn.Dropout(dropout),  # Dropout for regularization
            nn.Linear(hidden_dim, hidden_dim),  # Second linear layer
            nn.ELU(),  # Exponential Linear Unit activation
            nn.Dropout(dropout),  # Dropout for regularization
            nn.Linear(hidden_dim, output_dim)  # Final linear layer
        )
        # Adaptive average pooling to reduce dimensionality
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Temperature parameter for contrastive loss
        self.tau = tau
        # Initialize weights using Xavier initialization
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        """
        Compute the similarity matrix between two sets of embeddings.
        Args:
            z1 (torch.Tensor): First set of embeddings.
            z2 (torch.Tensor): Second set of embeddings.
        Returns:
            torch.Tensor: Similarity matrix.
        """
        # Normalize embeddings
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        # Compute dot product between embeddings
        dot_numerator = torch.mm(z1, z2.t())
        # Compute dot product of norms
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        # Compute similarity matrix using exponential and temperature (tau)
        sim_matrix = torch.exp(dot_numerator / (dot_denominator + 1e-8) / self.tau)
        return sim_matrix

    def forward(self, za, zb):
        """
        Forward pass for the contrastive model.
        Args:
            za (torch.Tensor): First set of embeddings.
            zb (torch.Tensor): Second set of embeddings.
        Returns:
            torch.Tensor: Projected and pooled embeddings for za and zb.
        """
        # Project embeddings to a new space
        za_proj = self.proj(za)
        zb_proj = self.proj(zb)
        # Apply adaptive average pooling
        za_proj = self.pool(za_proj.unsqueeze(2)).squeeze(2)
        zb_proj = self.pool(zb_proj.unsqueeze(2)).squeeze(2)
        # Ensure both sets of embeddings have the same size
        if za_proj.size(0) != zb_proj.size(0):
            min_size = min(za_proj.size(0), zb_proj.size(0))
            za_proj = za_proj[:min_size]
            zb_proj = zb_proj[:min_size]
        return za_proj, zb_proj

def info_nce_loss(model, za_proj, zb_proj, tau):
    """
    Compute the InfoNCE loss for contrastive learning.
    Args:
        model (Contrast): The contrastive model.
        za_proj (torch.Tensor): Projected embeddings for the first view.
        zb_proj (torch.Tensor): Projected embeddings for the second view.
        tau (float): Temperature parameter.
    Returns:
        torch.Tensor: InfoNCE loss.
    """
    # Compute similarity matrix
    sim_matrix = model.sim(za_proj, zb_proj) / tau
    # Create labels for positive pairs
    labels = torch.arange(za_proj.size(0)).to(za_proj.device)
    # Extract positive pairs (diagonal of the similarity matrix)
    positive_pairs = sim_matrix[labels, labels].view(-1, 1)
    # Extract negative pairs (off-diagonal elements)
    negative_pairs = sim_matrix[labels.unsqueeze(1) != labels.unsqueeze(0)].view(za_proj.size(0), -1)
    # Compute positive loss using log-softmax
    positive_loss = -torch.log(positive_pairs / (positive_pairs + negative_pairs.sum(1, keepdim=True)))
    # Compute mean loss
    loss = positive_loss.mean()
    return loss