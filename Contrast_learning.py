import torch
import torch.nn as nn

class Contrast(nn.Module):
    def __init__(self, hidden_dim, output_dim, tau=0.01, dropout=0.5):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.tau = tau
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / (dot_denominator + 1e-8) / self.tau)  # 使用 tau 参数
        return sim_matrix

    def forward(self, za, zb):
        za_proj = self.proj(za)
        zb_proj = self.proj(zb)
        za_proj = self.pool(za_proj.unsqueeze(2)).squeeze(2)
        zb_proj = self.pool(zb_proj.unsqueeze(2)).squeeze(2)
        if za_proj.size(0) != zb_proj.size(0):
            min_size = min(za_proj.size(0), zb_proj.size(0))
            za_proj = za_proj[:min_size]
            zb_proj = zb_proj[:min_size]
        return za_proj, zb_proj

def info_nce_loss(model, za_proj, zb_proj, tau):
    sim_matrix = model.sim(za_proj, zb_proj) / tau  # 使用 tau 参数
    labels = torch.arange(za_proj.size(0)).to(za_proj.device)
    positive_pairs = sim_matrix[labels, labels].view(-1, 1)
    negative_pairs = sim_matrix[labels.unsqueeze(1) != labels.unsqueeze(0)].view(za_proj.size(0), -1)
    positive_loss = -torch.log(positive_pairs / (positive_pairs + negative_pairs.sum(1, keepdim=True)))
    loss = positive_loss.mean()
    return loss
