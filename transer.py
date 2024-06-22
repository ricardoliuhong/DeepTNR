import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, dropout=0.5):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = torch.relu(self.convs[i](x, edge_index))
            if i < self.num_layers - 1:
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x

class Contrast(nn.Module):
    def __init__(self, hidden_dim, output_dim, tau, dropout=0.5):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
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
        sim_matrix = torch.exp(dot_numerator / (dot_denominator + 1e-8) / self.tau)
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
    sim_matrix = model.sim(za_proj, zb_proj)
    labels = torch.arange(za_proj.size(0)).to(za_proj.device)
    positive_pairs = sim_matrix[labels, labels].view(-1, 1)
    negative_pairs = sim_matrix[labels.unsqueeze(1) != labels.unsqueeze(0)].view(za_proj.size(0), -1)
    positive_loss = -torch.log(positive_pairs / (positive_pairs + negative_pairs.sum(1, keepdim=True)))
    loss = positive_loss.mean()
    return loss

def data_augmentation(embeddings, method='noise', noise_factor=0.1, drop_rate=0.1):
    if method == 'noise':
        noise = torch.randn_like(embeddings) * noise_factor
        return embeddings + noise
    elif method == 'drop':
        mask = torch.rand_like(embeddings) > drop_rate
        return embeddings * mask.float()
    elif method == 'mix':
        return embeddings + torch.roll(embeddings, shifts=1, dims=0)
    else:
        raise ValueError("Unknown data augmentation method")

def normalize_embeddings(embeddings):
    return (embeddings - embeddings.mean(dim=0)) / (embeddings.std(dim=0) + 1e-8)

def train_contrastive_model(model, source_node_embeddings, target_node_embeddings, source_edge_index, target_edge_index, optimizer, scheduler, tau, epochs=100):
    model.train()
    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        source_node_embeddings_aug = data_augmentation(source_node_embeddings, method='noise')
        target_node_embeddings_aug = data_augmentation(target_node_embeddings, method='drop')
        source_node_embeddings_aug = normalize_embeddings(source_node_embeddings_aug)
        target_node_embeddings_aug = normalize_embeddings(target_node_embeddings_aug)
        source_node_embeddings_aug = model.gcn(source_node_embeddings_aug, source_edge_index)
        target_node_embeddings_aug = model.gcn(target_node_embeddings_aug, target_edge_index)
        za_proj, zb_proj = model(source_node_embeddings_aug, target_node_embeddings_aug)
        loss = info_nce_loss(model, za_proj, zb_proj, tau)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Loss is NaN or Inf at epoch {epoch}")
            continue
        loss.backward()
        optimizer.step()
        scheduler.step(epoch)
        loss_history.append(loss.item())
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    return model, loss_history

def predict_on_target_domain(model, target_node_embeddings, target_edge_index):
    model.eval()
    with torch.no_grad():
        target_node_embeddings = model.gcn(target_node_embeddings, target_edge_index)
        target_node_embeddings = normalize_embeddings(target_node_embeddings)
    return target_node_embeddings

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    source_node_embeddings = torch.tensor(np.load(args.source_node_embeddings), dtype=torch.float32).to(device)
    target_node_embeddings = torch.tensor(np.load(args.target_node_embeddings), dtype=torch.float32).to(device)
    source_edge_index = torch.tensor(np.load(args.source_edge_index), dtype=torch.long).to(device)
    target_edge_index = torch.tensor(np.load(args.target_edge_index), dtype=torch.long).to(device)

    model = Contrast(hidden_dim=args.hidden_dim, output_dim=args.output_dim, tau=args.tau).to(device)
    model.gcn = GCN(input_dim=source_node_embeddings.shape[1], hidden_dim=args.hidden_dim, output_dim=args.output_dim, num_layers=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    model, loss_history = train_contrastive_model(model, source_node_embeddings, target_node_embeddings, source_edge_index, target_edge_index, optimizer, scheduler, tau=args.tau, epochs=args.epochs)

    target_node_embeddings = predict_on_target_domain(model, target_node_embeddings, target_edge_index)
    #np.save(args.output_path, target_node_embeddings.cpu().numpy())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cross-View Contrastive Learning')
    parser.add_argument('--source_node_embeddings', type=str, default='Bulk_node_embeddings.npy', help='Path to Bulk node embeddings')
    parser.add_argument('--target_node_embeddings', type=str, default='scRNA_node_embeddings.npy', help='Path to scRNA node embeddings')
    parser.add_argument('--source_edge_index', type=str, default='Bulk_edge_index.npy', help='Path to Bulk edge index')
    parser.add_argument('--target_edge_index', type=str, default='scRNA_edge_index.npy', help='Path to scRNA edge index')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--output_dim', type=int, default=128, help='Output dimension size')
    parser.add_argument('--tau', type=float, default=0.1, help='Temperature parameter for contrastive loss')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train')

    args = parser.parse_args()
    main(args)