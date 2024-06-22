import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Contrast(nn.Module):
    def __init__(self, hidden_dim, output_dim, tau, dropout=0.5):
        super(Contrast, self).__init__()

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim))
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
        
        # 池化操作
        za_proj = self.pool(za_proj.unsqueeze(2)).squeeze(2)
        zb_proj = self.pool(zb_proj.unsqueeze(2)).squeeze(2)
        
        # 确保 za_proj 和 zb_proj 在第一个维度上的大小相同
        if za_proj.size(0) != zb_proj.size(0):
            min_size = min(za_proj.size(0), zb_proj.size(0))
            za_proj = za_proj[:min_size]
            zb_proj = zb_proj[:min_size]

        return za_proj, zb_proj

def info_nce_loss(model, za_proj, zb_proj, tau):
    sim_matrix = model.sim(za_proj, zb_proj)
    labels = torch.arange(za_proj.size(0)).to(za_proj.device)
    loss = F.cross_entropy(sim_matrix / tau, labels)
    return loss

def data_augmentation(embeddings, noise_factor=0.1):
    noise = torch.randn_like(embeddings) * noise_factor
    return embeddings + noise

def normalize_embeddings(embeddings):
    return (embeddings - embeddings.mean(dim=0)) / (embeddings.std(dim=0) + 1e-8)

def train_contrastive_model(model, bulk_node_embeddings, scrna_node_embeddings, optimizer, scheduler, tau, epochs=100):
    model.train()
    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 数据增强
        bulk_node_embeddings_aug = data_augmentation(bulk_node_embeddings)
        scrna_node_embeddings_aug = data_augmentation(scrna_node_embeddings)
        
        # 归一化
        bulk_node_embeddings_aug = normalize_embeddings(bulk_node_embeddings_aug)
        scrna_node_embeddings_aug = normalize_embeddings(scrna_node_embeddings_aug)
        
        # 前向传播
        za_proj, zb_proj = model(bulk_node_embeddings_aug, scrna_node_embeddings_aug)
        
        # 计算InfoNCE损失
        loss = info_nce_loss(model, za_proj, zb_proj, tau)
        
        # 检查损失是否为 NaN 或 Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Encountered NaN or Inf in loss at epoch {epoch+1}")
            break
        
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)  # 调整max_norm的值
        
        optimizer.step()
        scheduler.step(loss)
        
        loss_history.append(loss.item())
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    
    return loss_history

def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    # 加载嵌入表示
    bulk_node_embeddings = np.load(args.bulk_node_embeddings)
    scrna_node_embeddings = np.load(args.scrna_node_embeddings)

    # 检查数据是否包含 NaN 或 Inf
    if np.isnan(bulk_node_embeddings).any() or np.isnan(scrna_node_embeddings).any():
        raise ValueError("Input embeddings contain NaN values")
    if np.isinf(bulk_node_embeddings).any() or np.isinf(scrna_node_embeddings).any():
        raise ValueError("Input embeddings contain Inf values")

    # 转换为Tensor
    bulk_node_embeddings = torch.tensor(bulk_node_embeddings, dtype=torch.float32).to(device)
    scrna_node_embeddings = torch.tensor(scrna_node_embeddings, dtype=torch.float32).to(device)

    # 假设所有嵌入表示的维度相同
    hidden_dim = args.hidden_dim  # 嵌入表示的维度

    # 定义对比学习模型
    model = Contrast(hidden_dim, args.output_dim, args.tau, dropout=0.5).to(device)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

    # 进行跨视图对比学习训练
    loss_history = train_contrastive_model(model, bulk_node_embeddings, scrna_node_embeddings, optimizer, scheduler, args.tau, args.epochs)

    # 保存对齐后的嵌入表示
    model.eval()
    with torch.no_grad():
        za_proj, zb_proj = model(bulk_node_embeddings, scrna_node_embeddings)
        aligned_bulk_node_embeddings = za_proj.cpu().numpy()
        aligned_scrna_node_embeddings = zb_proj.cpu().numpy()

        np.save('Aligned_Bulk_node_embeddings.npy', aligned_bulk_node_embeddings)
        np.save('Aligned_scRNA_node_embeddings.npy', aligned_scrna_node_embeddings)

        print('Aligned embeddings saved.')

    # 输出训练损失
    print("Training Loss History:")
    for epoch, loss in enumerate(loss_history, 1):
        print(f"Epoch {epoch}: Loss = {loss}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cross-View Contrastive Learning')
    parser.add_argument('--bulk_node_embeddings', type=str, default='Bulk_node_embeddings.npy', help='Path to Bulk node embeddings')
    parser.add_argument('--scrna_node_embeddings', type=str, default='scRNA_node_embeddings.npy', help='Path to scRNA node embeddings')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension of embeddings')
    parser.add_argument('--output_dim', type=int, default=256, help='Output dimension after projection')
    parser.add_argument('--tau', type=float, default=0.5, help='Temperature parameter for contrastive loss')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for optimizer')  # 进一步减小学习率
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training')
    args = parser.parse_args()

    main(args)
