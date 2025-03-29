import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import pandas as pd
import random
import os
import argparse
import optuna
import anndata
import warnings
warnings.filterwarnings("ignore")
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import k_hop_subgraph
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from Transfer_learning import DomainAdversarialClassifier,  adjust_alpha
from Contrastive_learning import Contrast, info_nce_loss

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, dropout=0.5, heads=4):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First GAT layer
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, concat=True))
        self.bns.append(nn.BatchNorm1d(hidden_dim * heads))

        # Intermediate GAT layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim * heads))

        # Final GAT layer
        self.convs.append(GATConv(hidden_dim * heads, output_dim, heads=1, concat=False))
        self.bns.append(nn.BatchNorm1d(output_dim))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = torch.relu(x)
            if i < self.num_layers - 1:
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First GCN layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Intermediate GCN layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Final GCN layer
        self.convs.append(GCNConv(hidden_dim, output_dim))
        self.bns.append(nn.BatchNorm1d(output_dim))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = torch.relu(x)
            if i < self.num_layers - 1:
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x

class MLPClassifier(nn.Module):
    def __init__(self, MLP_input_dim, MLP_hidden_dims, MLP_output_dim=1, MLP_dropout=0.6, MLP_activation=nn.GELU, MLP_num_heads=4):
        super(MLPClassifier, self).__init__()
        
        self.MLP_activation = MLP_activation
        self.MLP_dropout = nn.Dropout(MLP_dropout)
        
        # Initialize MLP layers
        self.MLP_mlp_layers = nn.ModuleList()
        prev_dim = MLP_input_dim
        for hidden_dim in MLP_hidden_dims:
            self.MLP_mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        # Initialize multi-head attention
        self.MLP_multihead_attn = nn.MultiheadAttention(embed_dim=MLP_input_dim, num_heads=MLP_num_heads, dropout=MLP_dropout)
        
        # Output layer
        self.MLP_output_layer = nn.Linear(prev_dim, MLP_output_dim)
    
    def forward(self, x):
        """
        Forward pass function.
        """
        # Check if input is 2D or 3D tensor
        if x.dim() == 2:  # [batch_size, embed_dim]
            x = x.unsqueeze(1)  # Add sequence length dimension
        elif x.dim() != 3:
            raise ValueError(f"Expected input tensor with 2 or 3 dimensions, got {x.dim()}")

        # Ensure input dimension matches the embedding dimension of multi-head attention
        assert x.shape[2] == self.MLP_multihead_attn.embed_dim, (
            f"Input dimension mismatch for multi-head attention: expected {self.MLP_multihead_attn.embed_dim}, got {x.shape[2]}"
        )
        
        # Transpose input to [seq_len, batch_size, embed_dim]
        x = x.permute(1, 0, 2)
        
        # Multi-head attention
        attn_output, _ = self.MLP_multihead_attn(x, x, x)
        x = attn_output  # Directly use attention output
        
        # Transpose back to [batch_size, seq_len, embed_dim]
        x = x.permute(1, 0, 2)
        
        # Forward through MLP layers
        for layer in self.MLP_mlp_layers:
            x = layer(x)
            x = self.MLP_activation()(x)
            x = self.MLP_dropout(x)
        
        # Output layer
        x = self.MLP_output_layer(x)
        return x


def domain_adversarial_loss(features, domain_classifier, domain_labels, alpha):
    domain_preds = domain_classifier(features, alpha)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(domain_preds, domain_labels)
    return loss

def clean_data(tensor):
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)  # Replace NaN with 0
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)  # Replace Inf with 0
    return tensor

def evaluate_model(classifier, embeddings, labels, threshold=0.5):
    classifier.eval()
    with torch.no_grad():
        embeddings = clean_data(embeddings)  # Clean embeddings of NaN and Inf
        outputs = classifier(embeddings).view(-1)
        predictions = torch.sigmoid(outputs).detach().cpu().numpy().flatten()
        labels = labels.cpu().numpy()

        # Check if predictions contain NaN
        if np.isnan(predictions).any():
            raise ValueError("Predictions contain NaN values")

        valid_indices = (labels != -1)
        valid_labels = labels[valid_indices]
        valid_predictions = predictions[valid_indices.squeeze()]

        binary_predictions = (valid_predictions >= threshold).astype(int)

        auroc = roc_auc_score(valid_labels, valid_predictions)
        precision = precision_score(valid_labels, binary_predictions, zero_division=1)
        recall = recall_score(valid_labels, binary_predictions, zero_division=1)
        f1 = f1_score(valid_labels, binary_predictions, zero_division=1)

    return auroc, precision, recall, f1

def train_classifier_on_domain(embeddings, labels, hidden_dims, input_dim, save_path='./model_checkpoints', domain='source'):
    output_dim = 1
    assert embeddings.shape[0] == labels.shape[0]

    classifier = MLPClassifier(MLP_input_dim=input_dim, MLP_hidden_dims=hidden_dims, MLP_output_dim=output_dim).to(embeddings.device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.01, weight_decay=1e-3)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    classifier.train()
    best_loss = float('inf')
    early_stop_patience = 100
    no_improvement_epochs = 0

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for epoch in range(500):
        optimizer.zero_grad()
        outputs = classifier(embeddings.detach()).view(-1)
        mask = (labels != -1).float()
        loss = criterion(outputs * mask, labels.float() * mask)

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"NaN or Inf detected at epoch {epoch}, stopping training.")
            break

        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if loss.item() < best_loss:
            best_loss = loss.item()
            no_improvement_epochs = 0
            torch.save({
                'epoch': epoch,
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, os.path.join(save_path, f'best_classifier_{domain}.pth'))
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= early_stop_patience:
            print(f"Early stopping at epoch {epoch} due to no improvement.")
            break

    return classifier, optimizer

def save_predictions_to_h5ad(predictions, adata_path, drug_name):
    adata = anndata.read_h5ad(adata_path)
    column_name = f"{drug_name}_Drug_Sensitivity_Predictions"
    adata.obs[column_name] = predictions
    adata.write(adata_path)
    print(f"Predictions saved to {adata_path} with column name: {column_name}")

def save_metrics(domain, auroc, precision, recall, f1, trial_number, is_best, save_folder='optuna_metrics'):
    metrics = {
        "Domain": [domain],
        "AUROC": [auroc],
        "Precision": [precision],
        "Recall": [recall],
        "F1 Score": [f1]
    }
    metrics_df = pd.DataFrame(metrics)
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    filename = os.path.join(save_folder, f'trial_{trial_number}_metrics.csv')
    
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        if is_best:
            metrics_df.to_csv(filename, index=False)
            print(f"New highest AUROC metrics saved to {filename}")
        else:
            print(f"Current AUROC ({auroc}) is not higher than existing records, not saved.")
    else:
        metrics_df.to_csv(filename, index=False)
        print(f"Metrics saved to {filename}")

def save_predictions(predictions, trial_number, save_folder='optuna_predictions'):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    filename = os.path.join(save_folder, f'trial_{trial_number}_predictions.csv')
    pd.DataFrame(predictions, columns=['Prediction']).to_csv(filename, index=False)
    print(f"Predictions have been saved to {filename}")

def adjust_tau(initial_tau, final_tau, epoch, total_epochs):
    return initial_tau + (final_tau - initial_tau) * (epoch / total_epochs)

def feature_masking(features, mask_ratio=0.1):
    mask = torch.rand(features.size(1)) < mask_ratio
    masked_features = features.clone()
    masked_features[:, mask] = 0
    return masked_features

def feature_transformation(features, scale=0.1):
    transform = torch.randn_like(features) * scale
    transformed_features = features + transform
    return transformed_features

def feature_noise(features, noise_scale=0.01):
    noise = torch.randn_like(features) * noise_scale
    noisy_features = features + noise
    return noisy_features

def random_subgraph_sampling(edge_index, num_nodes, sample_ratio=0.8):
    subset = torch.randperm(num_nodes)[:int(sample_ratio * num_nodes)]
    
    # Ensure consistent number of nodes
    subset = torch.arange(num_nodes)
    
    subgraph_edge_index, _, _, _ = k_hop_subgraph(subset, 1, edge_index, relabel_nodes=True)
    return subgraph_edge_index

def edge_perturbation(edge_index, num_nodes, perturb_ratio=0.1):
    num_edges = edge_index.size(1)
    num_perturb = int(perturb_ratio * num_edges)
    
    # Add random edges
    new_edges = torch.randint(0, num_nodes, (2, num_perturb))
    perturbed_edge_index = torch.cat([edge_index, new_edges], dim=1)
    
    # Remove random edges
    indices_to_remove = torch.randperm(perturbed_edge_index.size(1))[:num_perturb]
    perturbed_edge_index = torch.cat([perturbed_edge_index[:, :indices_to_remove[0]], perturbed_edge_index[:, indices_to_remove[-1]+1:]], dim=1)
    
    return perturbed_edge_index

def node_dropout(edge_index, num_nodes, dropout_ratio=0.1):
    drop_indices = torch.randperm(num_nodes)[:int(dropout_ratio * num_nodes)]
    mask = torch.ones(num_nodes, dtype=torch.bool)
    mask[drop_indices] = False
    remaining_nodes = torch.arange(num_nodes)[mask]
    
    # Ensure consistent number of nodes
    remaining_nodes = torch.arange(num_nodes)
    
    subgraph_edge_index, _, _, _ = k_hop_subgraph(remaining_nodes, 1, edge_index, relabel_nodes=True)
    return subgraph_edge_index

def create_views(features, edge_index, num_views=3):
    views = []
    num_nodes = features.size(0)
    
    for _ in range(num_views):
        # Feature augmentation
        augmented_features = feature_noise(feature_transformation(feature_masking(features)))
        
        # Structure augmentation
        augmented_edge_index = edge_perturbation(edge_index, num_nodes, perturb_ratio=0.1)
        
        # Ensure consistent number of nodes
        augmented_edge_index = augmented_edge_index[:, augmented_edge_index[0] < num_nodes]
        augmented_edge_index = augmented_edge_index[:, augmented_edge_index[1] < num_nodes]
        

        # Create view
        view = Data(x=augmented_features, edge_index=augmented_edge_index)
        views.append(view)
    return views

def objective(trial):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    source_features = torch.tensor(np.load(args.source_features), dtype=torch.float32).to(device)
    target_features = torch.tensor(np.load(args.target_features), dtype=torch.float32).to(device)
    source_edge_index = torch.tensor(np.load(args.source_edge_index), dtype=torch.long).to(device)
    target_edge_index = torch.tensor(np.load(args.target_edge_index), dtype=torch.long).to(device)
    source_labels = torch.tensor(np.load(args.source_labels), dtype=torch.float32).to(device).squeeze()
    #target_labels = torch.tensor(np.load(args.target_labels), dtype=torch.float32).to(device).squeeze()


    source_data = Data(x=source_features, edge_index=source_edge_index, y=source_labels)
    target_data = Data(x=target_features, edge_index=target_edge_index)

    # Create DataLoader and set num_workers to the number of CPU cores
    batch_size = 64  # Adjust batch size as needed

    source_loader = DataLoader([source_data], batch_size=batch_size, shuffle=True)
    target_loader = DataLoader([target_data], batch_size=batch_size, shuffle=True)

    
    input_dim = source_features.shape[1]
    hidden_dim = trial.suggest_categorical('hidden_dim', [512])
    output_dim = trial.suggest_categorical('output_dim', [256])
    num_layers = trial.suggest_categorical('num_layers', [3, 4, 5])
    dropout = trial.suggest_categorical('dropout', [0.6, 0.7, 0.8])
    initial_tau = trial.suggest_categorical('initial_tau', [0.1])
    final_tau = trial.suggest_categorical('final_tau', [0.001])
    subgraph = trial.suggest_categorical('subgraph', [4,5,6])

    hidden_dims = [trial.suggest_categorical(f'hidden_dim_{i}', [256, 512]) for i in range(trial.suggest_categorical('num_hidden_layers', [2, 3, 4, 5]))]

    domain_loss_weight = trial.suggest_categorical('domain_loss_weight', [ 1.0, 2.0, 3.0])
    heads = trial.suggest_categorical('heads', [4])

    # Choose model type (GCN or GAT)
    if args.model == 'GCN':
        model = GCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, dropout=dropout).to(device)
    else:
        model = GAT(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, dropout=dropout, heads=heads).to(device)

    # Initialize contrastive learning and domain classifier models
    contrast_model = Contrast(hidden_dim=hidden_dim, output_dim=output_dim, tau=initial_tau, dropout=dropout).to(device)
    domain_classifier = DomainAdversarialClassifier(input_dim=output_dim, hidden_dim=hidden_dim).to(device)

    # Optimizer and learning rate scheduler
    optimizer = optim.Adam(list(model.parameters()) + list(contrast_model.parameters()) + list(domain_classifier.parameters()), lr=trial.suggest_categorical('learning_rate', [1e-4, 1e-3]), weight_decay=trial.suggest_categorical('weight_decay', [1e-4, 1e-3]))
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0001)

    best_loss = float('inf')
    early_stop_patience = 50
    no_improvement_epochs = 0
    total_epochs = 500

    for epoch in range(total_epochs):
        optimizer.zero_grad()
        
        # Load data using DataLoader
        for batch in source_loader:
            source_features, source_edge_index, source_labels = batch.x, batch.edge_index, batch.y
            source_embeddings = model(source_features, source_edge_index)

        for batch in target_loader:
            target_features, target_edge_index = batch.x, batch.edge_index
            target_embeddings = model(target_features, target_edge_index)

        # Dynamically adjust tau
        tau = adjust_tau(initial_tau, final_tau, epoch, total_epochs)

        # Create multiple views
        source_views = create_views(source_features, source_edge_index, subgraph)
        target_views = create_views(target_features, target_edge_index, subgraph)

        # Compute contrastive loss
        contrastive_loss = 0
        for source_view, target_view in zip(source_views, target_views):
            source_view_embeddings = model(source_view.x, source_view.edge_index)
            target_view_embeddings = model(target_view.x, target_view.edge_index)
            source_proj, target_proj = contrast_model(source_view_embeddings, target_view_embeddings)
            contrastive_loss += info_nce_loss(contrast_model, source_proj, target_proj, tau=tau)
        contrastive_loss /= len(source_views)

        domain_labels = torch.cat([torch.zeros(source_embeddings.size(0)), torch.ones(target_embeddings.size(0))]).long().to(device)
        features = torch.cat([source_embeddings, target_embeddings], dim=0)
        domain_loss = domain_adversarial_loss(features, domain_classifier, domain_labels, adjust_alpha(epoch))

        total_loss = contrastive_loss * 0.2 + domain_loss_weight * domain_loss

        total_loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 40 == 0:
            print(f"Epoch {epoch}, Contrastive Loss: {contrastive_loss}, Domain Loss: {domain_loss}, Total Loss: {total_loss}, Tau: {tau}")


        if total_loss < best_loss:
            best_loss = total_loss
            no_improvement_epochs = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),  # General model saving, no longer using gcn_model
                'contrast_state_dict': contrast_model.state_dict(),
                'domain_classifier_state_dict': domain_classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            }, 'best_model.pth')
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= early_stop_patience:
            print("Early stopping due to no improvement.")
            break

    # Generate embeddings using the trained model
    source_embeddings = model(source_features, source_edge_index).detach()
    target_embeddings = model(target_features, target_edge_index).detach()

    # Train classifier
    classifier_source, _ = train_classifier_on_domain(source_embeddings, source_labels, hidden_dims, output_dim, save_path='./model_checkpoints', domain='source')

    classifier_source.eval()
    with torch.no_grad():
        target_predictions = torch.sigmoid(classifier_source(target_embeddings)).detach().cpu().numpy().flatten()

    # Evaluate target domain using source domain classifier
    auroc_source, precision_source, recall_source, f1_source = evaluate_model(classifier_source, source_embeddings, source_labels)

    is_source_best = auroc_source > 0.5  # Modify this logic to determine the best target domain
    save_metrics("Source", auroc_source, precision_source, recall_source, f1_source, trial.number, True)
    save_predictions_to_h5ad(target_predictions, args.Spatial_dataset, args.Drug)
    print(f"Target domain predictions for {args.Drug} have been saved to {args.Spatial_dataset}")
    return best_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_features', type=str, default='Bulk_features.npy', help='Path to Bulk node features')
    parser.add_argument('--target_features', type=str, default='scRNA_features.npy', help='Path to scRNA node features')
    parser.add_argument('--source_edge_index', type=str, default='Bulk_edge_index.npy')
    parser.add_argument('--target_edge_index', type=str, default='scRNA_edge_index.npy')
    parser.add_argument('--Spatial_dataset', type=str, default='CRC1.h5ad', help='Path to the scRNA-seq Data file')
    parser.add_argument('--source_labels', type=str, default='Bulk_labels.npy')
    parser.add_argument('--Drug', type=str, default='DOCETAXEL', help='Name of the drug to process')
    parser.add_argument('--model', type=str, default='GAT', choices=['GCN', 'GAT'], help='Choose between GCN or GAT model')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    set_seed(411)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=1)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")