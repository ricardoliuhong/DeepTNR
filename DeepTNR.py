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
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
warnings.filterwarnings("ignore")
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import k_hop_subgraph
from DomainAdversarialClassifier import DomainAdversarialClassifier, GradientReversalLayer, adjust_alpha
from Contrast_learning import Contrast, info_nce_loss

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

        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, concat=True))
        self.bns.append(nn.BatchNorm1d(hidden_dim * heads))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim * heads))

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

        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

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


def domain_adversarial_loss(features, domain_classifier, domain_labels, alpha):
    domain_preds = domain_classifier(features, alpha)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(domain_preds, domain_labels)
    return loss

def clean_data(tensor):
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor

def evaluate_model(classifier, embeddings, labels, threshold=0.5):
    classifier.eval()
    with torch.no_grad():
        embeddings = clean_data(embeddings)
        outputs = classifier(embeddings).view(-1)
        predictions = torch.sigmoid(outputs).detach().cpu().numpy().flatten()
        labels = labels.cpu().numpy()
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


def save_predictions_to_h5ad(predictions, adata_path, drug_name, trial_number=None):
    adata = anndata.read_h5ad(adata_path)
    prefix = f"trial_{trial_number}_" if trial_number is not None else ""
    column_name = f"{prefix}{drug_name}_Drug_Sensitivity_Predictions"
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
            print(f"Current AUROC ({auroc}) is not higher than existing record, not saved.")
    else:
        metrics_df.to_csv(filename, index=False)
        print(f"Metrics saved to {filename}")

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

def edge_perturbation(edge_index, num_nodes, perturb_ratio=0.1):
    num_edges = edge_index.size(1)
    num_perturb = int(perturb_ratio * num_edges)
    new_edges = torch.randint(0, num_nodes, (2, num_perturb))
    perturbed_edge_index = torch.cat([edge_index, new_edges], dim=1)
    indices_to_remove = torch.randperm(perturbed_edge_index.size(1))[:num_perturb]
    perturbed_edge_index = torch.cat([perturbed_edge_index[:, :indices_to_remove[0]], perturbed_edge_index[:, indices_to_remove[-1]+1:]], dim=1)
    return perturbed_edge_index

def random_subgraph_sampling(edge_index, num_nodes, sample_ratio=0.8):
    subset = torch.randperm(num_nodes)[:int(sample_ratio * num_nodes)]
    subset = torch.arange(num_nodes)
    subgraph_edge_index, _, _, _ = k_hop_subgraph(subset, 1, edge_index, relabel_nodes=True)
    return subgraph_edge_index

def node_dropout(edge_index, num_nodes, dropout_ratio=0.1):
    drop_indices = torch.randperm(num_nodes)[:int(dropout_ratio * num_nodes)]
    mask = torch.ones(num_nodes, dtype=torch.bool)
    mask[drop_indices] = False
    remaining_nodes = torch.arange(num_nodes)[mask]
    remaining_nodes = torch.arange(num_nodes)
    subgraph_edge_index, _, _, _ = k_hop_subgraph(remaining_nodes, 1, edge_index, relabel_nodes=True)
    return subgraph_edge_index

def create_views(features, edge_index, num_views=3):
    views = []
    num_nodes = features.size(0)
    for _ in range(num_views):
        augmented_features = feature_noise(feature_transformation(feature_masking(features)))
        augmented_edge_index = edge_perturbation(edge_index, num_nodes, perturb_ratio=0.1)
        augmented_edge_index = augmented_edge_index[:, augmented_edge_index[0] < num_nodes]
        augmented_edge_index = augmented_edge_index[:, augmented_edge_index[1] < num_nodes]
        view = Data(x=augmented_features, edge_index=augmented_edge_index)
        views.append(view)
    return views

def objective(trial):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    source_features = torch.tensor(np.load(args.source_features), dtype=torch.float32).to(device)
    target_features = torch.tensor(np.load(args.target_features), dtype=torch.float32).to(device)
    source_edge_index = torch.tensor(np.load(args.source_edge_index), dtype=torch.long).to(device)
    target_edge_index = torch.tensor(np.load(args.target_edge_index), dtype=torch.long).to(device)
    source_labels = torch.tensor(np.load(args.source_labels), dtype=torch.float32).to(device).squeeze()

    source_data = Data(x=source_features, edge_index=source_edge_index, y=source_labels)
    target_data = Data(x=target_features, edge_index=target_edge_index)

    batch_size = 64
    source_loader = DataLoader([source_data], batch_size=batch_size, shuffle=True)
    target_loader = DataLoader([target_data], batch_size=batch_size, shuffle=True)

    input_dim = source_features.shape[1]
    hidden_dim = trial.suggest_categorical('hidden_dim', [1024])
    output_dim = trial.suggest_categorical('output_dim', [512])
    num_layers = trial.suggest_categorical('num_layers', [3, 4,])
    dropout = trial.suggest_categorical('dropout', [0.5,0.6,0.7, 0.8, 0.9])
    initial_tau = trial.suggest_categorical('initial_tau', [0.1])
    final_tau = trial.suggest_categorical('final_tau', [0.01])
    subgraph = trial.suggest_categorical('subgraph', [4])
    hidden_dims = [trial.suggest_categorical(f'hidden_dim_{i}', [256, 512]) for i in range(trial.suggest_categorical('num_hidden_layers', [2, 3, 4]))]
    domain_loss_weight = trial.suggest_categorical('domain_loss_weight', [1.0, 2.0])
    heads = trial.suggest_categorical('heads', [4])

    if args.model == 'GCN':
        model = GCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, dropout=dropout).to(device)
    else:
        model = GAT(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, dropout=dropout, heads=heads).to(device)

    contrast_model = Contrast(hidden_dim=hidden_dim, output_dim=output_dim, tau=initial_tau, dropout=dropout).to(device)
    domain_classifier = DomainAdversarialClassifier(input_dim=output_dim, hidden_dim=hidden_dim).to(device)

    optimizer = optim.Adam(list(model.parameters()) + list(contrast_model.parameters()) + list(domain_classifier.parameters()),
                           lr=trial.suggest_categorical('learning_rate', [1e-3, 1e-2]),
                           weight_decay=trial.suggest_categorical('weight_decay', [1e-4, 1e-3]))
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.0001)

    best_loss = float('inf')
    early_stop_patience = 40
    no_improvement_epochs = 0
    total_epochs = 100

    for epoch in range(total_epochs):
        optimizer.zero_grad()

        for batch in source_loader:
            source_features_batch, source_edge_index_batch, source_labels_batch = batch.x, batch.edge_index, batch.y
            source_embeddings = model(source_features_batch, source_edge_index_batch)

        for batch in target_loader:
            target_features_batch, target_edge_index_batch = batch.x, batch.edge_index
            target_embeddings = model(target_features_batch, target_edge_index_batch)

        tau = adjust_tau(initial_tau, final_tau, epoch, total_epochs)

        source_views = create_views(source_features, source_edge_index, subgraph)
        target_views = create_views(target_features, target_edge_index, subgraph)

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

        total_loss = contrastive_loss * 0.05 + domain_loss_weight * domain_loss

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 40 == 0:
            print(f"Epoch {epoch}, Contrastive Loss: {contrastive_loss:.4f}, Domain Loss: {domain_loss:.4f}, Total Loss: {total_loss:.4f}, Tau: {tau:.4f}")

        if total_loss < best_loss:
            best_loss = total_loss
            no_improvement_epochs = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
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

    source_embeddings = model(source_features, source_edge_index).detach()
    target_embeddings = model(target_features, target_edge_index).detach()

    X_train = source_embeddings.cpu().numpy()
    y_train = source_labels.cpu().numpy()
    X_test = target_embeddings.cpu().numpy()

    tabpfn_clf = TabPFNClassifier(device='cpu', ignore_pretraining_limits=True)
    tabpfn_clf.fit(X_train, y_train)

    target_pred_probs = tabpfn_clf.predict_proba(X_test)[:, 1]
    source_pred_probs = tabpfn_clf.predict_proba(X_train)[:, 1]
    source_preds = tabpfn_clf.predict(X_train)

    auroc_source = roc_auc_score(y_train, source_pred_probs)
    precision_source = precision_score(y_train, source_preds, zero_division=1)
    recall_source = recall_score(y_train, source_preds, zero_division=1)
    f1_source = f1_score(y_train, source_preds, zero_division=1)
    save_predictions_to_h5ad(target_pred_probs, args.scRNA_dataset, args.Drug, trial.number)
    print(f"Target domain predictions for {args.Drug} have been saved to {args.scRNA_dataset}")

    is_source_best = auroc_source > 0.5
    save_metrics("Source", auroc_source, precision_source, recall_source, f1_source, trial.number, is_source_best)

    return best_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_features', type=str, default='Bulk_features.npy', help='Path to Bulk node features')
    parser.add_argument('--target_features', type=str, default='scRNA_features.npy', help='Path to scRNA node features')
    parser.add_argument('--source_edge_index', type=str, default='Bulk_edge_index.npy')
    parser.add_argument('--target_edge_index', type=str, default='scRNA_edge_index.npy')
    parser.add_argument('--scRNA_dataset', type=str, default='CRC1.h5ad', help='Path to the scRNA-seq Data file')
    parser.add_argument('--source_labels', type=str, default='Bulk_labels.npy')
    parser.add_argument('--Drug', type=str, default='DOCETAXEL', help='Name of the drug to process')
    parser.add_argument('--model', type=str, default='GAT', choices=['GCN', 'GAT'], help='Choose between GCN or GAT model')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    set_seed(21)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Parameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
