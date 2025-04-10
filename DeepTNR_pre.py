import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from imblearn.over_sampling import RandomOverSampler
import dgl
import argparse
import scipy.sparse as sp
from scipy.sparse import issparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MaxAbsScaler
import scipy
import networkx as nx
import scipy.sparse as sp
import random
from sklearn.preprocessing import OneHotEncoder
from joblib import dump

torch.set_num_threads(os.cpu_count() * 2)
print(f"PyTorch is using {torch.get_num_threads()} threads.")

parser = argparse.ArgumentParser(description='spatialdrug')
parser.add_argument('--expid', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--train_dataset', type=str, default='Bulk_train_adata.h5ad')
parser.add_argument('--train_expression_file', type=str, default='Bulk_features.csv')
parser.add_argument('--train_binary_labels_file', type=str, default='Bulk_train_labels.csv')
parser.add_argument('--Spatial_dataset', type=str, default='CRC2.h5ad', help='Path to spatial dataset')
parser.add_argument('--expression_file', type=str, default='ALL_expression.csv', help='Path to expression file')
parser.add_argument('--binary_labels_file', type=str, default='ALL_label_binary_wf.csv', help='Path to binary labels file')
parser.add_argument('--output_prefix', type=str, default='CRC2', help='Output prefix for saved files')
parser.add_argument('--Drug', type=str, default='DOCETAXEL', help='Name of the drug to process')
parser.add_argument('--visium_path', type=str, default='VISDS000772', help='Path to Visium data')
parser.add_argument('--count_file', type=str, default='filtered_feature_bc_matrix.h5', help='Path to Visium count file')
parser.add_argument('--output_file', type=str, default='CRC2_annotated.h5ad', help='Output file for annotated Visium data')
args = parser.parse_args()

output_dir = 'DeepTNR_Data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def save_file(file_name, data):
    file_path = os.path.join(output_dir, file_name)
    if isinstance(data, pd.DataFrame):
        data.to_csv(file_path)
    elif isinstance(data, sc.AnnData):
        data.write(file_path)
    else:
        np.save(file_path, data)

def load_and_align_data(Spatial_dataset, expression_file, binary_labels_file):
    adata = sc.read_h5ad(Spatial_dataset)

    Bulk_expression = pd.read_csv(expression_file, index_col=[0])
    binary_labels_data = pd.read_csv(binary_labels_file)

    common_genes = list(set(adata.var_names).intersection(Bulk_expression.columns))
    adata = adata[:, common_genes]
    Bulk_expression = Bulk_expression[common_genes]
    adata.write(os.path.join(output_dir, args.Spatial_dataset))
    genes = adata.var_names.tolist()
    expr_matrix = adata.X
    scaler = MaxAbsScaler()
    Bulk_features = Bulk_expression[genes]  # Ensure Bulk_features has same number of columns as genes
    Bulk_features = scaler.fit_transform(Bulk_features)
    Bulk_features = np.log1p(Bulk_features)
    Bulk_features = pd.DataFrame(Bulk_features, index=Bulk_expression.index, columns=genes)
    Bulk_features['DepMap_ID'] = Bulk_expression.index  # Assuming Bulk_expression index is DepMap_ID

    # Only keep DepMap_IDs that exist in binary_labels_file
    valid_depmap_ids = binary_labels_data['DepMap_ID'].unique()
    Bulk_features = Bulk_features[Bulk_features['DepMap_ID'].isin(valid_depmap_ids)]

    # Generate label matrix based on Bulk_features order
    drug_names = binary_labels_data.columns[1:]  # Get drug names

    # Convert "resistant" and "sensitive" to 0 and 1
    binary_labels_data[drug_names] = binary_labels_data[drug_names].replace({'resistant': 0, 'sensitive': 1})

    labels_matrix = binary_labels_data.set_index('DepMap_ID')[drug_names]  # Directly use drug labels from binary_labels_data

    # Ensure correct label matrix data type
    labels_matrix = labels_matrix.apply(pd.to_numeric, errors='coerce').fillna(2)

    # Remove DepMap_ID column from Bulk_features
    Bulk_features = Bulk_features.drop(columns=['DepMap_ID'])

    # Ensure Bulk_features and labels_matrix are in same order
    Bulk_features = Bulk_features.loc[labels_matrix.index]
    labels_tensor = torch.tensor(labels_matrix.values, dtype=torch.float32)

    # Store index information
    Bulk_index = list(Bulk_features.index)
    gene_names = genes

    return Bulk_features, Bulk_index, gene_names, labels_tensor, labels_matrix, drug_names

def split_data(Bulk_features, labels_tensor, split_ratio=1):
    # Get the number of samples
    num_samples = Bulk_features.shape[0]

    # Create an array of indices
    indices = np.arange(num_samples)

    # Shuffle the indices
    np.random.shuffle(indices)

    # Calculate the split point
    split_point = int(num_samples * split_ratio)

    # Split the indices into training and validation sets
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]

    # Subset the Bulk_features and labels_tensor
    train_features = Bulk_features.iloc[train_indices]
    train_labels = labels_tensor[train_indices]

    val_features = Bulk_features.iloc[val_indices]
    val_labels = labels_tensor[val_indices]

    # Add DepMap_ID column to the features DataFrames
    train_features['DepMap_ID'] = train_features.index
    val_features['DepMap_ID'] = val_features.index

    # Move DepMap_ID column to the first position
    train_features = train_features[['DepMap_ID'] + train_features.columns[:-1].tolist()]
    val_features = val_features[['DepMap_ID'] + val_features.columns[:-1].tolist()]

    return train_features, train_labels, val_features, val_labels

def upsample_data(features, labels):
    # Convert labels to numpy array
    labels_np = labels.numpy().ravel()

    # Initialize RandomOverSampler
    ros = RandomOverSampler(random_state=42)

    # Upsample the data
    features_resampled, labels_resampled = ros.fit_resample(features, labels_np)

    # Convert back to DataFrame and Tensor
    features_resampled_df = pd.DataFrame(features_resampled, columns=features.columns)
    labels_resampled_tensor = torch.tensor(labels_resampled, dtype=torch.float32).unsqueeze(1)

    return features_resampled_df, labels_resampled_tensor

def create_adata(features, labels, drug_names):
    # Create AnnData object
    adata = sc.AnnData(features.values)

    # Set obs information
    adata.obs = features.index.to_frame(name='cell_id')

    # Set var information
    adata.var = pd.DataFrame(index=features.columns)

    # Add label information to obs
    labels_df = pd.DataFrame(labels.numpy(), index=features.index, columns=drug_names)
    adata.obs = adata.obs.join(labels_df)

    # Add sensitive and sensitivity columns
    for drug_name in drug_names:
        adata.obs[f'sensitive_{drug_name}'] = adata.obs[drug_name].apply(lambda x: 1 if x == 1 else 0)
        adata.obs[f'sensitivity_{drug_name}'] = adata.obs[drug_name].apply(lambda x: 'Sensitive' if x == 1 else 'Resistant')

    return adata

def process_and_save_data(Spatial_dataset, expression_file, binary_labels_file, output_prefix, drug_name, split_ratio=0.9):
    # Load and align data
    Bulk_features, Bulk_index, gene_names, labels_tensor, labels_matrix, drug_names = load_and_align_data(Spatial_dataset, expression_file, binary_labels_file)

    # Filter data for the specified drug
    if drug_name not in drug_names:
        raise ValueError(f"Drug name '{drug_name}' not found in the available drug names: {drug_names}")

    drug_labels = labels_matrix[drug_name]
    Bulk_features = Bulk_features.loc[drug_labels.index]
    labels_tensor = torch.tensor(drug_labels.values, dtype=torch.float32).unsqueeze(1)  # Convert to 2D tensor

    # Upsample the data
    Bulk_features_upsampled, labels_tensor_upsampled = upsample_data(Bulk_features, labels_tensor)
    train_features, train_labels, val_features, val_labels = split_data(Bulk_features_upsampled, labels_tensor_upsampled, split_ratio)
    
    # Save train labels
    train_labels_df = pd.DataFrame(train_labels.numpy(), index=train_features['DepMap_ID'], columns=[drug_name])
    save_file('Bulk_train_labels.csv', train_labels_df)
    
    # Save train adata
    train_adata = create_adata(train_features.drop(columns=['DepMap_ID']), train_labels, [drug_name])
    save_file('Bulk_train_adata.h5ad', train_adata)

def preprocess_visium_data(visium_path, count_file, output_file, n_top_genes=4000):
    adata = sc.read_visium(visium_path, count_file=count_file, load_images=True)
    adata.var_names_make_unique(join='-')
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var.highly_variable]
    save_file(output_file, adata)
    return adata

def Bulk_load_and_align_data(train_dataset, train_binary_labels_file, Drug):
    try:
        # Construct full file paths
        train_dataset_path = os.path.join(output_dir, train_dataset)
        train_binary_labels_path = os.path.join(output_dir, train_binary_labels_file)

        # Check if files exist
        if not os.path.exists(train_dataset_path):
            raise FileNotFoundError(f"File {train_dataset} not found in {output_dir}.")
        if not os.path.exists(train_binary_labels_path):
            raise FileNotFoundError(f"File {train_binary_labels_file} not found in {output_dir}.")

        # Load scRNA-seq data
        adata = sc.read_h5ad(train_dataset_path)
        print(f"Loaded scRNA-seq data with shape: {adata.shape}")

        # Load binary labels data
        binary_labels_data = pd.read_csv(train_binary_labels_path)
        print(f"Loaded binary labels data with shape: {binary_labels_data.shape}")

        # Extract features and labels
        features = adata.X
        labels_matrix = binary_labels_data.set_index('DepMap_ID')
        labels = torch.tensor(labels_matrix.values, dtype=torch.float32)

        # If features are sparse matrix, convert to dense
        if issparse(features):
            features = features.toarray()

        # Data preprocessing
        sc.tl.pca(adata, svd_solver='arpack')
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=10)
        sc.tl.umap(adata)
        sc.tl.leiden(adata, key_added="clusters")
        adj = adata.obsp['connectivities']

        # Convert to PyTorch tensors
        features = torch.FloatTensor(features)
        adj = normalize_adj(adj)
        adj_coo = adj.tocoo()
        adj = (adj + sp.eye(adj.shape[0])).todense()

        # Build DGL graph
        dgl_graph = dgl.graph((adj_coo.row, adj_coo.col), num_nodes=adata.shape[0])
        dgl_graph.ndata['expr'] = torch.tensor(features.toarray(), dtype=torch.float32) if issparse(features) else torch.tensor(features.clone().detach(), dtype=torch.float32)
        scrna_edge_index = np.vstack((adj_coo.row, adj_coo.col))

        # Save data
        np.save(f'DeepTNR_Data/{Drug}_Bulk_edge_index.npy', scrna_edge_index)
        np.save(f'DeepTNR_Data/{Drug}_Bulk_features.npy', features)
        np.save(f'DeepTNR_Data/{Drug}_Bulk_labels.npy', labels)

        return adata, features, adj

    except Exception as e:
        raise ValueError(f"Error in Bulk_load_and_align_data: {e}")


def Sc_load_and_align_data(val_dataset):
    try:
        # Construct full file path
        val_dataset_path = os.path.join(output_dir, val_dataset)

        # Check if file exists
        if not os.path.exists(val_dataset_path):
            raise FileNotFoundError(f"File {val_dataset} not found in {output_dir}.")

        # Load scRNA-seq data
        adata = sc.read_h5ad(val_dataset_path)
        print(f"Loaded scRNA-seq data with shape: {adata.shape}")

        # Extract features
        features = adata.X

        # If features are sparse matrix, convert to dense
        if issparse(features):
            features = features.toarray()

        # Data preprocessing
        sc.tl.pca(adata, svd_solver='arpack')
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)
        sc.tl.umap(adata)
        sc.tl.leiden(adata, key_added="clusters")
        adj = adata.obsp['connectivities']
        labels = adata.obs['clusters'].astype(str)
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        labels = torch.tensor(labels, dtype=torch.float32)

        # Convert to PyTorch tensors
        features = torch.FloatTensor(features)
        adj = normalize_adj(adj)
        adj_coo = adj.tocoo()
        adj = (adj + sp.eye(adj.shape[0])).todense()

        # Build DGL graph
        dgl_graph = dgl.graph((adj_coo.row, adj_coo.col), num_nodes=adata.shape[0])
        dgl_graph.ndata['expr'] = torch.tensor(features.toarray(), dtype=torch.float32) if issparse(features) else torch.tensor(features.clone().detach(), dtype=torch.float32)
        scrna_edge_index = np.vstack((adj_coo.row, adj_coo.col))

        # Save data
        np.save(f'DeepTNR_Data/scRNA_edge_index.npy', scrna_edge_index)
        np.save(f'DeepTNR_Data/scRNA_features.npy', features)

        return adata, features, adj

    except Exception as e:
        raise ValueError(f"Error in Sc_load_and_align_data: {e}")

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_adj

def create_bias_matrix(adj_matrix_normalized_coo):
    degrees = np.array(adj_matrix_normalized_coo.sum(axis=1))
    bias_matrix = sp.diags(degrees.flatten())
    return bias_matrix

# Preprocess Visium data
preprocess_visium_data(args.visium_path, args.count_file, args.output_file)

# Process and save data
process_and_save_data(args.Spatial_dataset, args.expression_file, args.binary_labels_file, args.output_prefix, args.Drug)

# Load and preprocess data
train_data = Bulk_load_and_align_data(args.train_dataset, args.train_binary_labels_file, args.Drug)
val_data = Sc_load_and_align_data(args.Spatial_dataset)
