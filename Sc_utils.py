import scanpy as sc
import anndata
import dgl
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MaxAbsScaler
import scipy
import networkx as nx
import scipy.sparse as sp
import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
import torch

def load_and_align_data(scRNA_dataset, expression_file, binary_labels_file):
    # 加载scRNA-seq数据
    adata = sc.read_h5ad(scRNA_dataset)
    #sc.pp.normalize_total(adata, target_sum=1e4)
    #sc.pp.log1p(adata)
    #sc.pp.highly_variable_genes(adata, batch_key='dataset', subset=True)
    #adata = adata[:, adata.var.highly_variable]
    cell_ids = adata.obs_names.tolist()
    genes = adata.var_names.tolist()

    # 加载Bulk RNA-seq数据
    Bulk_expression = pd.read_csv(expression_file, index_col=[0, 1])
    binary_labels_data = pd.read_csv(binary_labels_file, index_col=[0, 1])
    cell_lines = Bulk_expression.index.get_level_values(0).tolist()
    genes_bulk = Bulk_expression.columns.tolist()

    # 对齐基因
    common_genes = list(set(genes).intersection(genes_bulk))
    adata = adata[:, common_genes]
    Bulk_expression = Bulk_expression[common_genes]

    # 提取scRNA-seq特征矩阵和邻接矩阵
    expr_matrix = adata.X
    scaler = MaxAbsScaler()
    scRNA_features = scaler.fit_transform(expr_matrix)
    scRNA_features = scRNA_features.todense()
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
    sc.tl.umap(adata)
    scRNA_adj = adata.obsp['connectivities']

    # 提取Bulk RNA-seq特征矩阵
    #Bulk_genes = pd.DataFrame(data=Bulk_expression.values, index=Bulk_expression.index, columns=common_genes)
    #Bulk_features = scaler.fit_transform(Bulk_expression.values)
    #Bulk_features = np.log1p(Bulk_features)
    #Bulk_features = scaler.fit_transform(Bulk_features)
   # Bulk_nb_nodes = Bulk_features.shape[0]
    #Bulk_ft_size = Bulk_features.shape[1]

    # 给Bulk RNA-seq特征矩阵加上细胞系编号
    #Bulk_features_df = pd.DataFrame(Bulk_features, index=Bulk_expression.index, columns=common_genes)
    #Bulk_features_df['Cell_Line_ID'] = range(len(Bulk_features_df))
    #Bulk_features_df.set_index('Cell_Line_ID', append=True, inplace=True)

    # 存储索引信息
    scRNA_index = list(adata.obs.index)
    #Bulk_index = list(Bulk_features_df.index)
    gene_names = common_genes

    # 将scRNA-seq特征矩阵转换为torch.FloatTensor
    scRNA_features_tensor = torch.FloatTensor(scRNA_features)

    return adata, scRNA_features_tensor, scRNA_adj, scRNA_index, gene_names


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_adj

def create_bias_matrix(adj_matrix_normalized_coo):
    # 计算每个节点的度数
    """
    The create_bias_matrix function takes in a normalized adjacency matrix and returns the bias matrix.
    
    :param adj_matrix_normalized_coo: Create the bias matrix
    :return: A diagonal matrix with the degrees of each node on the diagonal
    :doc-author: Trelent
    """
    degrees = np.array(adj_matrix_normalized_coo.sum(axis=1))
    # 构建对角线偏置矩阵
    bias_matrix = sp.diags(degrees.flatten())
    return bias_matrix


def train_val_test_split(adata, num_samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=None):
    assert train_ratio + val_ratio + test_ratio == 1.0, "Train, validation, and test ratios must sum to 1.0"
    assert train_ratio > 0 and val_ratio > 0 and test_ratio > 0, "Ratios must be greater than 0"
    assert 0 <= train_ratio <= 1 and 0 <= val_ratio <= 1 and 0 <= test_ratio <= 1, "Ratios must be between 0 and 1"
    
    np.random.seed(seed)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    test_size = num_samples - train_size - val_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    mask_train = np.zeros(num_samples, dtype=bool)
    mask_val = np.zeros(num_samples, dtype=bool)
    mask_test = np.zeros(num_samples, dtype=bool)
    mask_train[train_indices] = True
    mask_val[val_indices] = True
    mask_test[test_indices] = True
    combined_labels = adata.obs['Cancer'].astype(str) + '_' + adata.obs['dataset'].astype(str) + '_' + adata.obs['in_tissue'].astype(str)

    onehot_encoder = OneHotEncoder()
    labels = onehot_encoder.fit_transform(combined_labels.values.reshape(-1, 1)).toarray()
    nb_classes = labels.shape[1]



    return train_indices, val_indices, test_indices,mask_train, mask_val, mask_test,labels,nb_classes


def generate_rwr_subgraph(dgl_graph, subgraph_size):
    # 获取所有节点索引
    all_nodes = torch.arange(dgl_graph.number_of_nodes(), dtype=torch.int64)
    # 初始化子图节点列表
    subv = []

    # 对每个节点进行处理
    for node in all_nodes:
        # 随机游走，包括重启概率
        traces, _ = dgl.sampling.random_walk(
            g=dgl_graph,
            nodes=torch.tensor([node]),
            length=subgraph_size * 3,  # 假设最大长度
            restart_prob=torch.full((subgraph_size * 3,), 0.9)  # 除第一步外，后续步骤有较高的继续概率
        )

        # 提取并处理轨迹
        trace = traces[0]
        trace = trace[trace != -1]  # 移除填充值
        unique_nodes = torch.unique(trace)
        
        # 若轨迹长度不足，尝试重新游走
        retry_time = 0
        while len(unique_nodes) < subgraph_size and retry_time < 10:
            retry_traces, _ = dgl.sampling.random_walk(
                g=dgl_graph,
                nodes=torch.tensor([node]),
                length=subgraph_size * 5,  # 允许更长的游走
                restart_prob=torch.full((subgraph_size * 5,), 0.1)  # 降低重启概率，鼓励更长的游走
            )
            retry_trace = retry_traces[0]
            retry_trace = retry_trace[retry_trace != -1]
            unique_nodes = torch.unique(retry_trace)
            retry_time += 1

        # 确保子图大小，若仍不足，则填充；若过多，进行截断
        if len(unique_nodes) > subgraph_size:
            unique_nodes = unique_nodes[:subgraph_size]
        elif len(unique_nodes) < subgraph_size:
            while len(unique_nodes) < subgraph_size:
                # 重复填充直至达到期望大小
                unique_nodes = torch.cat([unique_nodes, unique_nodes[:subgraph_size - len(unique_nodes)]])

        subv.append(unique_nodes.tolist())

    return subv

def aug_random_edge(input_adj, drop_percent=0.2):
    """
    Augment graph data by randomly deleting and adding edges.

    Parameters:
    - input_adj: Input adjacency matrix in scipy.sparse format.
    - drop_percent: The percentage of edges to randomly delete and add.

    Returns:
    - new_adj: New adjacency matrix after random edge modifications.
    """

    # Ensure input_adj is in csr format for efficient row slicing.
    if not isinstance(input_adj, sp.csr_matrix):
        input_adj = input_adj.tocsr()

    # Get indices of non-zero elements (edges).
    row_idx, col_idx = input_adj.nonzero()
    num_edges = len(row_idx)
    num_drop = int(num_edges * drop_percent / 2)
    
    # Randomly select edges to drop.
    drop_indices = np.random.choice(np.arange(num_edges), size=num_drop, replace=False)
    
    # Create a mask to keep the edges that are not dropped.
    keep_mask = np.ones(num_edges, dtype=bool)
    keep_mask[drop_indices] = False
    row_idx_kept = row_idx[keep_mask]
    col_idx_kept = col_idx[keep_mask]

    # Prepare for adding edges.
    num_nodes = input_adj.shape[0]
    existing_edges = set(zip(row_idx, col_idx))
    possible_adds = []

    while len(possible_adds) < num_drop:
        # Randomly generate a pair of nodes.
        rand_row = np.random.randint(0, num_nodes)
        rand_col = np.random.randint(0, num_nodes)
        
        # If this pair does not form a self-loop and the edge does not already exist,
        # add it to the list of possible additions.
        if rand_row != rand_col and (rand_row, rand_col) not in existing_edges:
            possible_adds.append((rand_row, rand_col))
            # Update the set of existing edges to avoid adding duplicate edges.
            existing_edges.add((rand_row, rand_col))

    # Extract row and column indices from the list of additions.
    add_row_idx, add_col_idx = zip(*possible_adds)

    # Combine kept and new edges.
    final_row_idx = np.hstack([row_idx_kept, add_row_idx])
    final_col_idx = np.hstack([col_idx_kept, add_col_idx])
    data = np.ones_like(final_row_idx)

    # Create a new adjacency matrix with the updated edges.
    new_adj = sp.csr_matrix((data, (final_row_idx, final_col_idx)), shape=input_adj.shape)

    return new_adj
