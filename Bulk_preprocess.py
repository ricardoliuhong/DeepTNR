import pandas as pd
import numpy as np
import torch
import dgl
import scipy.sparse as sp
import scanpy as sc
from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split


def load_and_align_data(scRNA_dataset, expression_file, binary_labels_file):
    # 加载scRNA-seq数据
    adata = sc.read_h5ad(scRNA_dataset)
    cell_ids = adata.obs_names.tolist()
    genes = adata.var_names.tolist()

    # 加载Bulk RNA-seq数据
    Bulk_expression = pd.read_csv(expression_file, index_col=[0])
    binary_labels_data = pd.read_csv(binary_labels_file, index_col=[0])
    cell_lines = Bulk_expression.index.get_level_values(0).tolist()
    genes_bulk = Bulk_expression.columns.tolist()

    # 对齐基因
    common_genes = list(set(genes).intersection(genes_bulk))
    adata = adata[:, common_genes]
    Bulk_expression = Bulk_expression[common_genes]

    # 提取scRNA-seq特征矩阵和邻接矩阵
    expr_matrix = adata.X
    scaler = MaxAbsScaler()

    # 提取Bulk RNA-seq特征矩阵
    Bulk_genes = pd.DataFrame(data=Bulk_expression.values, index=Bulk_expression.index, columns=common_genes)
    Bulk_features = scaler.fit_transform(Bulk_expression.values)
    Bulk_features = np.log1p(Bulk_features)
    Bulk_features = scaler.fit_transform(Bulk_features)

    # 给Bulk RNA-seq特征矩阵加上细胞系编号
    Bulk_features= pd.DataFrame(Bulk_features, index=Bulk_expression.index, columns=common_genes)
    Bulk_features['Cell_Line_ID'] = range(len(Bulk_features))
    Bulk_features.set_index('Cell_Line_ID', append=True, inplace=True)
    if None in Bulk_features.columns:
        print("None found in columns. Replacing None with 0.")
        Bulk_features.fillna(0, inplace=True)
    
    # 为每个细胞系创建标签矩阵
    unique_drugs = binary_labels_data['name'].unique()
    labels_matrix = pd.DataFrame(0, index=Bulk_features.index, columns=unique_drugs)

    for depmap_id, group in binary_labels_data.groupby('DepMap_ID'):
        for _, row in group.iterrows():
            if (depmap_id, labels_matrix.index.get_level_values('Cell_Line_ID')[0]) in labels_matrix.index:
                labels_matrix.loc[(depmap_id, labels_matrix.index.get_level_values('Cell_Line_ID')[0]), row['name']] = row['binary_auc']

    # 将标签矩阵转换为tensor
    labels_tensor = torch.tensor(labels_matrix.values, dtype=torch.float32)

    # 存储索引信息
    Bulk_index = list(Bulk_features.index)
    gene_names = common_genes

    return Bulk_features, Bulk_index, gene_names, labels_tensor




def split_dataset(num_nodes, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, device='cpu'):
    all_indices = np.arange(num_nodes)
    
    train_mask = np.random.rand(num_nodes) < train_ratio
    val_mask = np.random.rand(num_nodes) < val_ratio
    test_mask = np.random.rand(num_nodes) < test_ratio
    
    train_indices = all_indices[train_mask]
    val_indices = all_indices[val_mask]
    test_indices = all_indices[test_mask]
    
    idx_train = torch.LongTensor(train_indices).to(device)
    idx_val = torch.LongTensor(val_indices).to(device)
    idx_test = torch.LongTensor(test_indices).to(device)
    
    return idx_train, idx_val, idx_test, train_mask, val_mask, test_mask

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

    degrees = np.array(adj_matrix_normalized_coo.sum(axis=1))

    bias_matrix = sp.diags(degrees.flatten())
    return bias_matrix

def create_multi_label_matrix(df_all_label):
    # 创建一个空的多标签矩阵
    multi_label_matrix = pd.DataFrame(index=df_all_label.index, columns=df_all_label.columns)

    # 遍历每个细胞系和药物，填充多标签矩阵
    for cell_line in df_all_label.index:
        for drug in df_all_label.columns:
            # 如果细胞系对该药物耐药，则标记为 0，否则标记为 1
            if df_all_label.loc[cell_line, drug] == 'resistant':
                multi_label_matrix.loc[cell_line, drug] = 0
            else:
                multi_label_matrix.loc[cell_line, drug] = 1
    
    return multi_label_matrix


def add_drug_sensitivity_labels(dgl_graph, df_all_label, df_all_expression):
    labels = {}
    for cell_line in df_all_expression.index:
        if cell_line in df_all_label.index:
            for drug_name in df_all_label.columns:
                label_value = df_all_label.loc[cell_line, drug_name]
                label_numeric = 1 if label_value == 'sensitive' else 0
                node_id = df_all_expression.index.get_loc(cell_line)
                if drug_name not in dgl_graph.ndata:
                    dgl_graph.ndata[drug_name] = torch.zeros(dgl_graph.number_of_nodes(), dtype=torch.float32)
                dgl_graph.ndata[drug_name][node_id] = torch.tensor(label_numeric, dtype=torch.float32)
                if node_id not in labels:
                    labels[node_id] = {}
                labels[node_id][drug_name] = label_numeric

    # 将标签数据转换成指定格式的字典
    converted_labels = {}
    for node_id, drug_labels in labels.items():
        converted_labels[node_id] = {drug_name: 'sensitive' if label_value == 1 else 'resistant' for drug_name, label_value in drug_labels.items()}
    
    return converted_labels

def generate_labels(df_all_label, df_all_expression):
    labels = {}
    for drug_name in df_all_label.columns:
        drug_sensitivity_labels = df_all_label[drug_name].values
        cell_lines = df_all_label.index.values
        for cell_line, label in zip(cell_lines, drug_sensitivity_labels):
            if cell_line in df_all_expression.index:
                node_id = df_all_expression.index.get_loc(cell_line)
                if node_id not in labels:
                    labels[node_id] = {}
                label_value = 1 if label == 'sensitive' else 0
                labels[node_id][drug_name] = label_value
    return labels

def labels_to_array(labels):
    # 获取节点数量和药物数量
    num_nodes = len(labels)
    num_drugs = len(next(iter(labels.values())))

    # 初始化一个二维数组来存储标签数据
    labels_array = np.zeros((num_nodes, num_drugs), dtype=np.float32)

    # 获取药物名称列表
    drug_names = list(next(iter(labels.values())).keys())

    # 将 labels 字典中的数据转移到数组中
    for node_id, drug_labels in labels.items():
        for drug_id, label_value in drug_labels.items():
            drug_idx = drug_names.index(drug_id)  # 获取药物的索引位置
            labels_array[node_id, drug_idx] = label_value

    return labels


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
