import torch
import random
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

def add_random_edges(edge_index, num_nodes, edge_modify_ratio):
    num_edges_to_add = int(edge_modify_ratio * edge_index.size(1))
    new_edges = torch.randint(0, num_nodes, (2, num_edges_to_add))
    new_edge_index = torch.cat([edge_index, new_edges], dim=1)
    return new_edge_index, num_edges_to_add

def remove_random_edges(edge_index, edge_modify_ratio):
    num_edges_to_remove = int(edge_modify_ratio * edge_index.size(1))
    indices_to_remove = torch.randperm(edge_index.size(1))[:num_edges_to_remove]
    new_edge_index = torch.cat([edge_index[:, :indices_to_remove[0]], edge_index[:, indices_to_remove[-1]+1:]], dim=1)
    return new_edge_index, num_edges_to_remove

def generate_subgraphs(features, edge_index, num_subgraphs=5, walk_length=3, edge_modify_ratio=0.1, sample_ratio=0.8, add_edges=False, noise_factor=0.01):
    subgraphs = []
    num_nodes = features.size(0)
    num_edges = edge_index.size(1)

    for _ in range(num_subgraphs):
        if add_edges:
            modified_edge_index, _ = add_random_edges(edge_index, num_nodes, edge_modify_ratio)
        else:
            modified_edge_index, _ = remove_random_edges(edge_index, edge_modify_ratio)

        subset, new_edge_index, _, _ = k_hop_subgraph(
            random.sample(range(num_nodes), int(sample_ratio * num_nodes)),
            walk_length,
            modified_edge_index,
            relabel_nodes=True
        )

        subgraph_features = features[subset]

        if noise_factor > 0:
            subgraph_features = augment_features(subgraph_features, noise_factor)

        subgraph_data = Data(x=subgraph_features, edge_index=new_edge_index)

        subgraphs.append(subgraph_data)

    return subgraphs

def augment_features(features, noise_factor=0.1):
    noise = torch.randn_like(features) * noise_factor
    return features + noise