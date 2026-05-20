import torch
import torch.nn as nn

def node_subgraph_contrast_loss(model, node_embeddings, subgraph_embeddings, tau):
    # 计算相似度矩阵
    sim_matrix = model.sim(node_embeddings, subgraph_embeddings) / tau  # 使用 tau 参数
    
    # 确保相似度矩阵的形状是 [节点数量, 子图数量]
    assert sim_matrix.shape == (node_embeddings.size(0), subgraph_embeddings.size(0)), "相似度矩阵形状不匹配"
    
    # 构造正样本标签
    batch_size = node_embeddings.size(0)
    labels = torch.arange(batch_size).to(node_embeddings.device)
    
    # 计算交叉熵损失
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(sim_matrix, labels)
    
    return loss

def subgraph_subgraph_contrast_loss(model, subgraph_embeddings1, subgraph_embeddings2, tau):
    # 计算相似度矩阵
    sim_matrix = model.sim(subgraph_embeddings1, subgraph_embeddings2) / tau  # 使用 tau 参数
    
    # 确保相似度矩阵的形状是 [子图数量1, 子图数量2]
    assert sim_matrix.shape == (subgraph_embeddings1.size(0), subgraph_embeddings2.size(0)), "相似度矩阵形状不匹配"
    
    # 构造正样本标签
    min_size = min(subgraph_embeddings1.size(0), subgraph_embeddings2.size(0))
    labels = torch.arange(min_size).to(subgraph_embeddings1.device)
    
    # 计算交叉熵损失
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(sim_matrix[:min_size, :min_size], labels)
    
    return loss
