import torch
from model import Model
from Bulk_preprocess import *
from sklearn.metrics import roc_auc_score
import random
import os
import dgl
import argparse

from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import NearestNeighbors

torch.set_num_threads(os.cpu_count() * 2)

def train_and_evaluate(params, run):
    lr = params['lr']
    weight_decay = params['weight_decay']
    embedding_dim = params['embedding_dim']
    batch_size = params['batch_size']
    subgraph_size = params['subgraph_size']
    dropout = params['dropout']
    alpha = params['alpha']
    beta = params['beta']

    print(f"Running with parameters: {params}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    seed = run + 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    features, Bulk_index, gene_names, labels_tensor = load_and_align_data(args.scRNA_dataset, args.expression_file, args.binary_labels_file)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]

    knn = NearestNeighbors(n_neighbors=10, algorithm='ball_tree')
    knn.fit(features)
    distances, indices = knn.kneighbors(features)
    num_nodes = indices.shape[0]
    k = knn.n_neighbors - 1
    rows = np.repeat(np.arange(num_nodes), k)
    cols = indices[:, 1:].flatten()
    data = np.ones_like(rows)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))

    src_nodes = np.repeat(np.arange(indices.shape[0]), indices.shape[1] - 1)
    dst_nodes = indices[:, 1:].flatten()
    dgl_graph = dgl.graph((src_nodes, dst_nodes))

    features = torch.FloatTensor(features.values)
    dgl_graph.ndata['feat'] = features

    assert dgl_graph.number_of_nodes() == len(labels_tensor), "Graph node count does not match label count"
    dgl_graph.ndata['labels'] = labels_tensor

    bias_matrix = create_bias_matrix(adj)
    adj_edge_modification = aug_random_edge(adj, 0.2)
    adj = normalize_adj(adj)
    adj_coo = adj.tocoo()
    adj = (adj + sp.eye(adj.shape[0])).todense()
    adj_hat = normalize_adj(adj_edge_modification)
    adj_hat = (adj_hat + sp.eye(adj_hat.shape[0])).todense()
    Bulk_edge_index = np.vstack((adj_coo.row, adj_coo.col))
    np.save('Bulk_edge_index.npy', Bulk_edge_index)

    bias_matrix = create_bias_matrix(adj)
    idx_train, idx_val, idx_test, train_mask, val_mask, test_mask = split_dataset(num_nodes, device=device)

    labels = dgl_graph.ndata['labels']
    adj = torch.FloatTensor(adj).to(device)
    adj_hat = torch.FloatTensor(adj_hat).to(device)
    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)
    first_cell_genes = features[0, :]
    print(f"First cell genes: {first_cell_genes}")

    all_auc = []
    dgl.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = Model(ft_size, embedding_dim, 'prelu', args.negsamp_ratio_patch, args.negsamp_ratio_context, args.readout, dropout).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.1, patience=10, verbose=True)
    b_xent_patch = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio_patch]).to(device))
    b_xent_context = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio_context]).to(device))

    cnt_wait = 0
    best = 1e9
    best_t = 0
    batch_num = nb_nodes // batch_size + 1

    for epoch in range(args.num_epoch):
        model.train()
        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)
        total_loss = 0.

        subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)
        for batch_idx in range(batch_num):
            optimiser.zero_grad()
            is_final_batch = (batch_idx == (batch_num - 1))
            if not is_final_batch:
                idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            else:
                idx = all_idx[batch_idx * batch_size:]

            cur_batch_size = len(idx)
            if cur_batch_size == 0:
                continue

            lbl_patch = torch.unsqueeze(torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio_patch))), 1).to(device)
            lbl_context = torch.unsqueeze(torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio_context))), 1).to(device)

            ba = []
            ba_hat = []
            bf = []
            added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
            added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
            added_adj_zero_col[:, -1, :] = 1.
            added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)
            for i in idx:
                if len(subgraphs[i]) == 0:
                    continue
                cur_adj = adj[subgraphs[i], :][:, subgraphs[i]].unsqueeze(0)
                cur_adj_hat = adj_hat[subgraphs[i], :][:, subgraphs[i]].unsqueeze(0)
                cur_feat = features[subgraphs[i], :].unsqueeze(0)
                ba.append(cur_adj)
                ba_hat.append(cur_adj_hat)
                bf.append(cur_feat)

            if not ba or not ba_hat or not bf:
                continue

            ba = torch.cat(ba)
            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)
            ba_hat = torch.cat(ba_hat)
            ba_hat = torch.cat((ba_hat, added_adj_zero_row), dim=1)
            ba_hat = torch.cat((ba_hat, added_adj_zero_col), dim=2)
            bf = torch.cat(bf)
            bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

            logits_1, logits_2, subgraph_embed, node_embed = model(bf, ba)
            logits_1_hat, logits_2_hat, subgraph_embed_hat, node_embed_hat = model(bf, ba_hat)

            subgraph_embed = F.normalize(subgraph_embed, dim=1, p=2)
            subgraph_embed_hat = F.normalize(subgraph_embed_hat, dim=1, p=2)
            sim_matrix_one = torch.matmul(subgraph_embed, subgraph_embed_hat.t())
            sim_matrix_two = torch.matmul(subgraph_embed, subgraph_embed.t())
            sim_matrix_three = torch.matmul(subgraph_embed_hat, subgraph_embed_hat.t())
            temperature = 1.0
            sim_matrix_one_exp = torch.exp(sim_matrix_one / temperature)
            sim_matrix_two_exp = torch.exp(sim_matrix_two / temperature)
            sim_matrix_three_exp = torch.exp(sim_matrix_three / temperature)
            nega_list = np.arange(0, cur_batch_size - 1, 1)
            nega_list = np.insert(nega_list, 0, cur_batch_size - 1)
            sim_row_sum = sim_matrix_one_exp[:, nega_list] + sim_matrix_two_exp[:, nega_list] + sim_matrix_three_exp[:, nega_list]
            sim_row_sum = torch.diagonal(sim_row_sum)
            sim_diag = torch.diagonal(sim_matrix_one)
            sim_diag_exp = torch.exp(sim_diag / temperature)
            NCE_loss = -torch.log(sim_diag_exp / (sim_row_sum))
            NCE_loss = torch.mean(NCE_loss)

            loss_all_1 = b_xent_context(logits_1, lbl_context)
            loss_all_1_hat = b_xent_context(logits_1_hat, lbl_context)
            loss_1 = torch.mean(loss_all_1)
            loss_1_hat = torch.mean(loss_all_1_hat)

            loss_all_2 = b_xent_patch(logits_2, lbl_patch)
            loss_all_2_hat = b_xent_patch(logits_2_hat, lbl_patch)
            loss_2 = torch.mean(loss_all_2)
            loss_2_hat = torch.mean(loss_all_2_hat)

            loss_1 = alpha * loss_1 + (1 - alpha) * loss_1_hat
            loss_2 = alpha * loss_2 + (1 - alpha) * loss_2_hat
            loss = beta * loss_1 + (1 - beta) * loss_2 + 0.1 * NCE_loss

            loss.backward()
            optimiser.step()

            loss = loss.detach().cpu().numpy()
            if not is_final_batch:
                total_loss += loss

        mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes
        scheduler.step(mean_loss)

        if mean_loss < best:
            best = mean_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), '{}.pkl'.format(args.dataset))
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!', flush=True)
            break

        print('Epoch:{} Loss:{:.8f}'.format(epoch, mean_loss), flush=True)

    return best

def hyperparameter_search():
    param_grid = {
        'lr': [1e-3, 1e-4, 1e-5],
        'weight_decay': [1e-2, 1e-3, 1e-4],
        'embedding_dim': [512],
        'batch_size': [32, 64, 128],
        'subgraph_size': [5,10, 20, 30],
        'dropout': [0.1,0.2,0.4,0.3, 0.5, 0.7],
        'alpha': [0.3, 0.5, 0.7],
        'beta': [0.3, 0.5, 0.7]
    }

    best_params = None
    best_loss = float('inf')

    for lr in param_grid['lr']:
        for weight_decay in param_grid['weight_decay']:
            for embedding_dim in param_grid['embedding_dim']:
                for batch_size in param_grid['batch_size']:
                    for subgraph_size in param_grid['subgraph_size']:
                        for dropout in param_grid['dropout']:
                            for alpha in param_grid['alpha']:
                                for beta in param_grid['beta']:
                                    params = {
                                        'lr': lr,
                                        'weight_decay': weight_decay,
                                        'embedding_dim': embedding_dim,
                                        'batch_size': batch_size,
                                        'subgraph_size': subgraph_size,
                                        'dropout': dropout,
                                        'alpha': alpha,
                                        'beta': beta
                                    }

                                    loss = train_and_evaluate(params, run=0)

                                    if loss < best_loss:
                                        best_loss = loss
                                        best_params = params

                                    print(f"Tested parameters: {params}, Loss: {loss}")

    print(f"Best parameters: {best_params}, Loss: {best_loss}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepTNR')
    parser.add_argument('--expid', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='CRC1.h5ad')
    parser.add_argument('--scRNA_dataset', type=str, default='CRC1.h5ad', help='Path to the scRNA-seq Data file')
    parser.add_argument('--expression_file', type=str, default='cellLineEncoderPretrainData.csv', help='Path to the Bulk RNA-seq Data file.')
    parser.add_argument('--binary_labels_file', type=str, default='binary_drug_response.csv', help='Path to the binary labels data file.')
    parser.add_argument('--num_epoch', type=int, default=500)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--negsamp_ratio_patch', type=int, default=10)
    parser.add_argument('--negsamp_ratio_context', type=int, default=1)
    parser.add_argument('--readout', type=str, default='avg')
    args = parser.parse_args()

    hyperparameter_search()
