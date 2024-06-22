from model import Model
from Sc_utils import *
from sklearn.metrics import roc_auc_score
import random
import os
import dgl
import argparse

from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='spatialdrug')
parser.add_argument('--expid', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dataset', type=str, default='CRC1.h5ad')
parser.add_argument('--scRNA_dataset', type=str, default='CRC1.h5ad', help='Path to the scRNA-seq Data file')
parser.add_argument('--expression_file', type=str, default='Expression_Public_23Q4_filtered.csv',
                    help='Path to the Bulk RNA-seq Data file.')
parser.add_argument('--binary_labels_file', type=str, default='binary_labels_4.csv',
                    help='Path to the binary labels data file.')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--embedding_dim', type=int, default=512)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--num_epoch', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--subgraph_size', type=int, default=6)
parser.add_argument('--readout', type=str, default='avg')
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--negsamp_ratio_patch', type=int, default=10)
parser.add_argument('--negsamp_ratio_context', type=int, default=2)
parser.add_argument('--dropout', type=int, default=0.1)

parser.add_argument('--alpha', type=float, default=0.1, help='how much the first view involves')
parser.add_argument('--beta', type=float, default=0.1, help='how much the second view involves')
args = parser.parse_args()

if __name__ == '__main__':
    print('Dataset: {}'.format(args.dataset), flush=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    for run in range(args.runs):
        seed = run + 1
        random.seed(seed)
        batch_size = args.batch_size
        subgraph_size = args.subgraph_size
        adata, features, adj, scRNA_index, gene_names = load_and_align_data(args.scRNA_dataset, args.expression_file, args.binary_labels_file)
        nb_nodes = features.shape[0]
        ft_size = features.shape[1]
        num_samples = adata.shape[0]

        train_indices, val_indices, test_indices, train_mask, val_mask, test_mask, labels, nb_classes = train_val_test_split(adata, num_samples)

        adj_edge_modification = aug_random_edge(adj, 0.2)
        adj = normalize_adj(adj)
        adj_coo = adj.tocoo()
        adj = (adj + sp.eye(adj.shape[0])).todense()
        adj_hat = normalize_adj(adj_edge_modification)
        adj_hat = (adj_hat + sp.eye(adj_hat.shape[0])).todense()
        dgl_graph = dgl.graph((adj_coo.row, adj_coo.col), num_nodes=adata.shape[0])
        dgl_graph.ndata['expr'] = torch.tensor(features.toarray()) if scipy.sparse.issparse(features) else torch.tensor(features)
        scrna_edge_index = np.vstack((adj_coo.row, adj_coo.col))
        np.save('scrna_edge_index.npy', scrna_edge_index)


        subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)
        bias_matrix = create_bias_matrix(adj)

        adj = torch.FloatTensor(adj[np.newaxis]).to(device)
        adj_hat = torch.FloatTensor(adj_hat[np.newaxis]).to(device)
        labels = torch.FloatTensor(labels[np.newaxis]).to(device)
        idx_train = torch.LongTensor(train_indices).to(device)
        idx_val = torch.LongTensor(val_indices).to(device)
        idx_test = torch.LongTensor(test_indices).to(device)

        all_auc = []
        print('\n# Run:{} with random seed:{}'.format(run, seed), flush=True)
        dgl.random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        os.environ['PYTHONHASHSEED'] = str(seed)
        model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio_patch, args.negsamp_ratio_context,
                      args.readout,args.dropout).to(device)  # Ensure all parameters are passed
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        b_xent_patch = nn.BCEWithLogitsLoss(reduction='none',
                                            pos_weight=torch.tensor([args.negsamp_ratio_patch]).to(device))
        b_xent_context = nn.BCEWithLogitsLoss(reduction='none',
                                              pos_weight=torch.tensor([args.negsamp_ratio_context]).to(device))        

        cnt_wait = 0
        best = 1e9
        best_t = 0
        batch_num = nb_nodes // batch_size + 1
        print("hhh")
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

                lbl_patch = torch.unsqueeze(torch.cat(
                    (torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio_patch))), 1).to(device)

                lbl_context = torch.unsqueeze(torch.cat(
                    (torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio_context))), 1).to(device)

                ba = []
                ba_hat = []
                bf = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)
                for i in idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_adj_hat = adj_hat[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[subgraphs[i], :]  # 确保只使用两个索引

                    ba.append(cur_adj)
                    ba_hat.append(cur_adj_hat)
                    bf.append(cur_feat)

                ba = torch.cat(ba)
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)
                ba_hat = torch.cat(ba_hat)
                ba_hat = torch.cat((ba_hat, added_adj_zero_row), dim=1)
                ba_hat = torch.cat((ba_hat, added_adj_zero_col), dim=2)
                bf = torch.stack(bf)  # 使用stack确保形状一致
                bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)  # 确保形状一致

                logits_1, logits_2, subgraph_embed, node_embed = model(bf, ba)
                logits_1_hat, logits_2_hat, subgraph_embed_hat, node_embed_hat = model(bf, ba_hat)

                # subgraph-subgraph contrast loss
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

                loss_1 = args.alpha * loss_1 + (1 - args.alpha) * loss_1_hat  # node-subgraph contrast loss
                loss_2 = args.alpha * loss_2 + (1 - args.alpha) * loss_2_hat  # node-node contrast loss
                loss = args.beta * loss_1 + (1 - args.beta) * loss_2 + 0.1 * NCE_loss  # total loss

                loss.backward()
                optimiser.step()

                loss = loss.detach().cpu().numpy()
                if not is_final_batch:
                    total_loss += loss

            mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes

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

        torch.save(model.state_dict(), f'model_{run+1}.pth')
        print(f'Model from run {run+1} saved.')

        model.load_state_dict(torch.load('{}.pkl'.format(args.dataset)))
        model.eval()

        # 获取所有节点的嵌入
        with torch.no_grad():
            all_node_embeds = []
            all_subgraph_embeds = []

            for batch_idx in range(batch_num):
                is_final_batch = (batch_idx == (batch_num - 1))
                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)
                ba = []
                ba_hat = []
                bf = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)
                for i in idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_adj_hat = adj_hat[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[subgraphs[i], :]  # 确保只使用两个索引

                    ba.append(cur_adj)
                    ba_hat.append(cur_adj_hat)
                    bf.append(cur_feat)

                ba = torch.cat(ba)
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)
                ba_hat = torch.cat(ba_hat)
                ba_hat = torch.cat((ba_hat, added_adj_zero_row), dim=1)
                ba_hat = torch.cat((ba_hat, added_adj_zero_col), dim=2)
                bf = torch.stack(bf)  # 使用stack确保形状一致
                bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)  # 确保形状一致

                _, _, subgraph_embed, node_embed = model(bf, ba)
                all_node_embeds.append(node_embed.cpu().numpy())
                all_subgraph_embeds.append(subgraph_embed.cpu().numpy())

            all_node_embeds = np.concatenate(all_node_embeds, axis=0)
            all_subgraph_embeds = np.concatenate(all_subgraph_embeds, axis=0)

            # 保存节点嵌入为numpy格式
            np.save('scRNA_node_embeddings.npy', all_node_embeds)
            np.save('scRNA_subgraph_embeddings.npy', all_subgraph_embeds)
            print(f'Node embeddings and subgraph embeddings saved as numpy arrays.')

    print('Training completed.')

