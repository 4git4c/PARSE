from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import degree
import torch
from types import SimpleNamespace
from ruamel.yaml import YAML
import numpy as np

class ModelN(nn.Module):
    def __init__(self, dataset_name, yaml_path,
                 running_device, feature_X, A_I_normal, nb_feature, nb_nodes,
                 label_matrix,  idx_train, 
                 custom_key="classification", results_dir=None):
        super().__init__()
        self.model_name = 'modelN'
        self.dataset_name = dataset_name
        self.custom_key = custom_key
        self.yaml_path = yaml_path
        self.args = self._load_args()
        self.model = RepBase(nb_feature, cfg=self.args.cfg, dropout=self.args.dropout)
        self.running_device = running_device
        self.nb_nodes = nb_nodes
        self.A_I_normal = A_I_normal
        self.feature_X = feature_X
        self.results_dir = results_dir
        
        self.label_matrix = label_matrix.numpy()
        
        self.idx_train = idx_train
        
    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.model = self.model.to(self.running_device)
        self.A_I_normal = self.A_I_normal.to(self.running_device)
        self.feature_X = self.feature_X.to(self.running_device)
        
        if self.args.dataset_name in ['ogbn-arxiv', 'ogbn-products', 'ogbn-mag']:
            self.feature_X = F.normalize(self.feature_X)
            self.feature_X = torch.spmm(self.A_I_normal, self.feature_X)

        A_degree = degree(self.A_I_normal._indices()[0], self.nb_nodes, dtype=int).tolist()
        if self.args.dataset_name in ['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'ogbn-mag']:
            edge_index = self.A_I_normal._indices()[0]
        else:
            edge_index = self.A_I_normal._indices()[1]
        
        # ===================================================#
        my_margin = self.args.margin1
        my_margin_2 = my_margin + self.args.margin2
        num_neg = self.args.NN
        lbl_z = torch.tensor([0.]).to(self.running_device)
        deg_list_2 = []
        deg_list_2.append(0)
        for i in range(self.nb_nodes):
            deg_list_2.append(deg_list_2[-1] + A_degree[i])
        idx_p_list = []
        for j in range(1, 101):
            random_list = [deg_list_2[i] + j % A_degree[i] for i in range(self.nb_nodes)]
            idx_p = edge_index[random_list]
            idx_p_list.append(idx_p)
            
        for current_iter, epoch in enumerate(tqdm(range(self.args.start_epoch, self.args.start_epoch + self.args.epochs + 1))):
            self.model.train()
            optimizer.zero_grad()
            
            h_a, h_p = self.model(self.feature_X, self.A_I_normal)

            h_p_1 = (h_a[idx_p_list[epoch % 100]] + h_a[idx_p_list[(epoch + 2) % 100]] + h_a[
                idx_p_list[(epoch + 4) % 100]] + h_a[idx_p_list[(epoch + 6) % 100]] + h_a[
                        idx_p_list[(epoch + 8) % 100]]) / 5

            label_weight_list = torch.ones([self.nb_nodes, self.label_matrix.shape[1]], device=self.running_device)
            label_weight_list = torch.tensor(self.label_matrix[self.idx_train], device=self.running_device).mean(0, keepdim=True).repeat(self.nb_nodes, 1)
            label_weight_list[self.idx_train] = torch.tensor(self.label_matrix[self.idx_train], device=self.running_device)
            hp1_weight_list = (label_weight_list[idx_p_list[epoch % 100]] + 
                                label_weight_list[idx_p_list[(epoch + 2) % 100]] + 
                                label_weight_list[idx_p_list[(epoch + 4) % 100]] + 
                                label_weight_list[idx_p_list[(epoch + 6) % 100]] + 
                                label_weight_list[idx_p_list[(epoch + 8) % 100]]) / 5
            hp1_weight_list = torch.min(label_weight_list, hp1_weight_list).sum(1) / ((torch.max(label_weight_list, hp1_weight_list)).sum(1) + 1e-8)
            hp1_norm = hp1_weight_list / (hp1_weight_list.sum() + 1e-8)
            
            idx_list = []
            neg_weight_list = torch.zeros((num_neg, self.nb_nodes)).to(self.running_device)
            for i in range(num_neg):
                idx_0 = np.random.permutation(self.nb_nodes)
                idx_list.append(idx_0)
                negw = label_weight_list[idx_0]
                neg_weight_list[i] = torch.min(label_weight_list, negw).sum(1) / ((torch.max(label_weight_list, negw)).sum(1) + 1e-8)
            neg_norm = neg_weight_list / neg_weight_list.sum(dim=1, keepdim=True)
            
            s_p = F.pairwise_distance(h_a, h_p)
            s_p_1 = F.pairwise_distance(h_a, h_p_1)
            s_n_list = []
            for h_n in idx_list:
                s_n = F.pairwise_distance(h_a, h_a[h_n])
                s_n_list.append(s_n)
            margin_label = -1 * torch.ones_like(s_p)

            loss_mar = 0
            loss_mar_1 = 0
            mask_margin_N = 0
            for i, s_n in enumerate(s_n_list):
                loss_mar += self.affinity_weighted_margin_ranking_loss(s_p, s_n, 1 / len(s_p), neg_norm[i], margin_label, my_margin).sum()
                loss_mar_1 += self.affinity_weighted_margin_ranking_loss(s_p_1, s_n, hp1_norm, neg_norm[i], margin_label, my_margin).sum()
                mask_margin_N += (torch.max((s_n - s_p.detach() - my_margin_2), lbl_z)).sum()
            mask_margin_N = mask_margin_N / num_neg

            loss = loss_mar * self.args.w_loss1 + loss_mar_1 * self.args.w_loss2 + mask_margin_N * self.args.w_loss3
            loss.backward()
            optimizer.step()
            if epoch % self.args.epochs == 0 and epoch != 0:
                self.model.eval()
                h_a, h_p = self.model.embed(self.feature_X, self.A_I_normal)
                embs = h_p
                embs = embs / embs.norm(dim=1)[:, None]
        if self.results_dir is not None:
            torch.save(self.model.state_dict(), f'{self.results_dir}/EXNweights.pth')
        return self.model.eval(), embs.detach()
    
    def affinity_weighted_margin_ranking_loss(self, d_pos, d_neg, alpha_pos, alpha_neg, target, margin=1.0):
        # # Compute squared distances
        weighted_pos = alpha_pos * d_pos
        weighted_neg = alpha_neg * d_neg
        losses = torch.clamp(-target * (weighted_pos - weighted_neg) + (2 * alpha_pos * alpha_neg) / (alpha_pos + alpha_neg) * margin, min=0)
        return losses

    def forward(self):
        self.model.eval()
        _, h_p = self.model.embed(self.feature_X, self.A_I_normal)
        embs = h_p
        embs = embs / embs.norm(dim=1)[:, None]
        return embs
    
    def _load_args(self):
        processed_key = self.custom_key.split("+")[0] if self.custom_key else ""
        
        defaults = {
            "gpu": 1,
            "black_list": None,
            "checkpoint_dir": "checkpoints",
            "model_name": 
                self.model_name,
            "custom_key": processed_key,
            "save_model": False,
            "data_root": "~/data",
            "dataset_name": self.dataset_name,
            "pretrain": False,
            "lr": 0.0025,
            "batch_size": 128,
            "epochs": 100,
            "start_epoch": 0,
            "lr2": 1e-2,
            "use_bn": False,
            "perf_task_for_val": "Node",
            "w_loss1": 1.0,
            "w_loss2": 1.0,
            "w_loss3": 1.0,
            "margin1": 0.8,
            "margin2": 0.2,
        }
        
        args_key = "-".join([
            self.model_name,
            self.dataset_name or self.dataset_class,
            processed_key
        ])
        
        yaml_config = {}
        try:
            with open(self.yaml_path) as f:
                yaml_data = YAML().load(f) or {}
                yaml_config = yaml_data.get(args_key, {})
        except KeyError:
            raise RuntimeError(f"Config key '{args_key}' not found in YAML")
        except FileNotFoundError:
            raise FileNotFoundError(f"YAML config file not found at: {self.yaml_path}")
        
        final_args = {**defaults, **yaml_config}
        final_args.update({ 
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "custom_key": processed_key,
        })
        
        return SimpleNamespace(**final_args)


class RepBase(nn.Module):
    def __init__(self, n_in ,cfg = None, dropout = 0.2):
        super(RepBase, self).__init__()
        self.MLP = make_mlplayers(n_in, cfg)
        self.act = nn.ReLU()
        self.dropout = dropout
        self.A = None
        self.sparse = True
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq_a, adj=None, edge_mask=None):
        if self.A is None:
            self.A = adj
        seq_a = F.dropout(seq_a, self.dropout, training=self.training)

        h_a = self.MLP(seq_a)
        h_p_0 = F.dropout(h_a, 0.2, training=self.training)
        if self.sparse:
            h_p = torch.sparse.mm(adj, h_p_0)
        else:
            h_p = torch.mm(adj, h_p_0)
        return h_a, h_p

    def embed(self,  seq_a , adj=None ):
        h_a = self.MLP(seq_a)
        if self.sparse:
            h_p = torch.spmm(adj, h_a)
        else:
            h_p = torch.mm(adj, h_a)
        return h_a.detach(), h_p.detach()
    
def make_mlplayers(in_channel, cfg, batch_norm=False, out_layer =None):
    layers = []
    in_channels = in_channel
    layer_num  = len(cfg)
    for i, v in enumerate(cfg):
        out_channels =  v
        mlp = nn.Linear(in_channels, out_channels)
        if batch_norm:
            layers += [mlp, nn.BatchNorm1d(out_channels, affine=False), nn.ReLU()]
        elif i != (layer_num-1):
            layers += [mlp, nn.ReLU()]
        else:
            layers += [mlp]
        in_channels = out_channels
    if out_layer != None:
        mlp = nn.Linear(in_channels, out_layer)
        layers += [mlp]
    return nn.Sequential(*layers)