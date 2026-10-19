import logging
import numpy as np

from sklearn.metrics import pairwise_distances
import torch
from torch import Tensor
from torch.nn import ReLU, Sequential
import torch.nn.functional as F
from torch_geometric.utils import degree

from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.config import (
    ExplanationType,
    ModelMode,
    ModelTaskLevel,
)
from torch_geometric.nn import Linear
from torch_geometric.nn.inits import reset
from tqdm import tqdm

class ModelE(ExplainerAlgorithm):
    coeffs = {
        'edge_size': 0.05,
        'edge_ent': 1.0,
        'temp': [10.0, 1.0],
        'bias': 0.01,
    }

    def __init__(self, epochs: int, features, adj, 
                 label_matrix,  idx_train, 
                 lr: float = 0.003, results_dir=None, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)
        self.results_dir = results_dir

        self.mlp = Sequential(
            Linear(-1, 64),
            ReLU(),
            Linear(64, 1),
        )
        self.mlp = self.mlp.to(features.device)
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        self._curr_epoch = -1
        
        self.recon_edge_index, self.edge_fatures = self.build_edge_features(adj.to(features.device), features, consider_direction=True, include_self_loop=True)

        label_matrix[~idx_train] = 0
        self.label_matrix = label_matrix.numpy()
        self.nb_nodes = features.shape[0]
        self.running_device = features.device
        
        self.idx_train = idx_train
        self.distance_matrix, self.global_to_labeled, self.default_distance = self.compute_jaccard_similarity_train(self.label_matrix, idx_train)
        self.distance_matrix = torch.from_numpy(self.distance_matrix).float()
        
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.mlp)

    def train(
        self,
        model: torch.nn.Module,
        x: Tensor,
        A_I: Tensor,
        en_args
    ):
        model.eval()
        
        nb_nodes = x.shape[0]
        A_degree = degree(A_I._indices()[0], nb_nodes, dtype=int).tolist()
        edge_index = A_I._indices()[1]
        
        my_margin = en_args.margin1
        my_margin_2 = my_margin + en_args.margin2
        margin_loss = torch.nn.MarginRankingLoss(margin=my_margin, reduction='none')
        num_neg = en_args.NN
        lbl_z = torch.tensor([0.]).to(x.device)
        deg_list_2 = []
        deg_list_2.append(0)
        for i in range(nb_nodes):
            deg_list_2.append(deg_list_2[-1] + A_degree[i])
            
        idx_p_list = []
        edge_idx_list = []
        for j in range(1, 101):
            random_list = [deg_list_2[i] + j % A_degree[i] for i in range(nb_nodes)]
            edge_idx_list.append(random_list)
            idx_p = edge_index[random_list]
            idx_p_list.append(idx_p)
        
        for epoch in tqdm(range(self.epochs)):
            self.optimizer.zero_grad()
            
            temperature = self._get_temperature(epoch)
            logits = self.mlp(self.edge_fatures).view(-1)
            edge_mask = self._concrete_sample(logits, temperature).sigmoid()

            edge_input = torch.sparse_coo_tensor(
                A_I._indices(),
                edge_mask,
                size=A_I.size(),
                device=x.device
            )
            
            with torch.no_grad():
                h_a, h_p = model(x, edge_input)
                
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
            
            h_p_1 = (h_a[idx_p_list[epoch % 100]] * edge_input._values()[edge_idx_list[epoch % 100]].unsqueeze(-1) + 
                     h_a[idx_p_list[(epoch + 2) % 100]] * edge_input._values()[edge_idx_list[(epoch + 2) % 100]].unsqueeze(-1) + 
                     h_a[idx_p_list[(epoch + 4) % 100]]  * edge_input._values()[edge_idx_list[(epoch + 4) % 100]].unsqueeze(-1) + 
                     h_a[idx_p_list[(epoch + 6) % 100]]  * edge_input._values()[edge_idx_list[(epoch + 6) % 100]].unsqueeze(-1) + 
                     h_a[idx_p_list[(epoch + 8) % 100]]  * edge_input._values()[edge_idx_list[(epoch + 8) % 100]].unsqueeze(-1)) / (
                         edge_input._values()[edge_idx_list[epoch % 100]].unsqueeze(-1) + edge_input._values()[edge_idx_list[(epoch + 2) % 100]].unsqueeze(-1) +
                         edge_input._values()[edge_idx_list[(epoch + 4) % 100]].unsqueeze(-1) + edge_input._values()[edge_idx_list[(epoch + 6) % 100]].unsqueeze(-1) +
                         edge_input._values()[edge_idx_list[(epoch + 8) % 100]].unsqueeze(-1) + 1e-8
                     )
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

            con_loss = loss_mar * en_args.w_loss1 + loss_mar_1 * en_args.w_loss2 + mask_margin_N * en_args.w_loss3
            
            reg_loss = self._regularization_loss(edge_mask)
            loss = (con_loss + reg_loss)
            loss.backward()
            self.optimizer.step()
            self._curr_epoch = epoch
        if self.results_dir is not None:
            torch.save(self.mlp.state_dict(), f'{self.results_dir}/EXEweights.pth')
        emb_hat = h_p / h_p.norm(dim=1)[:, None]
        self.mlp.eval()
        with torch.no_grad():
            temperature = self._get_temperature(epoch)
            logits = self.mlp(self.edge_fatures).view(-1)
            edge_mask = self._concrete_sample(logits, temperature).sigmoid()
        return emb_hat.detach(), logits.sigmoid().detach()
    
    def affinity_weighted_margin_ranking_loss(self, d_pos, d_neg, alpha_pos, alpha_neg, target, margin=1.0):
        # # Compute squared distances
        weighted_pos = alpha_pos * d_pos
        weighted_neg = alpha_neg * d_neg
        losses = torch.clamp(-target * (weighted_pos - weighted_neg) +  (2 * alpha_pos * alpha_neg / (alpha_pos+alpha_neg)) * margin, min=0)
        return losses
    
    def compute_jaccard_similarity_train(self, Y, idx_train):
        labeled_indices = idx_train
        labeled_Y = Y[labeled_indices]
        labeled_Y = labeled_Y.astype(bool)
        
        distance_matrix = 1 - pairwise_distances(labeled_Y, metric='jaccard', n_jobs=-1)
        
        default_value = distance_matrix.mean() if len(labeled_indices) > 0 else 0.5
        
        global_to_labeled = np.full(self.nb_nodes, -1, dtype=int)
        for label_idx, global_idx in enumerate(labeled_indices):
            global_to_labeled[global_idx] = label_idx
        
        return distance_matrix, global_to_labeled, default_value
        
    def forward(
        self,
    ) -> Explanation:

        logits = self.mlp(self.edge_fatures).view(-1)
        edge_mask = self._post_process_mask(logits, None, apply_sigmoid=True)
        return Explanation(edge_mask=edge_mask)

    def supports(self) -> bool:
        explanation_type = self.explainer_config.explanation_type
        if explanation_type != ExplanationType.phenomenon:
            logging.error(f"'{self.__class__.__name__}' only supports "
                          f"phenomenon explanations "
                          f"got (`explanation_type={explanation_type.value}`)")
            return False

        task_level = self.model_config.task_level
        if task_level not in {ModelTaskLevel.node, ModelTaskLevel.graph}:
            logging.error(f"'{self.__class__.__name__}' only supports "
                          f"node-level or graph-level explanations "
                          f"got (`task_level={task_level.value}`)")
            return False

        node_mask_type = self.explainer_config.node_mask_type
        if node_mask_type is not None:
            logging.error(f"'{self.__class__.__name__}' does not support "
                          f"explaining input node features "
                          f"got (`node_mask_type={node_mask_type.value}`)")
            return False

        return True

    def build_edge_features(self, adj, features, consider_direction=True, include_self_loop=True):
    
        src = adj._indices()[0]  
        dst = adj._indices()[1]
        
        if not include_self_loop:
            mask = (src != dst)
            src = src[mask]
            dst = dst[mask]
        
        if not consider_direction:
            mask = src < dst
            src = src[mask]
            dst = dst[mask]
            edge_index = torch.stack([torch.cat([src, dst]), 
                                    torch.cat([dst, src])], dim=0)
        else:
            edge_index = torch.stack([src, dst], dim=0)

        src_features = features[edge_index[0]]  # [E, D]
        dst_features = features[edge_index[1]]  # [E, D]
        edge_features = torch.cat([src_features, dst_features], dim=1)  # [E, 2D]
        
        return edge_index, edge_features

    def _get_temperature(self, epoch: int) -> float:
        temp = self.coeffs['temp']
        return temp[0] * pow(temp[1] / temp[0], epoch / self.epochs)

    def _concrete_sample(self, logits: Tensor,
                         temperature: float = 1.0) -> Tensor:
        bias = self.coeffs['bias']
        eps = (1 - 2 * bias) * torch.rand_like(logits) + bias
        return (eps.log() - (1 - eps).log() + logits) / temperature
    
    def gumbel_softmax(logits: Tensor, temperature=1.0, hard=False):
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
        y = (logits + gumbel_noise) / temperature
        y = torch.softmax(y, dim=-1)
        if hard:
            y_hard = torch.eye(y.size(-1), device=y.device)[y.argmax(dim=-1)]
            y = y_hard - y.detach() + y
        return y

    def _loss(self, y_hat: Tensor, y: Tensor, edge_mask: Tensor) -> Tensor:
        if self.model_config.mode == ModelMode.binary_classification:
            loss = self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y)

        # Regularization loss:
        mask = edge_mask.sigmoid()
        size_loss = mask.sum() * self.coeffs['edge_size']
        mask = 0.99 * mask + 0.005
        mask_ent = -mask * mask.log() - (1 - mask) * (1 - mask).log()
        mask_ent_loss = mask_ent.mean() * self.coeffs['edge_ent']

        return loss + size_loss + mask_ent_loss
    
    def _regularization_loss(self, mask: Tensor) -> Tensor:
        size_loss = mask.mean() * self.coeffs['edge_size']
        mask = 0.99 * mask + 0.005
        entropy = -mask * mask.log() - (1 - mask) * (1 - mask).log()
        return size_loss + entropy.mean() * self.coeffs['edge_ent']
