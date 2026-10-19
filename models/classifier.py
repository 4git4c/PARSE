from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.nn as nn
from models.gcn import GCN
from deeprobust.graph.utils import accuracy
from tqdm import tqdm
import faiss
import numpy as np

class Classifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, feature_dim, classes,
                 features, edge_index, plabels, groundtruth, edge_weight, explain_features, explain_weight, 
                 bias=True, self_loop=True, relu=True, dropout=0.5, results_dir=None):
        super().__init__()
        self.model = GCN(feature_dim, hidden_channels, feature_dim, classes, bias, self_loop, relu, dropout)
        self.model.reset_parameters()
        self.hidden = hidden_channels
        self.results_dir = results_dir
        
        self.features = explain_features
        self.edge_index = edge_index
        self.edge_weight = explain_weight
        self.explain_features = explain_features
        self.explain_weight = explain_weight
        
        self.plabels = plabels
        self.groundtruth = groundtruth
    
    def forward(self, index=None):
        logits = self.model(self.features, self.edge_index, self.edge_weight)
        if index is not None:
            return logits[index]
        return logits
    
    def train(self, epochs, idx_train, idx_val, idx_test, lr, wd, momentum=0.99, device='cpu', alpha=0.9, xi=1.0):
        self.features = self.features.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_weight = self.edge_weight.to(device)
        self.explain_features = self.explain_features.to(device)
        self.explain_weight = self.explain_weight.to(device)
        self.plabels = self.plabels.to(device)
        self.groundtruth = self.groundtruth.to(device)
        self.model = self.model.to(device)
        idx_train = torch.from_numpy(idx_train).to(device)
        idx_val = torch.from_numpy(idx_val).to(device)
        idx_test = torch.from_numpy(idx_test).to(device)
        
        assert torch.isnan(self.features).sum() == 0
        assert torch.isinf(self.features).sum() == 0
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999))
        
        nn_ind = mine_nearest_neighbors(self.explain_features, self.edge_index, device, 20)
        nn_ind = torch.from_numpy(nn_ind).to(device)
        nn_loss = SCANLoss(entropy_weight=1.0)
        
        acc_val_best = 0
        pconf = self.plabels[idx_train]
        denominator = pconf.sum(dim=1).unsqueeze(1) + 1e-10
        nconf_train = pconf / denominator
        
        for epoch in tqdm(range(epochs)):
            self.model.train()
            with torch.no_grad():
                rand_idx = torch.randint(low=0, high=nn_ind.shape[1], size=(nn_ind.shape[0],), 
                                        device=nn_ind.device, dtype=torch.int64)
                neighbor_indices = nn_ind[torch.arange(nn_ind.shape[0], device=nn_ind.device), rand_idx]
            
            logits, emb = self.model(self.features, self.edge_index, self.explain_weight, return_features=True)
            loss_c = pce_loss(nconf_train, F.log_softmax(logits, dim=1)[idx_train])
            loss = loss_c + xi * global_consistency_loss(logits, neighbor_indices)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            self.model.eval()
            with torch.no_grad():
                acc_val = self.acc_eval(idx_val)
                if acc_val > acc_val_best:
                    acc_val_best = acc_val
                    acc_test = self.acc_eval(idx_test)
                    best_ep = epoch
                    best_weights = deepcopy(self.model.state_dict())
                    
                logits_E = self.model(self.features, self.edge_index, self.explain_weight)
                pconf = self.plabels[idx_train] * logits_E.softmax(dim=1)[idx_train]
                denominator = pconf.sum(dim=1).unsqueeze(1) + 1e-10
                nconf_train = alpha * nconf_train + (1 - alpha) * pconf / denominator
                
            string = "epoch: {:d}, best_epoch: {:d}, best_acc_val: {:.2f}, acc_test: {:.2f}".format(
                epoch, best_ep, (acc_val_best * 100).item(), (acc_test * 100).item())
        tqdm.write(string)
        self.model.load_state_dict(best_weights)
        if self.results_dir is not None:
            torch.save(self.model.state_dict(), f'{self.results_dir}/cls.pth')
        return acc_test, acc_val_best
        
    def label_smoothing(self, confidence, index, nn_ind):
        confidence = confidence.softmax(dim=1)
        confidence[index] = confidence[index] * self.plabels[index]
        confidence[index] = confidence[index] / confidence[index].sum(dim=1).unsqueeze(1)
        
        weights = F.cosine_similarity(self.explain_features.unsqueeze(1), self.explain_features[nn_ind], dim=2)
        weights_c = weights.mean(dim=1, keepdim=True)
        weights = weights.softmax(dim=1)
        
        neighbor_probs = (weights.unsqueeze(-1) * confidence[nn_ind]).sum(dim=1)
        
        pconf = (confidence + weights_c * neighbor_probs) / (1 + weights_c)
        pconf[index] = pconf[index] * self.plabels[index]
        denominator = pconf.sum(dim=1).unsqueeze(1) + 1e-10
        pconf = pconf / denominator
        return pconf.detach()
    
    def confidence_psuedo(self, confidence, idx_train):
        pconf = confidence
        pconf[idx_train] = confidence[idx_train] * self.plabels[idx_train]
        denominator = pconf[idx_train].sum(dim=1).unsqueeze(1) + 1e-10
        pconf[idx_train] = pconf[idx_train] / denominator
        return pconf.detach()
    
    @torch.no_grad()
    def acc_eval(self, index):
        logits = self.model(self.features, self.edge_index, self.edge_weight)
        return accuracy(logits[index], self.groundtruth[index])
    
    def _contrastive_loss(self, pred, target, temperature=0.5):
        batch_size = pred.size(0)
        pred = F.normalize(pred, dim=-1, p=2)  # [B, D]
        target = F.normalize(target, dim=-1, p=2)  # [B, D]
        
        similarity = torch.einsum('id,jd->ij', pred, target)
        similarity /= temperature
        
        mask = torch.eye(batch_size, dtype=torch.float32, device=pred.device)
        
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()  
        
        exp_logits = torch.exp(logits) 
        logits_mask = (1.0 - torch.eye(batch_size, device=pred.device))
        exp_logits = exp_logits * logits_mask
        
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        loss = - (mask * log_prob).sum(dim=1)
        return loss.mean()
    
def structure_aggregate(probs, edge_index, edge_weight=None):
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), 
                                device=probs.device,
                                dtype=probs.dtype)
    assert edge_weight.dim() == 1
    assert edge_weight.size(0) == edge_index.size(1)
    
    src_nodes = edge_index[0]
    weighted_probs = probs[src_nodes] * edge_weight.unsqueeze(-1)
    target_nodes = edge_index[1]
    summed = torch.zeros_like(probs).scatter_add_(
        0, 
        target_nodes.unsqueeze(-1).expand(-1, probs.size(1)),
        weighted_probs
    )
    norm = torch.zeros(probs.size(0), 
                      device=probs.device,
                      dtype=probs.dtype).scatter_add_(
                          0, 
                          target_nodes, 
                          edge_weight
                      ).clamp_min(1e-8).unsqueeze(-1)
    return summed / norm

def sample_edges(edge_weights):
    n_edges = edge_weights.size(0)
    k = n_edges // 2
    
    eps = 1e-10
    adjusted_weights = edge_weights + eps
    
    u = torch.rand_like(adjusted_weights)
    keys = -torch.log(u) / adjusted_weights
    
    _, top_indices = torch.topk(keys, k, sorted=False)
    
    mask = torch.zeros_like(edge_weights, dtype=torch.bool)
    mask[top_indices] = 1
    return mask


def pce_loss(conf, log_probs):
    loss = conf * log_probs
    return -((loss).sum(dim=1)).mean()

def momentum_model(model_tea, model_stu, momentum=0.5):
    for param_tea, param_stu in zip(model_tea.parameters(), model_stu.parameters()):
        param_tea.data = param_tea.data * momentum + param_stu.data * (1 - momentum)

def mine_nearest_neighbors(features, edge_index, device, topk=20):
    features_np = features.cpu().numpy()
    n_nodes = features.shape[0]
    in_degree = torch.bincount(edge_index[1], minlength=n_nodes).cpu().numpy()
    
    dynamic_ks = (in_degree + topk).clip(max=n_nodes-1)
    max_k = int(np.max(dynamic_ks)) + 1 
    max_k = max_k if max_k <= 2048 else (2048 - topk)
    
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatIP(features.shape[1])
    gpu_index = faiss.index_cpu_to_gpu(res, torch.device(device).index, index)
    gpu_index.add(features_np)
    
    distances, indices = gpu_index.search(features_np, max_k)
    
    mask = (indices != np.arange(n_nodes)[:, None]) 
    
    result = np.full((n_nodes, topk + np.max(in_degree)), -1, dtype=np.int64)
    
    for i in range(n_nodes):
        valid_neighbors = indices[i][mask[i]][:dynamic_ks[i]]
        result[i, :len(valid_neighbors)] = valid_neighbors
        
    return result[:, :topk]

def entropy(x, input_as_probabilities):
    """ 
    Helper function to compute the entropy over the batch 

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ =  torch.clamp(x, min = 1e-10)
        b =  x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim = 1) * F.log_softmax(x, dim = 1)

    if len(b.size()) == 2: # Sample-wise entropy
        return -b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))

def global_consistency_loss(logits, dst):
    probs = F.softmax(logits, dim=1)
    return F.mse_loss(probs, probs[dst])

class SCANLoss(nn.Module):
    def __init__(self, entropy_weight=2.0):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight 
        
    def forward(self, anchors, neighbors):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)
       
        # Similarity in output space
        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)
        
        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities = True)

        # Total loss
        total_loss = consistency_loss - self.entropy_weight * entropy_loss
        
        return total_loss, consistency_loss, entropy_loss
    