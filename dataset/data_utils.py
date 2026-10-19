import os
import random
import torch
from torch_geometric.utils import to_undirected
import os.path as osp
import numpy as np
from torch_geometric.datasets import Planetoid, Amazon
from ogb.nodeproppred import PygNodePropPredDataset
from scipy.sparse import coo_matrix, csr_matrix
from data_unit.utils import mkdir_if_missing, row_normalize, sparse_mx_to_torch_sparse_tensor,normalize_graph
from scipy.sparse import coo_matrix, lil_matrix, eye
import concurrent.futures
from dataset.plcifar10 import PLCIFAR10GraphDataset

def multiply_block(block, adj_matrix, k):
    result = block
    for i in range(k):
        result = result.dot(adj_matrix)
    return result.tocoo()


def power_adj_matrix_multi_process(adj_matrix, k, num_threads=16):
    identity_matrix = eye(adj_matrix.shape[0], dtype=adj_matrix.dtype, format='lil')
    identity_matrix = lil_matrix(identity_matrix)
    adj_matrix = lil_matrix(adj_matrix)

    block_size = adj_matrix.shape[0] // num_threads
    blocks = []
    for i in range(num_threads):
        start = i * block_size
        end = (i + 1) * block_size if i < num_threads - 1 else adj_matrix.shape[0]
        block = identity_matrix[start:end, :]
        blocks.append(block)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(multiply_block, block, adj_matrix, k) for block in blocks]
        results = [future.result() for future in futures]

    all_data = []
    all_row = []
    all_col = []
    
    for i, result in enumerate(results):
        start = i * block_size
        all_data.append(result.data)
        all_row.append(result.row + start)
        all_col.append(result.col)

    all_data = np.concatenate(all_data)
    all_row = np.concatenate(all_row)
    all_col = np.concatenate(all_col)

    final_result = coo_matrix((all_data, (all_row, all_col)), shape=adj_matrix.shape)
    final_result.data = (final_result.data > 0).astype(np.uint8)
    
    return final_result.toarray()


def power_adj_matrix(adj_matrix, k):
    result = eye(adj_matrix.shape[0], dtype=adj_matrix.dtype, format='coo')
    for _ in range(k):
        result = result.dot(adj_matrix)
    result.data = (result.data > 0).astype(np.uint8)
    return result.toarray()

def generate_neighbor_pl(train_labels, adj_matrix, k=1, dataset=None):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1
        
    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]
    if dataset is not None and dataset=='ogbn-arxiv':
        k_hop_adj = power_adj_matrix_multi_process(adj_matrix, k)
    else:
        k_hop_adj = power_adj_matrix(adj_matrix, k)
    neighbor_mask = torch.from_numpy(k_hop_adj).bool()
    train_labels = train_labels.to(torch.uint8)
    neighbor_labels = train_labels.repeat(n, 1)[neighbor_mask].to(torch.int64)
    node_indices = torch.arange(n).unsqueeze(1).expand(-1, n)[neighbor_mask]
    partialY = torch.zeros(n, K)
    partialY[node_indices, neighbor_labels] = 1
    partialY[torch.arange(n), train_labels.to(torch.int64)] = 1
    return partialY

def generate_candidate_labels(train_labels, dataset, partial_rate=None, k=None, adj_matrix=None):
    if dataset == 'plcifar10':
        return train_labels
    if partial_rate is not None:
        plabels = generate_uniform_candidate_labels(train_labels, partial_rate=partial_rate)
    elif k is not None:
        if adj_matrix is None:
            raise ValueError("adj_matrix must be provided when k is specified.")
        plabels = generate_neighbor_pl(train_labels, adj_matrix, k=k, dataset=dataset)
    else:
        raise ValueError("Either partial_rate or k must be specified.")
    assert torch.all(plabels[torch.arange(len(train_labels)), train_labels] == 1)
    return plabels

# https://github.com/hbzju/PiCO/tree/main
def generate_uniform_candidate_labels(train_labels, partial_rate=0.1):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    transition_matrix =  np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))] = partial_rate

    random_n = np.random.uniform(0, 1, size=(n, K))
    
    for j in range(n):  # for each instance
        partialY[j, :] = torch.from_numpy((random_n[j, :] < transition_matrix[train_labels[j], :]) * 1)

    partialY[torch.arange(n), train_labels] = 1.0
    return partialY


def generate_neighbor_candidate_labels(train_labels, adj_matrix, num_candidates_per_instance, max_hop=3, dataset=None):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0

    hop_adj_matrices = []
    current_adj = adj_matrix
    for _ in range(max_hop):
        hop_adj_matrices.append(current_adj)
        current_adj = np.dot(current_adj, adj_matrix)

    num_candidates_per_instance = np.array(num_candidates_per_instance, dtype=int)  # 确保为整数类型
    sampled_indices = []

    for i in range(n):
        num_needed = num_candidates_per_instance[i] - 1
        collected_labels = set()
        for hop in range(max_hop):
            rows, cols = np.nonzero(hop_adj_matrices[hop][i])
            neighbor_labels = train_labels[cols].tolist()
            for label in neighbor_labels:
                if len(collected_labels) < num_needed:
                    collected_labels.add(label) if label != train_labels[i].item() else None
                else:
                    break
            if len(collected_labels) >= num_needed:
                break

        remaining = num_needed - len(collected_labels)
        if remaining > 0:
            all_labels = set(range(K)) - {train_labels[i].item()}
            available_labels = list(all_labels - collected_labels)
            random_sampled = np.random.choice(available_labels, remaining, replace=False)
            for label in random_sampled:
                collected_labels.add(label)

        collected_labels.add(train_labels[i].item())
        sampled = torch.tensor(list(collected_labels))
        sampled_indices.append(sampled)

    
    padded_sampled_indices = torch.nn.utils.rnn.pad_sequence(sampled_indices, batch_first=True)
    
    row_indices = []
    col_indices = []
    for i in range(n):
        valid_length = num_candidates_per_instance[i]
        row_indices.extend([i] * valid_length)
        col_indices.extend(padded_sampled_indices[i][:valid_length].tolist())

    row_indices = torch.tensor(row_indices)
    col_indices = torch.tensor(col_indices)
    partialY[row_indices, col_indices] = 1
    partialY[torch.arange(n), train_labels] = 1
    return partialY


def train_val_test_split(num_samples, train_ratio, val_ratio):
    assert train_ratio >= 0 and val_ratio >= 0 and (train_ratio + val_ratio) < 1
    
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)

    indices = np.random.permutation(num_samples)

    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]

    return train_indices, val_indices, test_indices

def get_dataset(args, data_root, device):
    if args.dataset_name in ['Cora', 'CiteSeer', 'PubMed', 'Photo', 'Computers']:
        if args.dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
            path = osp.join(osp.dirname(osp.realpath(__file__)), f'{data_root}/{args.dataset_name}')
            dataset = Planetoid(path, args.dataset_name)
        elif args.dataset_name in ['Photo', 'Computers']:
            path = osp.join(osp.dirname(osp.realpath(__file__)), f'{data_root}/{args.dataset_name}')
            dataset = Amazon(path, args.dataset_name, pre_transform=None) 
        data = dataset[0]
        
        i = torch.LongTensor(np.array([np.array(data.edge_index[0]), np.array(data.edge_index[1])]))
        v = torch.FloatTensor(torch.ones([data.num_edges]))
        A_sp = torch.sparse.FloatTensor(i, v, torch.Size([data.num_nodes, data.num_nodes]))
        A = A_sp.to_dense()
        I = torch.eye(A.shape[1]).to(A.device)
        A_I = A + I
        
        A_I_normal = normalize_graph(A_I)
        A_I_normal = A_I_normal.to_sparse()
        A_I = A_I.to_sparse()

        label = data.y
        nb_feature = data.num_features
        nb_classes = int(label.max() - label.min()) + 1
        nb_nodes = data.num_nodes
        data.x = torch.FloatTensor(data.x)
        eps = 2.2204e-16
        norm = data.x.norm(p=1, dim=1, keepdim=True).clamp(min=0.) + eps
        data.x = data.x.div(norm.expand_as(data.x))
        adj = csr_matrix(
            (np.ones(data.num_edges), (data.edge_index[0].numpy(), data.edge_index[1].numpy())),
            shape=(data.num_nodes, data.num_nodes))
    elif args.dataset_name == "plcifar10":
        path = osp.join(osp.dirname(osp.realpath(__file__)), f'{data_root}/{args.dataset_name}')
        save_root = osp.join(path, 'plcifar10_cache')
        dataset = PLCIFAR10GraphDataset(data_root=path, save_root=save_root, k=5, device=device)
        return dataset.get_data()
    else:
        path = osp.join(osp.dirname(osp.realpath(__file__)), f'{data_root}/{args.dataset_name}')
        dataset = PygNodePropPredDataset(name=args.dataset_name, root=path)
        if args.dataset_name in ['ogbn-arxiv']:  
            data = dataset[0]
            data.edge_index = to_undirected(edge_index=data.edge_index, num_nodes=data.num_nodes)
                
        data.x = torch.FloatTensor(data.x)
        eps = 2.2204e-16
        norm = data.x.norm(p=1, dim=1, keepdim=True).clamp(min=0.) + eps
        data.x = data.x.div(norm.expand_as(data.x))
        adj = coo_matrix(
            (np.ones(data.num_edges), (data.edge_index[0].numpy(), data.edge_index[1].numpy())),
            shape=(data.num_nodes, data.num_nodes))
        nb_nodes = data.num_nodes
        I = coo_matrix((np.ones(nb_nodes), (np.arange(0, nb_nodes, 1), np.arange(0, nb_nodes, 1))),
                       shape=(nb_nodes, nb_nodes))
        adj_I = adj + I  # coo_matrix(sp.eye(adj.shape[0]))
        A_I = sparse_mx_to_torch_sparse_tensor(adj_I)
        A_I_normal = row_normalize(adj_I)
        A_I_normal = sparse_mx_to_torch_sparse_tensor(A_I_normal).float()
        label = data.y
        nb_feature = data.num_features
        nb_classes = int(label.max() - label.min()) + 1

    return data, [A_I_normal, adj, A_I], [data.x], [label.flatten(), nb_feature, nb_classes, nb_nodes]

def init_seed(seed, benchmark=True, deterministic=True):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    torch.manual_seed(seed)
