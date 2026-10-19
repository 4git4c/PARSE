import os
import pickle
import numpy as np
from scipy.sparse import coo_matrix

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from torchvision import transforms, models
from torchvision.datasets import CIFAR10

class PLCIFAR10GraphDataset:
    def __init__(self, data_root='./dataset/plcifar10', k=5, save_root='./dataset/plcifar10/plcifar10_cache', device='cpu'):
        self.data_root = data_root
        self.k = k
        self.save_root = save_root
        self.device = device
        os.makedirs(save_root, exist_ok=True)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        # Load full dataset (train + test)
        self.train_set = CIFAR10(root=data_root, train=True, download=True, transform=self.transform)
        self.test_set = CIFAR10(root=data_root, train=False, download=True, transform=self.transform)
        self.dataset = torch.utils.data.ConcatDataset([self.train_set, self.test_set])

        self.ground_truth = torch.tensor([label for _, label in self.train_set] + [label for _, label in self.test_set], dtype=torch.long)

        # Index split
        n_train = len(self.train_set)  # 50000
        n_test = len(self.test_set)    # 10000

        val_size = int(0.1 * n_train)  # 5000
        perm = torch.randperm(n_train)
        self.val_idx = perm[:val_size]
        self.train_idx = perm[val_size:]
        self.test_idx = torch.arange(n_train, n_train + n_test)  # 50000~59999

        self.partial_labels = self.load_partial_labels()
        self.features = self.load_or_extract_features()
        self.edge_index = self.load_or_build_knn_graph()

    def load_partial_labels(self):
        label_path = os.path.join(self.save_root, 'partial_labels_60000.npy')
        if os.path.exists(label_path):
            print("Loading cached partial labels...")
            return torch.tensor(np.load(label_path), dtype=torch.float)

        print("Generating partial labels from pickle and ground-truth...")

        # Load partial label from pkl (only for training set)
        pl_path = os.path.join(self.data_root, 'plcifar10.pkl')
        if not os.path.exists(pl_path):
            raise FileNotFoundError(f"Partial label file {pl_path} does not exist. Please download it from https://github.com/wwangwitsel/PLCIFAR10/tree/main.")
        with open(pl_path, 'rb') as f:
            partial_label_all = pickle.load(f)

        n_train = len(self.train_set)
        n_test = len(self.test_set)
        total = n_train + n_test

        partial_targets = np.zeros((total, 10))  # 60000 x 10

        for key, value in partial_label_all.items():
            for candidate_set in value:
                for label in candidate_set:
                    partial_targets[key, label] = 1

        for i in range(n_test):
            label = self.test_set[i][1]
            idx = n_train + i 
            partial_targets[idx, label] = 1

        np.save(label_path, partial_targets)
        return torch.tensor(partial_targets, dtype=torch.float)

    def load_or_extract_features(self):
        feat_path = os.path.join(self.save_root, 'features.npy')
        if os.path.exists(feat_path):
            print("Loading cached image features...")
            return torch.tensor(np.load(feat_path))

        print("Extracting image features with ResNet18...")
        loader = DataLoader(self.dataset, batch_size=128, shuffle=False, num_workers=4)
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Identity()
        model = model.to(self.device).eval()

        features = []
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(self.device)
                feats = model(imgs).cpu()
                features.append(feats)
        features = torch.cat(features, dim=0)
        np.save(feat_path, features.numpy())
        return features

    def load_or_build_knn_graph(self):
        edge_path = os.path.join(self.save_root, f'edge_index_full_k{self.k}.pt')
        if os.path.exists(edge_path):
            print("Loading cached kNN graph...")
            return torch.load(edge_path)

        print("Building kNN graph...")
        x = self.features
        k = self.k
        n_nodes = len(x)
        batch_size = 1000

        row_indices = []
        col_indices = []

        for i in range(0, n_nodes, batch_size):
            end = min(i + batch_size, n_nodes)
            dist = torch.norm(x[i:end, None] - x, dim=2)
            knn = dist.topk(k + 1, dim=1, largest=False).indices[:, 1:]
            rows = torch.arange(i, end).unsqueeze(1).repeat(1, k).flatten()
            row_indices.extend(rows.tolist())
            col_indices.extend(knn.flatten().tolist())

        edge_index = torch.tensor([row_indices, col_indices], dtype=torch.long)
        torch.save(edge_index, edge_path)
        return edge_index

    def _sparse_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def _normalize_adj(self, adj):
        adj = adj.tocoo()
        rowsum = np.array(adj.sum(1)).flatten()
        d_inv = 1. / (rowsum + 1e-10)
        d_mat_inv = coo_matrix((d_inv, (np.arange(len(d_inv)), np.arange(len(d_inv)))), shape=adj.shape)
        return d_mat_inv.dot(adj)

    def get_data(self):
        x = self.features
        y = self.partial_labels
        edge_index = self.edge_index

        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1]

        adj = coo_matrix((np.ones(num_edges), (edge_index[0].numpy(), edge_index[1].numpy())),
                         shape=(num_nodes, num_nodes))
        I = coo_matrix((np.ones(num_nodes), (np.arange(num_nodes), np.arange(num_nodes))),
                       shape=(num_nodes, num_nodes))
        adj_I = adj + I
        A_I = self._sparse_to_torch_sparse_tensor(adj_I)
        A_I_normal = self._normalize_adj(adj_I)
        A_I_normal = self._sparse_to_torch_sparse_tensor(A_I_normal).float()

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_idx = self.train_idx
        data.val_idx = self.val_idx
        data.test_idx = self.test_idx
        data.ground_truth = self.ground_truth

        nb_feature = x.shape[1]
        nb_classes = 10
        nb_nodes = num_nodes

        return data, [A_I_normal, adj, A_I], [x], [y, nb_feature, nb_classes, nb_nodes]
