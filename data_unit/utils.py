import errno
import os
from scipy.sparse import diags
import numpy as np
import torch


def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize_graph(A):
    eps = 2.2204e-16
    if A.is_sparse:
        deg = torch.sparse.sum(A, dim=-1).to_dense()
        deg_inv_sqrt = (deg.clamp(min=0.) + eps).pow(-0.5)

        indices = torch.arange(A.size(0)).unsqueeze(0).repeat(2, 1)
        values = deg_inv_sqrt
        deg_inv_sqrt_sparse = torch.sparse_coo_tensor(indices, values, size=(A.size(0), A.size(0)), device=A.device)

        if A.size()[0] != A.size()[1]:
            A = torch.sparse.mm(deg_inv_sqrt_sparse, A)
            A = torch.sparse.mm(A, deg_inv_sqrt_sparse)
        else:
            A = torch.sparse.mm(deg_inv_sqrt_sparse, A)
            A = torch.sparse.mm(A, deg_inv_sqrt_sparse.t())
    else:
        deg_inv_sqrt = (A.sum(dim=-1).clamp(min=0.) + eps).pow(-0.5)
        if A.size()[0] != A.size()[1]:
            A = deg_inv_sqrt.unsqueeze(-1) * (deg_inv_sqrt.unsqueeze(-1) * A)
        else:
            A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
