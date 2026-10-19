import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, feature_dim, classes, bias=True, self_loop=True,
                 relu=True, dropout=0.5):
        super().__init__()
        self.gc1 = GCNConv(in_channels, hidden_channels, bias=bias, add_self_loops=self_loop)
        self.gc2 = GCNConv(hidden_channels, feature_dim, bias=bias, add_self_loops=self_loop)
        self.head = LogReg(feature_dim, classes)
        
        self.relu=relu
        self.dropout=dropout
        
    def forward(self, x, edge_index, edge_weight, return_features=False):
        identity = x
        x = self.gc1(x, edge_index, edge_weight)
        if self.relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        features = self.gc2(x, edge_index, edge_weight)
        features = features / features.norm(dim=1)[:, None]  
        features = features + identity
        logits = self.head(features)
        if return_features:
            return logits, features
        else:
            return logits

    def reset_parameters(self):
        """Initialize parameters of GCN."""
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()
