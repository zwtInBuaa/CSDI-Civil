import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input, adj):
        """

        :param input: *xKxC
        :param adj: KxK
        :return: *xKxC
        """
        support = self.linear(input)  # *xKxC_out
        output = torch.einsum('kj,bjc->bkc', adj, support)  # *xKxC_out
        if self.bias is not None:
            return output + self.bias
        else:
            return output







class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(input_size, hidden_size)
        self.gc2 = GraphConvolution(hidden_size, output_size)
        self.dropout = dropout


    def forward(self, x, adj):
        adj = GCN.normalize(adj)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

    @staticmethod
    def normalize(adj):
        """
        :param adj: KxK
        :return: KxK
        """
        adj = adj + torch.eye(adj.size(0)).to(adj.device)
        rowsum = adj.sum(1)
        r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
        return torch.mm(torch.mm(r_mat_inv_sqrt, adj), r_mat_inv_sqrt)


if __name__ == '__main__':
    model = GCN(input_size=5, hidden_size=10, output_size=10)


    # input is BxKxC
    # adj is KxK
    B = 10
    K = 36
    C = 5
    input = torch.randn(B, K, C)
    # adj is a symmetric matrix, with zero and one
    adj = torch.randint(0, 2, (K, K)).float()
    adj = adj + adj.t()
    adj[adj > 1] = 1

    output = model(input, adj)
    print(output.shape)

