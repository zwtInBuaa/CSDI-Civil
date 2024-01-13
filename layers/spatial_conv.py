import torch
from torch import nn

epsilon = 1e-10


class SpatialConvOrderK(nn.Module):
    """
    Spatial convolution of order K with possibly different diffusion matrices (useful for directed graphs)

    Efficient implementation inspired from graph-wavenet codebase
    """

    def __init__(self, c_in, c_out, order=1, include_self=True):
        super(SpatialConvOrderK, self).__init__()
        self.include_self = include_self
        c_in = (order + 1) * c_in if include_self else order * c_in
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.order = order

    @staticmethod
    def compute_support(adj, device=None):
        if device is not None:
            adj = adj.to(device)

        adj_normalized = adj / (adj.sum(1, keepdims=True) + epsilon)  # [nodes, nodes]

        support = [adj_normalized]
        return support

    @staticmethod
    def compute_support_orderK(adj, k, include_self=True, device=None):
        if isinstance(adj, (list, tuple)):
            support = adj
        else:
            support = SpatialConvOrderK.compute_support(adj, device)
        supp_k = []
        for a in support:
            ak = a
            for i in range(k - 1):
                ak = torch.matmul(ak, a.T)
                if not include_self:
                    ak.fill_diagonal_(0.)
                supp_k.append(ak)
        return support + supp_k

    def forward(self, x, support):
        # [batch, features, nodes, steps]
        if x.dim() < 4:
            squeeze = True
            x = torch.unsqueeze(x, -1)
        else:
            squeeze = False
        out = [x] if self.include_self else []
        if (type(support) is not list):
            support = [support]
        for a in support:
            x1 = torch.einsum('ncvl,wv->ncwl', (x, a)).contiguous()
            out.append(x1)

        out = torch.cat(out, dim=1)
        out = self.mlp(out)
        if squeeze:
            out = out.squeeze(-1)
        return out


class SpatialDiffusionConv(nn.Module):
    """
        Spatial convolution of order K with possibly different diffusion matrices (useful for directed graphs)

        Efficient implementation inspired from graph-wavenet codebase
        """

    def __init__(self, c_in, c_out, adj, order=1, include_self=True):
        super().__init__()
        input_size = (order + 1) * c_in if include_self else order * c_in
        self.mlp = nn.ModuleList([nn.Conv1d(input_size, c_out, kernel_size=1),
                                  nn.ReLU(),
                                  nn.Conv1d(c_out,c_in, kernel_size=1)])


        self.support = self.compute_support_orderK(adj, order, include_self)
        self.include_self = include_self


    def compute_support_orderK(self, adj, k, include_self=True):
        adj_normalized = adj / (adj.sum(1, keepdims=True) + epsilon)  # [nodes, nodes]
        support = [adj_normalized]
        supp_k = []
        for a in support:
            ak = a
            for i in range(k - 1):
                ak = torch.matmul(ak, a.T)
                if not include_self:
                    ak.fill_diagonal_(0.)
                supp_k.append(ak)
        return support + supp_k

    def forward(self, x):
        # input x is of shape [spatial_dim, batch, hidden_dim]

        # reshape x to [batch, hidden_dim, spatial_dim]
        x = x.permute(1, 2, 0)

        out = [x] if self.include_self else []

        for a in self.support:
            x1 = torch.einsum('ncv,wv->ncw', (x, a)).contiguous()
            out.append(x1)

        out = torch.cat(out, dim=1)
        for layer in self.mlp:
            out = layer(out)

        # reshape out to [spatial_dim, batch, hidden_dim]
        out = out.permute(2, 0, 1)

        return out
