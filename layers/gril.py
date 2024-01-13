import torch
import torch.nn as nn
from einops import rearrange

from .spatial_conv import SpatialConvOrderK
from .gcrnn import GCGRUCell
from .spatial_attention import SpatialAttention
from ..utils.ops import reverse_tensor


class GRIL(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 n_nodes,
                 n_layers=1,
                 dropout=0.,
                 order=1,
                 layer_norm=False
                 ):
        super(GRIL, self).__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)

        self.n_layers = int(n_layers)
        self.order = int(order)
        rnn_input_size = self.input_size

        # Spatio-temporal encoder (rnn_input_size -> hidden_size)
        self.cells = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(self.n_layers):
            self.cells.append(GCGRUCell(d_in=rnn_input_size if i == 0 else self.hidden_size,
                                        num_units=self.hidden_size, order=self.order))
            if layer_norm:
                self.norms.append(nn.GroupNorm(num_groups=1, num_channels=self.hidden_size))
            else:
                self.norms.append(nn.Identity())
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        # Fist stage readout
        self.first_stage = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.input_size, kernel_size=1)


        # Hidden state initialization embedding
        self.h0 = self.init_hidden_states(n_nodes)
        self.supp = None




    def init_hidden_states(self, n_nodes):
        h0 = []
        for l in range(self.n_layers):
            std = 1. / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float))
            vals = torch.distributions.Normal(0, std).sample((self.hidden_size, n_nodes))
            h0.append(nn.Parameter(vals))
        return nn.ParameterList(h0)

    def get_h0(self, x):
        if self.h0 is not None:
            return [h.expand(x.shape[0], -1, -1) for h in self.h0]
        return [torch.zeros(size=(x.shape[0], self.hidden_size, x.shape[2])).to(x.device)] * self.n_layers

    def update_state(self, x, h, adj):
        rnn_in = x
        for layer, (cell, norm) in enumerate(zip(self.cells, self.norms)):
            rnn_in = h[layer] = norm(cell(rnn_in, h[layer], adj))
            if self.dropout is not None and layer < (self.n_layers - 1):
                rnn_in = self.dropout(rnn_in)
        return h

    def forward(self, x, adj=None, h=None, cached_support=True):
        # x:[batch, features, nodes, steps]
        *_, nodes, steps = x.size()

        if adj is None:
            adj = torch.eye(nodes, dtype=torch.bool, device=x.device)

        if cached_support and (self.supp is not None):
            supp = self.supp
        else:
            supp = SpatialConvOrderK.compute_support_orderK(adj, self.order, include_self=True, device=x.device)
            self.supp = supp if cached_support else None

        # init hidden state using node embedding or the empty state
        if h is None:
            h = self.get_h0(x)
        elif not isinstance(h, list):
            h = [*h]

        # Temporal conv
        imputations, states = [], []

        for step in range(steps):
            inputs = x[..., step]

            # update state with original sequence filled using imputations
            h = self.update_state(inputs, h, supp)
            h_s = h[-1]
            # First stage readout
            x_hat = self.first_stage(h_s)
            # store imputations and states
            imputations.append(x_hat)
            states.append(torch.stack(h, dim=0))


        # Aggregate outputs -> [batch, features, nodes, steps]
        imputations = torch.stack(imputations, dim=-1)

        states = torch.stack(states, dim=-1)


        return imputations, states


class BiGRIL(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 n_layers=1,
                 dropout=0.,
                 n_nodes=None,
                 order=1,
                 layer_norm=False,
                 merge='mean'
                 ):
        super(BiGRIL, self).__init__()
        self.fwd_rnn = GRIL(input_size=input_size,
                            hidden_size=hidden_size,
                            n_layers=n_layers,
                            dropout=dropout,
                            n_nodes=n_nodes,
                            order=order,
                            layer_norm=layer_norm)
        self.bwd_rnn = GRIL(input_size=input_size,
                            hidden_size=hidden_size,
                            n_layers=n_layers,
                            dropout=dropout,
                            n_nodes=n_nodes,
                            order=order,
                            layer_norm=layer_norm
                            )


        self._impute_from_states = False
        self.out = getattr(torch, merge)
        self.supp = None

    def forward(self, x, adj=None, cached_support=False):
        # Forward
        fwd_out, _ = self.fwd_rnn(x, adj, cached_support=cached_support)
        # Backward
        rev_x = reverse_tensor(x)
        bwd_res, _ = self.bwd_rnn(rev_x, adj, cached_support=cached_support)
        bwd_out = reverse_tensor(bwd_res)


        imputation = torch.stack([fwd_out, bwd_out], dim=1)
        imputation = self.out(imputation, dim=1)

        return imputation
