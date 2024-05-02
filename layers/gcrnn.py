import torch
import torch.nn as nn

from .spatial_conv import SpatialConvOrderK


class GCGRUCell(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """

    def __init__(self, d_in, num_units, order, activation='tanh'):
        """
        :param num_units: the hidden dim of rnn
        :param support_len: the (weighted) adjacency matrix of the graph, in numpy ndarray form
        :param order: the max diffusion step
        :param activation: if None, don't do activation for cell state
        """
        super(GCGRUCell, self).__init__()
        self.activation_fn = getattr(torch, activation)

        self.forget_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units,
                                             order=order)
        self.update_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units,
                                             order=order)
        self.c_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, order=order)

    def forward(self, x, h, adj):
        """
        :param x: (B, input_dim, num_nodes)
        :param h: (B, num_units, num_nodes)
        :param adj: (num_nodes, num_nodes)
        :return:
        """
        # we start with bias 1.0 to not reset and not update
        x_gates = torch.cat([x, h], dim=1)
        r = torch.sigmoid(self.forget_gate(x_gates, adj))
        u = torch.sigmoid(self.update_gate(x_gates, adj))
        x_c = torch.cat([x, r * h], dim=1)
        c = self.c_gate(x_c, adj)  # batch_size, self._num_nodes * output_size
        c = self.activation_fn(c)
        return u * h + (1. - u) * c
