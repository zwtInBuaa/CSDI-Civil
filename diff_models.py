import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from einops import rearrange
from layers.S4Layer import S4, S4Layer, LinearActivation


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size=1, dilation=1, stride=1):
    padding = dilation * (kernel_size - 1) // 2
    layer = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding, stride=stride)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation,
                              padding=self.padding,
                              stride=stride)

        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out


class DownPool(nn.Module):
    def __init__(self, d_input, expand, pool):
        super().__init__()
        self.d_output = d_input * expand
        self.pool = pool

        self.linear = LinearActivation(
            d_input * pool,
            self.d_output,
            transposed=True,
            weight_norm=True,
        )

    def forward(self, x):
        x = rearrange(x, '... h (l s) -> ... (h s) l', s=self.pool)
        x = self.linear(x)
        return x


class UpPool(nn.Module):
    def __init__(self, d_input, expand, pool, causal=True):
        super().__init__()
        self.d_output = d_input // expand
        self.pool = pool
        self.causal = causal

        self.linear = LinearActivation(
            d_input,
            self.d_output * pool,
            transposed=True,
            weight_norm=True,
        )

    def forward(self, x):
        x = self.linear(x)

        if (self.causal):
            x = F.pad(x[..., :-1], (1, 0))  # Shift to ensure causality
        x = rearrange(x, '... (h s) l -> ... h (l s)', s=self.pool)

        return x


class FFBlock(nn.Module):

    def __init__(self, d_model, expand=2, dropout=0.0):
        """
        Feed-forward block.

        Args:
            d_model: dimension of input
            expand: expansion factor for inverted bottleneck
            dropout: dropout rate
        """
        super().__init__()

        input_linear = LinearActivation(
            d_model,
            d_model * expand,
            transposed=True,
            activation='gelu',
            activate=True,
        )
        dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        output_linear = LinearActivation(
            d_model * expand,
            d_model,
            transposed=True,
            activation=None,
            activate=False,
        )

        self.ff = nn.Sequential(
            input_linear,
            dropout,
            output_linear,
        )

    def forward(self, x):
        return self.ff(x), None


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_CSDI(nn.Module):

    def __init__(
            self,
            config,
            inputdim=2,
            pool=[2, 2],
            expand=2,
            ff=2,
            glu=True,
            unet=True,
            dropout=0.0,
            bidirectional=True,
            s4_lmax=1,
            s4_d_state=64,
            s4_dropout=0.0,
            s4_bidirectional=True,
    ):
        super().__init__()

        self.channels = H = config["channels"]
        self.unet = unet

        self.n_layers = config["layers"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)

        # nn.init.zeros_(self.output_projection2.weight)

        def s4_block(dim, stride):
            layer = S4(
                d_model=dim,
                l_max=s4_lmax,
                d_state=s4_d_state,
                bidirectional=s4_bidirectional,
                postact='glu' if glu else None,
                dropout=dropout,
                transposed=True,
                # hurwitz=True, # use the Hurwitz parameterization for stability
                # tie_state=True, # tie SSM parameters across d_state in the S4 layer
                trainable={
                    'dt': True,
                    'A': True,
                    'P': True,
                    'B': True,
                },  # train all internal S4 parameters
            )

            # def __init__(self, side_dim, channels, layer, dropout, diffusion_embedding_dim, nheads):

            return ResidualBlock(
                side_dim=config["side_dim"],
                channels=self.channels,
                diffusion_embedding_dim=config["diffusion_embedding_dim"],
                layer=layer,
                dropout=dropout,
                nheads=config["nheads"],
                stride=stride
            )

        def ff_block(dim, stride):
            layer = FFBlock(
                d_model=dim,
                expand=ff,
                dropout=dropout,
            )
            return ResidualBlock(
                side_dim=config["side_dim"],
                channels=self.channels,
                diffusion_embedding_dim=config["diffusion_embedding_dim"],
                layer=layer,
                dropout=dropout,
                nheads=config["nheads"],
                stride=stride
            )

        # Down blocks
        d_layers = []
        for i, p in enumerate(pool):
            if unet:
                # Add blocks in the down layers
                for _ in range(self.n_layers):
                    if i == 0:
                        d_layers.append(s4_block(H, 1))
                        if ff > 0:
                            d_layers.append(ff_block(H, 1))
                    elif i == 1:
                        d_layers.append(s4_block(H, p))
                        if ff > 0:
                            d_layers.append(ff_block(H, p))
            # Add sequence downsampling and feature expanding
            d_layers.append(DownPool(H, expand, p))
            H *= expand

            # Center block
        c_layers = []
        for _ in range(self.n_layers):
            c_layers.append(s4_block(H, pool[1] * 2))
            if ff > 0:
                c_layers.append(ff_block(H, pool[1] * 2))

        # Up blocks
        u_layers = []
        for i, p in enumerate(pool[::-1]):
            block = []
            H //= expand
            block.append(UpPool(H * expand, expand, p, causal=not bidirectional))

            for _ in range(self.n_layers):
                if i == 0:
                    block.append(s4_block(H, pool[0]))
                    if ff > 0:
                        block.append(ff_block(H, pool[0]))
                elif i == 1:
                    block.append(s4_block(H, 1))
                    if ff > 0:
                        block.append(ff_block(H, 1))
            u_layers.append(nn.ModuleList(block))

        self.d_layers = nn.ModuleList(d_layers)
        self.c_layers = nn.ModuleList(c_layers)
        self.u_layers = nn.ModuleList(u_layers)

        self.norm = nn.LayerNorm(H)

    def forward(self, x, cond_info, diffusion_step):
        B, input_dim, K, L = x.shape
        x = x.reshape(B, input_dim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)
        base_shape = x.shape
        x = x.reshape(B, self.channels, K * L)

        print("init x: ", x.shape)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        # down blocks
        outputs = [x]
        i = 0
        for layer in self.d_layers:
            if isinstance(layer, ResidualBlock):
                x = layer(x, base_shape, cond_info, diffusion_emb)
            else:
                x = layer(x)

            tmp = 1 if x.shape[1] > base_shape[1] else 2
            base_shape = (B, x.shape[1], K, x.shape / K / tmp)

            outputs.append(x)
            print("%d-th d_layers x: " % i, x.shape)
            i = i + 1

        # center block
        for layer in self.c_layers:
            if isinstance(layer, ResidualBlock):
                x = layer(x, base_shape, cond_info, diffusion_emb)
            else:
                x = layer(x)
            tmp = 1 if x.shape[1] > base_shape[1] else 2
            base_shape = (B, x.shape[1], K, x.shape / K / tmp)
            print("c_layers x: ", x.shape)
        x = x + outputs.pop()  # add a skip connection to the last output of the down block

        # Up blocks
        for block in self.u_layers:
            if self.unet:
                for layer in block:
                    if isinstance(layer, ResidualBlock):
                        x = layer(x, base_shape, cond_info, diffusion_emb)
                    else:
                        x = layer(x)
                    tmp = 1 if x.shape[1] > base_shape[1] else 2
                    base_shape = (B, x.shape[1], K, x.shape / K / tmp)
                    print("u_layers x: ", x.shape)
                    x = x + outputs.pop()  # skip connection
            else:
                for layer in block:
                    if isinstance(layer, ResidualBlock):
                        x = layer(x, base_shape, cond_info, diffusion_emb)
                    else:
                        x = layer(x)
                    tmp = 1 if x.shape[1] > base_shape[1] else 2
                    base_shape = (B, x.shape[1], K, x.shape / K / tmp)
                    print("u_layers x: ", x.shape)
                    if isinstance(layer, UpPool):
                        # Before modeling layer in the block
                        x = x + outputs.pop()
                        outputs.append(x)
                x = x + outputs.pop()  # add a skip connection from the input of the modeling part of this up block

        # feature projection
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x

        # def forward(self, x, cond_info, diffusion_step):


#     B, inputdim, K, L = x.shape
#
#     x = x.reshape(B, inputdim, K * L)
#     x = self.input_projection(x)
#     x = F.relu(x)
#     x = x.reshape(B, self.channels, K, L)
#
#     diffusion_emb = self.diffusion_embedding(diffusion_step)
#
#     skip = []
#     for layer in self.residual_layers:
#         x, skip_connection = layer(x, cond_info, diffusion_emb)
#         skip.append(skip_connection)
#
#     x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
#     x = x.reshape(B, self.channels, K * L)
#     x = self.output_projection1(x)  # (B,channel,K*L)
#     x = F.relu(x)
#     x = self.output_projection2(x)  # (B,1,K*L)
#     x = x.reshape(B, K, L)
#     return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, layer, dropout, diffusion_embedding_dim, nheads, stride):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)

        self.cond_projection = Conv1d_with_init(side_dim, channels, kernel_size=stride, stride=stride)
        # self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        # self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        # self.time_layer = S4Layer(features=channels, lmax=100)
        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        # self.s4_layer = S4Layer(features=channels, lmax=100)

        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.s4_layer = layer

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, x_shape, cond_info, diffusion_emb):
        x = x.reshape(x_shape)
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        # Pre norm
        y = self.norm(y.transpose(-1, -2)).transpose(-1, -2)
        # print(y.shape)

        y, _ = self.s4_layer(y)

        y_time = self.forward_time(y, base_shape)
        y_feature = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = torch.sigmoid(y_time) * torch.tanh(y_feature)
        # y = self.mid_projection(y)  # (B,2*channel,K*L)
        # y = self.mid_projection(y)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        # cond_info = self.s4(cond_info)
        y = y + cond_info

        # Dropout on the output of the layer
        y = self.dropout(y)

        # Residual connection
        x = x + y

        x = x.reshape(B, channel, K * L)

        return x
