import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from layers.S4Layer import S4Layer

from layers.longformer import LongformerTS
from layers.spatial_conv import SpatialDiffusionConv
from layers.bilstm import BiLSTM
# from layers.gril import BiGRIL
from layers.tcn import TemporalConvNet


# from pypots.imputation.transformer import EncoderLayer, PositionalEncoding


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def get_tcn(input_size, hidden_size=[64, 128, 64]):
    return TemporalConvNet(input_size, hidden_size)


def get_bilstm(channels, hidden_size=64, n_layers=1):
    return BiLSTM(input_size=channels, hidden_size=hidden_size, num_layers=n_layers)


def get_torch_trans(heads=8, layers=1, channels=64, hidden_size=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=hidden_size, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out


class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        return out


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
    def __init__(self, config, in_dim=72):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(in_dim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, in_dim, 1)
        nn.init.zeros_(self.output_projection2.weight)

        # self.conv2d_output_projection = nn.Conv2d(self.channels, 1, 1, stride=1)

        # self.cond_obs_s4 = S4Layer(72, lmax=100)
        # self.noise_target = S4Layer(72, lmax=100)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        x, cond_obs = x  # (B,K,L),(B,K,L)

        x = self.input_projection(x)
        B, C, L = x.shape
        x = F.relu(x)

        # print("y in base", x.shape)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_obs, cond_info, diffusion_emb)
            skip.append(skip_connection)
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))

        x = self.output_projection1(x)  # (B,channel,L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,K,L)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.diffusion_conv = Conv1d_with_init(channels, 2 * channels, 1)

        self.conv_layer = Conv(channels, 2 * channels, kernel_size=3)

        self.cond_conv = Conv(72 * 2, 2 * channels, kernel_size=1)

        self.time_conv = Conv(128, 2 * channels, kernel_size=1)

        # self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        # self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        # self.time_layer = S4Layer(features=channels, lmax=100)
        # self.time_layer = Conv1dBlock(channels, channels, L=100, num_blocks=3)

        # self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        # self.time_layer = get_bilstm(channels=channels, hidden_size=64)
        # self.time_layer = get_tcn(input_size=channels)

        # self.time_layer = get_longformerTS(heads=8, layers=1, channels=channels, hidden_size=64, attention_window=9)

        # self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        # self.feature_layer = get_bilstm(channels=channels, hidden_size=64)

        self.s4_init_layer = S4Layer(features=2 * channels, lmax=100)
        self.time_trans = get_torch_trans(heads=8, layers=1, channels=2 * channels, hidden_size=channels)
        self.s4_end_layer = S4Layer(features=2 * channels, lmax=100)

    def forward(self, x, cond_obs, cond_info, diffusion_emb):
        B, C, L = x.shape
        time_emb, feature_emb, cond_mask = cond_info  # (B,time_emb,L),(B,feature_emb,L),(B,K,L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb
        # print("y in RES", y.shape)

        y = self.conv_layer(y)
        # print("y after conv_layer", y.shape)
        y = self.s4_init_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        # print("y after s4_init_layer", y.shape)

        # y = self.time_trans(y.permute(2, 0, 1)).permute(1, 2, 0)

        # cond = torch.cat([cond_obs, cond_mask, time_emb, feature_emb], dim=1)
        cond = torch.cat([cond_obs, cond_mask], dim=1)
        cond = self.cond_conv(cond)

        # feature_time_emb = torch.cat([time_emb, feature_emb], dim=1)
        # feature_time_emb = self.feature_time_conv(feature_time_emb)
        time_emb = self.time_conv(time_emb)

        y = y + cond + time_emb
        y = self.s4_end_layer(y.permute(2, 0, 1)).permute(1, 2, 0)

        # y = self.mid_projection(y)  # (B,2*channel,K*L))

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)

        # x = x.reshape(base_shape)
        # residual = residual.reshape(base_shape)
        # skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip
