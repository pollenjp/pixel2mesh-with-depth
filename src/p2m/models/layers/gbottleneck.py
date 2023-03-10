# Third Party Library
import torch
import torch.nn as nn
import torch.nn.functional as F

# First Party Library
from p2m.models.layers.gconv import GConv


class GResBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, adj_mat, activation=None):
        super(GResBlock, self).__init__()

        self.conv1 = GConv(in_features=in_dim, out_features=hidden_dim, adj_mat=adj_mat)
        self.conv2 = GConv(in_features=hidden_dim, out_features=in_dim, adj_mat=adj_mat)
        self.activation = F.relu if activation else None

    def forward(self, inputs):
        x = self.conv1(inputs)
        if self.activation:
            x = self.activation(x)
        x = self.conv2(x)
        if self.activation:
            x = self.activation(x)

        return (inputs + x) * 0.5


class GBottleneck(nn.Module):
    def __init__(
        self,
        block_num: int,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        adj_mat: torch.Tensor,
        activation: bool | None = None,
    ):
        super(GBottleneck, self).__init__()

        resblock_layers = [
            GResBlock(
                in_dim=hidden_dim,
                hidden_dim=hidden_dim,
                adj_mat=adj_mat,
                activation=activation,
            )
            for _ in range(block_num)
        ]
        self.blocks = nn.Sequential(*resblock_layers)
        self.conv1 = GConv(in_features=in_dim, out_features=hidden_dim, adj_mat=adj_mat)
        self.conv2 = GConv(in_features=hidden_dim, out_features=out_dim, adj_mat=adj_mat)
        self.activation = F.relu if activation else None

    def forward(
        self,
        inputs: torch.Tensor,
    ):
        """_summary_

        Args:
            inputs (torch.Tensor):
                (num_batch, num_point, num_channel)

        Returns:
            _type_: _description_
        """
        x = self.conv1(inputs)
        if self.activation:
            x = self.activation(x)
        x_hidden = self.blocks(x)
        x_out = self.conv2(x_hidden)

        return x_out, x_hidden
