"""
Reference solution for GAT implementation.
This file is kept private and used by the autograder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch.nn import Parameter, Linear
import torch_scatter


class GATLayerSolution(MessagePassing):
    """Reference implementation of GAT Layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        **kwargs
    ):
        super(GATLayerSolution, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope

        # Linear transformation
        self.lin = Linear(in_channels, heads * out_channels, bias=False)

        # Attention parameters
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Glorot/Xavier initialization."""
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        size: tuple = None
    ) -> torch.Tensor:
        """Forward pass."""
        # Apply linear transformation
        x = self.lin(x)  # [N, heads * out_channels]

        # Reshape for multi-head attention
        x = x.view(-1, self.heads, self.out_channels)  # [N, heads, out_channels]

        # Propagate messages
        out = self.propagate(edge_index, x=x, size=size)  # [N, heads, out_channels]

        # Handle multi-head concatenation/averaging
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)  # [N, heads * out_channels]
        else:
            out = out.mean(dim=1)  # [N, out_channels]

        return out

    def message(
        self,
        x_j: torch.Tensor,
        x_i: torch.Tensor,
        index: torch.Tensor,
        ptr: torch.Tensor,
        size_i: int
    ) -> torch.Tensor:
        """Compute attention-weighted messages."""
        # Compute attention logits
        alpha_src = (x_i * self.att_src).sum(dim=-1)  # [E, heads]
        alpha_dst = (x_j * self.att_dst).sum(dim=-1)  # [E, heads]
        alpha = alpha_src + alpha_dst  # [E, heads]
        alpha = F.leaky_relu(alpha, self.negative_slope)

        # Normalize attention coefficients
        alpha = softmax(alpha, index, ptr, size_i)  # [E, heads]

        # Apply dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Weight messages by attention
        return x_j * alpha.unsqueeze(-1)  # [E, heads, out_channels]

    def aggregate(
        self,
        inputs: torch.Tensor,
        index: torch.Tensor,
        ptr: torch.Tensor = None,
        dim_size: int = None
    ) -> torch.Tensor:
        """Aggregate messages using sum."""
        return torch_scatter.scatter(inputs, index, dim=0, dim_size=dim_size, reduce='sum')


class GNNStackSolution(nn.Module):
    """Reference implementation of GNNStack."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        heads: int = 8,
        dropout: float = 0.6
    ):
        super(GNNStackSolution, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout

        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(
            GATLayerSolution(input_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATLayerSolution(heads * hidden_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
            )

        # Output layer
        self.convs.append(
            GATLayerSolution(heads * hidden_dim, output_dim, heads=1, concat=False, dropout=dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through GAT layers."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final layer
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

    def loss(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute negative log-likelihood loss."""
        return F.nll_loss(pred, label)
