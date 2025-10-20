"""
Graph Attention Network (GAT) implementation using PyTorch Geometric.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch.nn import Parameter, Linear
import torch_scatter


class GATLayer(MessagePassing):
    """
    Graph Attention Layer implementation.

    Implements the attention mechanism from "Graph Attention Networks"
    (Veličković et al., ICLR 2018) using PyTorch Geometric's MessagePassing.

    Args:
        in_channels: Size of each input sample
        out_channels: Size of each output sample
        heads: Number of multi-head attentions (default: 1)
        concat: If True, concatenate head outputs; if False, average them (default: True)
        dropout: Dropout probability for attention coefficients (default: 0.0)
        negative_slope: LeakyReLU negative slope for attention mechanism (default: 0.2)
    """

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
        super(GATLayer, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope

        # TODO: Initialize linear transformation
        # Hint: Use nn.Linear or nn.Parameter for weight matrix W
        # Shape should be [in_channels, heads * out_channels]
        self.lin = None  # REPLACE THIS

        # TODO: Initialize attention parameters
        # Hint: Create learnable vectors for source and target attention
        # att_src shape: [1, heads, out_channels]
        # att_dst shape: [1, heads, out_channels]
        self.att_src = None  # REPLACE THIS
        self.att_dst = None  # REPLACE THIS

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize learnable parameters.

        TODO: Use appropriate initialization (e.g., Xavier/Glorot)
        """
        pass  # IMPLEMENT THIS

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        size: tuple = None
    ) -> torch.Tensor:
        """
        Forward pass of the GAT layer.

        Args:
            x: Node feature matrix of shape [num_nodes, in_channels]
            edge_index: Edge indices of shape [2, num_edges]
            size: Size of source and target nodes (for bipartite graphs)

        Returns:
            Updated node features of shape:
                - [num_nodes, heads * out_channels] if concat=True
                - [num_nodes, out_channels] if concat=False
        """
        # TODO: Implement forward pass
        # Step 1: Apply linear transformation to node features
        # Step 2: Reshape for multi-head attention
        # Step 3: Initiate message passing via self.propagate()
        # Step 4: Handle concatenation/averaging of heads

        pass  # IMPLEMENT THIS

    def message(
        self,
        x_j: torch.Tensor,
        x_i: torch.Tensor,
        index: torch.Tensor,
        ptr: torch.Tensor,
        size_i: int
    ) -> torch.Tensor:
        """
        Construct messages from source nodes to target nodes.

        Args:
            x_j: Source node features [num_edges, heads, out_channels]
            x_i: Target node features [num_edges, heads, out_channels]
            index: Target node indices for each edge
            ptr: Compressed index pointer (CSR format)
            size_i: Number of target nodes

        Returns:
            Attention-weighted messages [num_edges, heads, out_channels]
        """
        # TODO: Implement attention mechanism
        # Step 1: Compute attention logits e_ij = LeakyReLU(a^T [h_i || h_j])
        #         Use self.att_src and self.att_dst
        # Step 2: Normalize attention coefficients using softmax
        #         Hint: Use torch_geometric.utils.softmax with index parameter
        # Step 3: Apply dropout to attention coefficients during training
        # Step 4: Weight source features by attention coefficients
        # Step 5: Return weighted messages

        pass  # IMPLEMENT THIS

    def aggregate(
        self,
        inputs: torch.Tensor,
        index: torch.Tensor,
        ptr: torch.Tensor = None,
        dim_size: int = None
    ) -> torch.Tensor:
        """
        Aggregate messages from neighbors.

        Args:
            inputs: Messages to aggregate [num_edges, heads, out_channels]
            index: Target node indices for each message
            ptr: Compressed index pointer (CSR format)
            dim_size: Number of target nodes

        Returns:
            Aggregated messages [num_nodes, heads, out_channels]
        """
        # TODO: Implement aggregation
        # Hint: Use torch_scatter.scatter to sum messages for each target node
        # Use scatter(inputs, index, dim=0, dim_size=dim_size, reduce='sum')

        pass  # IMPLEMENT THIS


class GNNStack(nn.Module):
    """
    Stack of GAT layers for node classification.

    Args:
        input_dim: Dimension of input node features
        hidden_dim: Dimension of hidden representations (per head)
        output_dim: Number of output classes
        num_layers: Number of GAT layers
        heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        heads: int = 8,
        dropout: float = 0.6
    ):
        super(GNNStack, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout

        # TODO: Build layers
        # Hint: Use nn.ModuleList to store GAT layers
        # - First layer: input_dim -> hidden_dim, with 'heads' heads, concat=True
        # - Middle layers: heads*hidden_dim -> hidden_dim, with 'heads' heads, concat=True
        # - Last layer: heads*hidden_dim -> output_dim, with 1 head, concat=False

        self.convs = nn.ModuleList()
        # IMPLEMENT LAYER CONSTRUCTION HERE

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the GNN stack.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Log-probabilities [num_nodes, output_dim]
        """
        # TODO: Implement forward pass
        # For each layer except the last:
        #   - Apply GAT layer
        #   - Apply ELU activation
        #   - Apply dropout
        # For the last layer:
        #   - Apply GAT layer
        #   - Apply log_softmax

        pass  # IMPLEMENT THIS

    def loss(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood loss.

        Args:
            pred: Log-probabilities [num_nodes, num_classes]
            label: Ground truth labels [num_nodes]

        Returns:
            Loss value (scalar tensor)
        """
        return F.nll_loss(pred, label)
