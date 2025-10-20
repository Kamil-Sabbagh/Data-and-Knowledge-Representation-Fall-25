"""
Training and evaluation functions for GNN models.
Complete Solution for Testing
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data


def train_step(
    model: torch.nn.Module,
    data: Data,
    optimizer: torch.optim.Optimizer,
    device: str
) -> dict:
    """
    Perform one training step.

    Args:
        model: The GNN model to train
        data: The graph data object with train_mask
        optimizer: The optimizer for parameter updates
        device: Device to run on ('cpu' or 'cuda')

    Returns:
        Dictionary with keys:
            - 'loss': float, the training loss value
            - 'accuracy': float, the training accuracy
    """
    model.train()
    optimizer.zero_grad()

    # Forward pass
    out = model(data.x, data.edge_index)

    # Compute loss on training nodes
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    # Backward pass
    loss.backward()
    optimizer.step()

    # Compute training accuracy
    train_acc = compute_accuracy(out, data.y, data.train_mask)

    return {
        'loss': loss.item(),
        'accuracy': train_acc
    }


@torch.no_grad()
def eval_step(
    model: torch.nn.Module,
    data: Data,
    mask: torch.Tensor,
    device: str
) -> dict:
    """
    Evaluate model on a given mask (validation or test).

    Args:
        model: The GNN model to evaluate
        data: The graph data object
        mask: Boolean mask indicating which nodes to evaluate
        device: Device to run on ('cpu' or 'cuda')

    Returns:
        Dictionary with keys:
            - 'accuracy': float, the accuracy on masked nodes
    """
    model.eval()

    # Forward pass
    out = model(data.x, data.edge_index)

    # Compute accuracy on masked nodes
    accuracy = compute_accuracy(out, data.y, mask)

    return {
        'accuracy': accuracy
    }


def compute_accuracy(pred: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Helper function to compute accuracy.

    Args:
        pred: Predicted log-probabilities [num_nodes, num_classes]
        labels: Ground truth labels [num_nodes]
        mask: Boolean mask [num_nodes]

    Returns:
        Accuracy as a float
    """
    pred_labels = pred.argmax(dim=1)
    correct = (pred_labels[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0
