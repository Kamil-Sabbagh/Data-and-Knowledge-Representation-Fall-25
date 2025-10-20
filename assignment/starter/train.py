"""
Training and evaluation functions for GNN models.
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
    # TODO: Implement training step
    # Step 1: Set model to training mode
    # Step 2: Zero gradients
    # Step 3: Forward pass
    # Step 4: Compute loss on training nodes (use data.train_mask)
    # Step 5: Backward pass
    # Step 6: Update parameters
    # Step 7: Compute training accuracy
    # Step 8: Return dictionary with loss and accuracy

    pass  # IMPLEMENT THIS


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
    # TODO: Implement evaluation step
    # Step 1: Set model to evaluation mode
    # Step 2: Forward pass
    # Step 3: Compute predictions (argmax of log-probabilities)
    # Step 4: Compute accuracy on masked nodes
    # Step 5: Return dictionary with accuracy

    pass  # IMPLEMENT THIS


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
