"""
Main experiment script for training GAT on Cora dataset.

Usage:
    python run_experiment.py --seed 42 --epochs 200 --lr 0.005
"""

import argparse
import json
import torch
import torch.optim as optim

from data_utils import load_cora, set_seed, get_device
from model import GNNStack
from train import train_step, eval_step


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train GAT on Cora dataset')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cora',
                        help='Dataset name (default: cora)')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for dataset (default: ./data)')

    # Model hyperparameters
    parser.add_argument('--heads', type=int, default=8,
                        help='Number of attention heads (default: 8)')
    parser.add_argument('--hidden_dim', type=int, default=8,
                        help='Hidden dimension per head (default: 8)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of GAT layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.6,
                        help='Dropout probability (default: 0.6)')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate (default: 0.005)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 regularization) (default: 5e-4)')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    return parser.parse_args()


def main():
    """Main training and evaluation loop."""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # TODO: Load data
    # Hint: Use load_cora() from data_utils
    # Move data to device: data = data.to(device)

    # TODO: Initialize model
    # Hint: Determine input_dim from data.num_features
    #       Determine output_dim from data.y.max().item() + 1
    # Create GNNStack with args.hidden_dim, args.num_layers, args.heads, args.dropout
    # Move model to device: model = model.to(device)

    # TODO: Initialize optimizer
    # Hint: Use torch.optim.Adam with model.parameters()
    #       Set lr=args.lr and weight_decay=args.weight_decay

    # TODO: Training loop
    # For each epoch:
    #   1. Call train_step() and get loss and train accuracy
    #   2. Call eval_step() with data.val_mask to get val accuracy
    #   3. Every 20 epochs, print:
    #      f"Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"

    # TODO: Final evaluation on test set
    # Call eval_step() with data.test_mask
    # Print: f"Test Accuracy: {test_acc:.4f}"

    # TODO: Output machine-readable JSON
    # Create a dictionary with the following keys:
    output = {
        "split": "test",
        "accuracy": 0.0,  # REPLACE with actual test accuracy
        "seed": args.seed,
        "config": {
            "heads": args.heads,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "epochs": args.epochs
        }
    }

    # Print JSON on a single line
    print(json.dumps(output))


if __name__ == '__main__':
    main()
