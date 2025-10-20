"""
Main experiment script for training GAT on Cora dataset.
Complete Solution for Testing

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

    # Load data
    data = load_cora(root=args.data_root)
    data = data.to(device)

    # Initialize model
    input_dim = data.num_features
    output_dim = data.y.max().item() + 1

    model = GNNStack(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout
    )
    model = model.to(device)

    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Training loop
    best_val_acc = 0.0
    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # Training step
        train_result = train_step(model, data, optimizer, device)
        loss = train_result['loss']
        train_acc = train_result['accuracy']

        # Validation step
        val_result = eval_step(model, data, data.val_mask, device)
        val_acc = val_result['accuracy']

        # Track best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Evaluate on test set when validation improves
            test_result = eval_step(model, data, data.test_mask, device)
            best_test_acc = test_result['accuracy']

        # Print progress every 20 epochs
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # Final evaluation on test set
    test_result = eval_step(model, data, data.test_mask, device)
    final_test_acc = test_result['accuracy']

    print(f"Test Accuracy: {final_test_acc:.4f}")

    # Output machine-readable JSON
    output = {
        "split": "test",
        "accuracy": round(final_test_acc, 4),
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
