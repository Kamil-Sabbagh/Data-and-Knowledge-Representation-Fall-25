"""
Data loading utilities for graph datasets.
"""

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data


def load_cora(root: str = './data') -> Data:
    """
    Load the Cora citation network dataset.

    Args:
        root: Root directory to store the dataset

    Returns:
        Data object containing the graph
    """
    dataset = Planetoid(root=root, name='Cora')
    data = dataset[0]
    return data


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: Setting deterministic algorithms may slow down training
    # torch.use_deterministic_algorithms(True)


def get_device() -> torch.device:
    """
    Get the appropriate device (CUDA if available, else CPU).

    Returns:
        torch.device object
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
