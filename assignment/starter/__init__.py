"""
Graph Attention Network Implementation
Data & Knowledge Representation - Fall 2025
"""

__version__ = "1.0.0"

from .model import GATLayer, GNNStack
from .train import train_step, eval_step

__all__ = ['GATLayer', 'GNNStack', 'train_step', 'eval_step']
