"""
Smoke tests to verify basic imports and structure.
Students can run these locally to check their implementation.
"""

import pytest


def test_imports():
    """Test that all required modules can be imported."""
    try:
        from model import GATLayer, GNNStack
        from train import train_step, eval_step
        from data_utils import load_cora, set_seed, get_device
    except ImportError as e:
        pytest.fail(f"Failed to import required modules: {e}")


def test_gat_layer_exists():
    """Test that GATLayer class is defined."""
    from model import GATLayer
    assert hasattr(GATLayer, '__init__')
    assert hasattr(GATLayer, 'forward')


def test_gnn_stack_exists():
    """Test that GNNStack class is defined."""
    from model import GNNStack
    assert hasattr(GNNStack, '__init__')
    assert hasattr(GNNStack, 'forward')


def test_functions_exist():
    """Test that training/eval functions are defined."""
    from train import train_step, eval_step
    assert callable(train_step)
    assert callable(eval_step)


def test_data_loading():
    """Test that Cora dataset can be loaded."""
    from data_utils import load_cora
    data = load_cora(root='./test_data')
    assert data is not None
    assert hasattr(data, 'x')
    assert hasattr(data, 'edge_index')
    assert hasattr(data, 'y')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
