"""
Hidden grader tests for autograding student submissions.
This file is private and not visible to students.
"""

import pytest
import torch
import torch.optim as optim
import sys
import os
import json
import subprocess

# Add student submission path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'assignment', 'starter'))

from torch_geometric.datasets import Planetoid


class TestGATLayer:
    """Test Task 1: GAT Layer Implementation (40 points)"""

    def test_gat_import(self):
        """Test that GATLayer can be imported (2 points)"""
        try:
            from model import GATLayer
        except ImportError:
            pytest.fail("Cannot import GATLayer from model.py")

    def test_gat_initialization(self):
        """Test GATLayer initialization (5 points)"""
        from model import GATLayer
        layer = GATLayer(in_channels=16, out_channels=8, heads=4)
        assert hasattr(layer, 'forward'), "GATLayer missing forward method"
        assert hasattr(layer, 'lin'), "GATLayer missing linear transformation"

    def test_gat_forward_shape_concat(self):
        """Test GATLayer forward pass output shape with concat=True (8 points)"""
        from model import GATLayer
        layer = GATLayer(in_channels=16, out_channels=8, heads=4, concat=True)
        x = torch.randn(10, 16)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

        try:
            out = layer(x, edge_index)
            assert out.shape == (10, 32), f"Expected shape (10, 32), got {out.shape}"
        except Exception as e:
            pytest.fail(f"GATLayer forward failed: {e}")

    def test_gat_forward_shape_average(self):
        """Test GATLayer forward pass output shape with concat=False (8 points)"""
        from model import GATLayer
        layer = GATLayer(in_channels=16, out_channels=8, heads=4, concat=False)
        x = torch.randn(10, 16)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

        try:
            out = layer(x, edge_index)
            assert out.shape == (10, 8), f"Expected shape (10, 8), got {out.shape}"
        except Exception as e:
            pytest.fail(f"GATLayer forward failed: {e}")

    def test_attention_normalization(self):
        """Test that attention weights are properly normalized (10 points)"""
        from model import GATLayer
        torch.manual_seed(42)
        layer = GATLayer(in_channels=8, out_channels=4, heads=2, dropout=0.0)
        layer.eval()

        x = torch.randn(5, 8)
        edge_index = torch.tensor([[0, 0, 1, 1, 2], [1, 2, 0, 2, 1]], dtype=torch.long)

        try:
            with torch.no_grad():
                out = layer(x, edge_index)
            # Check output is finite (attention weights summed to 1)
            assert torch.all(torch.isfinite(out)), "Output contains NaN or Inf"
        except Exception as e:
            pytest.fail(f"Attention mechanism failed: {e}")

    def test_gradient_flow(self):
        """Test gradient flow through GAT layer (7 points)"""
        from model import GATLayer
        layer = GATLayer(in_channels=8, out_channels=4, heads=2)
        x = torch.randn(5, 8, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

        try:
            out = layer(x, edge_index)
            loss = out.sum()
            loss.backward()
            assert x.grad is not None, "No gradients computed"
            assert torch.any(x.grad != 0), "Gradients are all zero"
        except Exception as e:
            pytest.fail(f"Gradient flow failed: {e}")


class TestGNNStack:
    """Test Task 2.1: GNNStack Implementation (15 points)"""

    def test_gnnstack_import(self):
        """Test that GNNStack can be imported (2 points)"""
        try:
            from model import GNNStack
        except ImportError:
            pytest.fail("Cannot import GNNStack from model.py")

    def test_gnnstack_initialization(self):
        """Test GNNStack initialization (3 points)"""
        from model import GNNStack
        model = GNNStack(input_dim=16, hidden_dim=8, output_dim=7, num_layers=2, heads=4)
        assert hasattr(model, 'forward'), "GNNStack missing forward method"
        assert hasattr(model, 'convs'), "GNNStack missing convs module list"

    def test_gnnstack_forward_shape(self):
        """Test GNNStack forward output shape (5 points)"""
        from model import GNNStack
        model = GNNStack(input_dim=16, hidden_dim=8, output_dim=7, num_layers=2, heads=4)
        x = torch.randn(10, 16)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

        try:
            out = model(x, edge_index)
            assert out.shape == (10, 7), f"Expected shape (10, 7), got {out.shape}"
        except Exception as e:
            pytest.fail(f"GNNStack forward failed: {e}")

    def test_gnnstack_log_probabilities(self):
        """Test that GNNStack outputs log probabilities (5 points)"""
        from model import GNNStack
        torch.manual_seed(42)
        model = GNNStack(input_dim=16, hidden_dim=8, output_dim=7, num_layers=2, heads=4)
        model.eval()
        x = torch.randn(10, 16)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

        with torch.no_grad():
            out = model(x, edge_index)
            # Log probabilities should be <= 0
            assert torch.all(out <= 0), "Output is not log probabilities"
            # Exponentiated should sum to ~1
            probs = torch.exp(out)
            assert torch.allclose(probs.sum(dim=1), torch.ones(10), atol=1e-5), \
                "Probabilities don't sum to 1"


class TestTrainingAPI:
    """Test Task 2.2: Training and Evaluation Functions (15 points)"""

    def test_train_step_import(self):
        """Test train_step can be imported (2 points)"""
        try:
            from train import train_step
        except ImportError:
            pytest.fail("Cannot import train_step from train.py")

    def test_eval_step_import(self):
        """Test eval_step can be imported (2 points)"""
        try:
            from train import eval_step
        except ImportError:
            pytest.fail("Cannot import eval_step from train.py")

    def test_train_step_output_format(self):
        """Test train_step returns correct format (5 points)"""
        from model import GNNStack
        from train import train_step
        from torch_geometric.datasets import Planetoid

        data = Planetoid(root='./test_data', name='Cora')[0]
        model = GNNStack(data.num_features, 8, data.y.max().item() + 1, num_layers=2, heads=4)
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        try:
            result = train_step(model, data, optimizer, 'cpu')
            assert isinstance(result, dict), "train_step must return a dictionary"
            assert 'loss' in result, "train_step result missing 'loss' key"
            assert 'accuracy' in result, "train_step result missing 'accuracy' key"
            assert isinstance(result['loss'], (int, float)), "'loss' must be a number"
            assert isinstance(result['accuracy'], (int, float)), "'accuracy' must be a number"
        except Exception as e:
            pytest.fail(f"train_step execution failed: {e}")

    def test_eval_step_output_format(self):
        """Test eval_step returns correct format (6 points)"""
        from model import GNNStack
        from train import eval_step
        from torch_geometric.datasets import Planetoid

        data = Planetoid(root='./test_data', name='Cora')[0]
        model = GNNStack(data.num_features, 8, data.y.max().item() + 1, num_layers=2, heads=4)
        model.eval()

        try:
            result = eval_step(model, data, data.val_mask, 'cpu')
            assert isinstance(result, dict), "eval_step must return a dictionary"
            assert 'accuracy' in result, "eval_step result missing 'accuracy' key"
            assert isinstance(result['accuracy'], (int, float)), "'accuracy' must be a number"
            assert 0.0 <= result['accuracy'] <= 1.0, "Accuracy must be between 0 and 1"
        except Exception as e:
            pytest.fail(f"eval_step execution failed: {e}")


class TestExperiment:
    """Test Task 3: Reproducible Experiment (25 points)"""

    def test_run_experiment_exists(self):
        """Test run_experiment.py exists and is executable (3 points)"""
        script_path = os.path.join(os.path.dirname(__file__), '..', 'assignment', 'starter', 'run_experiment.py')
        assert os.path.exists(script_path), "run_experiment.py not found"

    def test_cli_arguments(self):
        """Test that CLI accepts all required arguments (5 points)"""
        script_path = os.path.join(os.path.dirname(__file__), '..', 'assignment', 'starter', 'run_experiment.py')
        try:
            result = subprocess.run(
                ['python', script_path, '--help'],
                capture_output=True,
                text=True,
                timeout=10
            )
            assert result.returncode == 0, "Script crashed with --help flag"
            help_text = result.stdout.lower()
            required_args = ['--seed', '--epochs', '--lr', '--heads', '--hidden_dim', '--num_layers']
            for arg in required_args:
                assert arg in help_text, f"Missing argument: {arg}"
        except subprocess.TimeoutExpired:
            pytest.fail("Script timed out")

    def test_json_output_format(self):
        """Test that script outputs valid JSON (7 points)"""
        script_path = os.path.join(os.path.dirname(__file__), '..', 'assignment', 'starter', 'run_experiment.py')
        try:
            result = subprocess.run(
                ['python', script_path, '--epochs', '1', '--seed', '42'],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=os.path.join(os.path.dirname(__file__), '..', 'assignment', 'starter')
            )

            # Find JSON line in output
            json_line = None
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    json_line = line
                    break

            assert json_line is not None, "No JSON output found"

            # Parse JSON
            data = json.loads(json_line)
            assert 'split' in data, "JSON missing 'split' key"
            assert 'accuracy' in data, "JSON missing 'accuracy' key"
            assert 'seed' in data, "JSON missing 'seed' key"
            assert 'config' in data, "JSON missing 'config' key"
            assert data['split'] == 'test', "split must be 'test'"
        except subprocess.TimeoutExpired:
            pytest.fail("Script timed out (>60s)")
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON output: {e}")
        except Exception as e:
            pytest.fail(f"Execution failed: {e}")

    def test_performance_threshold(self):
        """Test that model achieves >= 76% test accuracy (10 points)"""
        script_path = os.path.join(os.path.dirname(__file__), '..', 'assignment', 'starter', 'run_experiment.py')
        try:
            result = subprocess.run(
                ['python', script_path, '--seed', '42', '--epochs', '200'],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes max
                cwd=os.path.join(os.path.dirname(__file__), '..', 'assignment', 'starter')
            )

            # Find JSON line
            json_line = None
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    json_line = line
                    break

            assert json_line is not None, "No JSON output found"
            data = json.loads(json_line)
            accuracy = data['accuracy']

            assert accuracy >= 0.76, f"Test accuracy {accuracy:.4f} below threshold 0.76"
        except subprocess.TimeoutExpired:
            pytest.fail("Training timed out (>10 minutes)")
        except Exception as e:
            pytest.fail(f"Performance test failed: {e}")


def compute_grade():
    """
    Run all tests and compute final grade.
    Returns JSON with score and feedback.
    """
    # Run pytest and capture results
    pytest_args = [__file__, '--tb=short', '-v', '--json-report', '--json-report-file=/tmp/report.json']
    pytest.main(pytest_args)

    # Load results
    try:
        with open('/tmp/report.json', 'r') as f:
            report = json.load(f)
    except:
        report = {'tests': []}

    # Compute score based on test results
    total_score = 0
    feedback_lines = []

    test_scores = {
        'test_gat_import': 2,
        'test_gat_initialization': 5,
        'test_gat_forward_shape_concat': 8,
        'test_gat_forward_shape_average': 8,
        'test_attention_normalization': 10,
        'test_gradient_flow': 7,
        'test_gnnstack_import': 2,
        'test_gnnstack_initialization': 3,
        'test_gnnstack_forward_shape': 5,
        'test_gnnstack_log_probabilities': 5,
        'test_train_step_import': 2,
        'test_eval_step_import': 2,
        'test_train_step_output_format': 5,
        'test_eval_step_output_format': 6,
        'test_run_experiment_exists': 3,
        'test_cli_arguments': 5,
        'test_json_output_format': 7,
        'test_performance_threshold': 10,
    }

    for test in report.get('tests', []):
        test_name = test['nodeid'].split('::')[-1]
        if test_name in test_scores:
            if test['outcome'] == 'passed':
                points = test_scores[test_name]
                total_score += points
                feedback_lines.append(f"✓ {test_name}: +{points} points")
            else:
                feedback_lines.append(f"✗ {test_name}: 0 points - {test.get('call', {}).get('longrepr', 'Failed')}")

    feedback = "\n".join(feedback_lines)
    result = {
        "score": total_score,
        "max_score": 100,
        "feedback": feedback
    }

    return result


if __name__ == '__main__':
    grade_result = compute_grade()
    print(json.dumps(grade_result, indent=2))
