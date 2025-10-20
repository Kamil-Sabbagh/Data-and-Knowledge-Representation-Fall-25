# Graph Attention Networks: Implementation and Evaluation
## Data & Knowledge Representation — Fall 2025

---

## Table of Contents
1. [Introduction](#introduction)
2. [Learning Objectives](#learning-objectives)
3. [Problem Statement](#problem-statement)
4. [Background: Graph Attention Networks](#background-graph-attention-networks)
5. [Assignment Tasks](#assignment-tasks)
6. [Technical Requirements](#technical-requirements)
7. [Submission Workflow](#submission-workflow)
8. [Grading Breakdown](#grading-breakdown)
9. [Academic Integrity](#academic-integrity)
10. [FAQ](#faq)

---

## 1. Introduction

Graph Neural Networks (GNNs) have revolutionized machine learning on graph-structured data. Among GNN architectures, **Graph Attention Networks (GAT)** introduce an attention mechanism that dynamically weighs the importance of neighboring nodes during message aggregation, enabling the network to focus on the most relevant structural patterns.

In this assignment, you will:
- Implement a custom GAT layer from scratch using PyTorch Geometric's MessagePassing framework
- Build a complete GNN pipeline with training and evaluation APIs
- Conduct reproducible experiments on a real-world citation network dataset
- Achieve a minimum performance threshold through hyperparameter tuning

This assignment combines implementation depth with experimental rigor, preparing you for research and industry applications in graph machine learning.

---

## 2. Learning Objectives

By completing this assignment, you will:

1. **Understand** the mathematical foundations of attention mechanisms in GNNs
2. **Implement** custom message-passing layers using PyTorch Geometric
3. **Design** modular, testable training pipelines with deterministic outputs
4. **Conduct** reproducible experiments with proper seed management and hyperparameter control
5. **Evaluate** model performance using standard metrics and structured output formats
6. **Debug** neural network implementations through systematic testing

---

## 3. Problem Statement

**Given:** A graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with:
- Node features: $\mathbf{X} \in \mathbb{R}^{N \times F}$ where $N$ = number of nodes, $F$ = feature dimension
- Edge set: $\mathcal{E}$ representing connections between nodes
- Node labels: $\mathbf{y} \in \{1, \ldots, C\}^N$ for $C$ classes

**Task:** Implement a Graph Attention Network to perform **node classification**, predicting the label $y_i$ for each node $v_i \in \mathcal{V}$ based on its features and graph structure.

**Dataset:** The **Cora** citation network, where:
- Nodes = academic papers (2,708 nodes)
- Edges = citation links (10,556 edges)
- Features = bag-of-words representation (1,433 dimensions)
- Labels = research topic (7 classes)

---

## 4. Background: Graph Attention Networks

### 4.1 GAT Layer Formulation

For each node $i$, a GAT layer performs:

**1. Linear Transformation:**
$$\mathbf{h}'_i = \mathbf{W}\mathbf{h}_i$$
where $\mathbf{W} \in \mathbb{R}^{F' \times F}$ is a learnable weight matrix.

**2. Attention Mechanism:**

Compute attention coefficients for each neighbor $j \in \mathcal{N}_i$:
$$e_{ij} = \text{LeakyReLU}\left(\vec{\mathbf{a}}^\top [\mathbf{h}'_i \| \mathbf{h}'_j]\right)$$
where $\vec{\mathbf{a}} \in \mathbb{R}^{2F'}$ is a learnable attention vector, $\|$ denotes concatenation.

Normalize using softmax:
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})}$$

**3. Aggregation:**
$$\mathbf{h}''_i = \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij} \mathbf{h}'_j\right)$$
where $\sigma$ is an activation function (e.g., ELU).

**4. Multi-Head Attention:**

Use $K$ independent attention mechanisms (heads) and concatenate:
$$\mathbf{h}''_i = \|_{k=1}^K \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha^{(k)}_{ij} \mathbf{h}'^{(k)}_j\right)$$

---

## 5. Assignment Tasks

### **Task 1: Implement the GAT Layer (40 points)**

Implement a custom `GATLayer` class as a subclass of PyTorch Geometric's `MessagePassing`.

**Requirements:**
- **Class Name:** `GATLayer`
- **Constructor Signature:**
  ```python
  def __init__(self, in_channels: int, out_channels: int, heads: int = 1,
               concat: bool = True, dropout: float = 0.0, negative_slope: float = 0.2)
  ```
- **Forward Signature:**
  ```python
  def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor
  ```

**Implementation Details:**
1. Use `self.lin` for the linear transformation $\mathbf{W}$
2. Use `self.att_src` and `self.att_dst` for attention parameters (split $\vec{\mathbf{a}}$)
3. Implement `message()` to compute attention-weighted messages
4. Implement `aggregate()` to sum messages from neighbors
5. Support multi-head attention with proper concatenation/averaging
6. Apply dropout to attention coefficients during training

**Output Contract:**
- For `concat=True`: output shape `[N, heads * out_channels]`
- For `concat=False`: output shape `[N, out_channels]` (averaged across heads)

**Testing Criteria:**
- Attention weights sum to 1 for each node
- Output shape matches specification
- Deterministic behavior with fixed seed
- Gradient flow through attention mechanism

---

### **Task 2: Build GNNStack and Training API (30 points)**

Implement a complete GNN model and training/evaluation functions.

**2.1 GNNStack Class (15 points)**

**Class Name:** `GNNStack`

**Constructor Signature:**
```python
def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
             num_layers: int, heads: int = 8, dropout: float = 0.6)
```

**Forward Signature:**
```python
def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor
```

**Requirements:**
- Stack multiple `GATLayer` instances
- Use `heads` attention heads in hidden layers
- Use 1 head in final layer (average aggregation)
- Apply dropout between layers
- Return log-probabilities (use `F.log_softmax`)

**2.2 Training and Evaluation Functions (15 points)**

**Function Signatures:**
```python
def train_step(model: torch.nn.Module, data: torch_geometric.data.Data,
               optimizer: torch.optim.Optimizer, device: str) -> dict:
    """
    Performs one training step.

    Returns:
        dict with keys: {'loss': float, 'accuracy': float}
    """
    pass

def eval_step(model: torch.nn.Module, data: torch_geometric.data.Data,
              mask: torch.Tensor, device: str) -> dict:
    """
    Evaluates model on given mask.

    Returns:
        dict with keys: {'accuracy': float}
    """
    pass
```

**Requirements:**
- `train_step`: compute loss on `data.train_mask`, return loss and train accuracy
- `eval_step`: compute accuracy on provided mask (val/test)
- Use `F.nll_loss` for loss computation
- Return dictionaries with exact keys as specified
- Functions must be stateless (no global variables)

---

### **Task 3: Reproducible Experiment (25 points)**

Implement `run_experiment.py` to train and evaluate the model.

**CLI Signature:**
```bash
python run_experiment.py --dataset cora --heads 8 --hidden_dim 8 --num_layers 2 \
                         --epochs 200 --lr 0.005 --weight_decay 5e-4 --dropout 0.6 \
                         --seed 42
```

**Requirements:**

1. **Data Loading:**
   - Use `torch_geometric.datasets.Planetoid` to load Cora
   - Dataset root: `./data/`

2. **Model Initialization:**
   - Create `GNNStack` with specified hyperparameters
   - Move model and data to appropriate device

3. **Training Loop:**
   - Train for specified epochs
   - Use Adam optimizer
   - Print progress every 20 epochs: `Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}`

4. **Final Evaluation:**
   - Evaluate on test set
   - Print final result: `Test Accuracy: {test_acc:.4f}`

5. **Machine-Readable Output:**
   - Print a single JSON line at the end:
   ```json
   {"split": "test", "accuracy": 0.8120, "seed": 42, "config": {"heads": 8, "hidden_dim": 8, "num_layers": 2, "lr": 0.005, "weight_decay": 0.0005, "dropout": 0.6, "epochs": 200}}
   ```

**Performance Requirement:**
- Achieve **≥76.0% test accuracy** on Cora with default hyperparameters
- Must complete within 200 epochs

**Reproducibility:**
- Set all random seeds: `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`, `np.random.seed(seed)`
- Use deterministic algorithms where possible

---

### **Stretch Goal: Attention Visualization (5 points bonus)**

Implement an attention analysis function:

```python
def export_attention_weights(model: torch.nn.Module, data: torch_geometric.data.Data,
                            node_id: int, layer_idx: int = 0, head_idx: int = 0) -> dict:
    """
    Export attention weights for a specific node.

    Returns:
        dict with keys: {'node_id': int, 'neighbor_ids': list, 'attention_weights': list}
    """
    pass
```

Save attention weights to `attention_weights.json` for a specified node.

---

## 6. Technical Requirements

### 6.1 Environment

**Python Version:** 3.8+

**Required Packages:**
```
torch>=1.13.0
torch-geometric>=2.2.0
numpy>=1.21.0
```

Install PyTorch Geometric:
```bash
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+{CUDA}.html
```

### 6.2 Code Structure

Your submission must follow this structure:
```
starter/
├── __init__.py
├── data_utils.py       # Dataset loading helpers
├── model.py            # GATLayer and GNNStack classes
├── train.py            # train_step and eval_step functions
└── run_experiment.py   # Main CLI script
```

### 6.3 Code Contract

**DO NOT CHANGE:**
- Function names and signatures as specified
- Return dictionary keys and types
- JSON output format

**YOU MAY:**
- Add helper methods/functions
- Add docstrings and comments
- Add type hints
- Refactor internal implementations

### 6.4 Output Format Specification

**JSON Output Schema:**
```json
{
  "split": "test",              // Must be "test"
  "accuracy": 0.8120,           // Float, 4 decimal places
  "seed": 42,                   // Integer
  "config": {
    "heads": 8,                 // Integer
    "hidden_dim": 8,            // Integer
    "num_layers": 2,            // Integer
    "lr": 0.005,                // Float
    "weight_decay": 0.0005,     // Float
    "dropout": 0.6,             // Float
    "epochs": 200               // Integer
  }
}
```

---

## 7. Submission Workflow

### 7.1 Fork and Branch

1. **Fork** the repository: `https://github.com/Kamil-Sabbagh/Data-and-Knowledge-Representation-Fall-25`
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/Data-and-Knowledge-Representation-Fall-25.git`
3. **Create a branch**: `git checkout -b submission/YOUR_NAME`

### 7.2 Implement

4. Work in the `assignment/starter/` directory
5. Implement all required classes and functions
6. Test locally using `tests_public/`

### 7.3 Test Locally

```bash
cd assignment/starter
python -m pytest ../tests_public/ -v
python run_experiment.py --seed 42
```

### 7.4 Submit

7. **Commit** your changes:
   ```bash
   git add assignment/starter/
   git commit -m "Implement GAT assignment"
   ```
8. **Push** to your fork:
   ```bash
   git push origin submission/YOUR_NAME
   ```
9. **Open a Pull Request** to the original repository:
   - Base: `main`
   - Compare: `your-branch`
   - Title: `Submission: YOUR_NAME`

### 7.5 Autograding

- The autograder runs automatically when you open/update a PR
- Grade and feedback appear as a PR comment within 5 minutes
- You have **3 submission attempts maximum**
- Each push to your PR branch counts as one attempt
- After 3 attempts, additional pushes will auto-fail

### 7.6 Attempt Counter

A label will be added to your PR indicating attempts:
- `attempt-1`: First submission
- `attempt-2`: Second submission
- `attempt-3`: Final submission
- After attempt-3, further submissions receive: `attempts-exhausted`

**To get an override:** Request manual review from the instructor by adding a comment to your PR.

---

## 8. Grading Breakdown

| Task | Points | Criteria |
|------|--------|----------|
| **Task 1: GAT Layer** | 40 | |
| - Correct attention computation | 15 | Attention weights sum to 1, proper softmax normalization |
| - Multi-head support | 10 | Handles multiple heads, correct concatenation/averaging |
| - Message passing implementation | 10 | Proper use of PyG's MessagePassing API |
| - Code quality | 5 | Docstrings, type hints, clean code |
| **Task 2: GNNStack & Training** | 30 | |
| - GNNStack architecture | 10 | Correct layer stacking, dropout, output format |
| - train_step function | 10 | Correct loss computation, gradient updates, return format |
| - eval_step function | 10 | Correct accuracy computation, return format |
| **Task 3: Experiment** | 25 | |
| - CLI and argument parsing | 5 | Correct argument handling, all hyperparameters exposed |
| - Training loop | 5 | Correct training procedure, progress logging |
| - JSON output format | 5 | Exact schema match, parseable |
| - Performance threshold | 10 | Test accuracy ≥ 76.0% with default hyperparameters |
| **Stretch Goal** | +5 | Attention visualization implementation |
| **Deductions** | | |
| - Modified function signatures | -10 | Per violation |
| - Missing JSON output | -15 | Cannot parse submission |
| - Non-deterministic results | -5 | Results vary with same seed |
| - Code quality issues | -5 | Poor style, no docstrings, hard-coded paths |
| **Total** | 100 (+5) | |

---

## 9. Academic Integrity

### Allowed:
- Consulting PyTorch and PyTorch Geometric documentation
- Reading GAT paper and related academic papers
- Discussing high-level concepts with classmates
- Using course materials and lecture notes

### **NOT Allowed:**
- Copying code from other students or online repositories
- Sharing your implementation with other students
- Using ChatGPT/Copilot to generate full implementations
- Submitting someone else's code as your own

**Violation Consequences:**
- First offense: 0 on assignment + academic integrity report
- Second offense: F in course + university disciplinary action

### Plagiarism Detection:
- All submissions are checked using automated similarity detection
- Git history is reviewed for suspicious patterns
- Attention weight distributions are analyzed for anomalies

---

## 10. FAQ

**Q: Can I use PyG's built-in GATConv?**
A: No. You must implement the GAT layer from scratch using MessagePassing. Using built-in GATConv will result in 0 points for Task 1.

**Q: What if I can't reach 76% accuracy?**
A: First, ensure your implementation is correct. Then, tune hyperparameters within reasonable ranges. Document your tuning process. Partial credit awarded for >70% with correct implementation.

**Q: Can I use additional libraries?**
A: Only if they're for visualization or logging (e.g., matplotlib, wandb). All core functionality must use torch/PyG.

**Q: How do I debug attention weights?**
A: Add print statements in `message()` to inspect attention coefficients. Check that they sum to 1 per node.

**Q: What device should I use?**
A: Your code should support both CPU and CUDA. Use `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`.

**Q: How long should training take?**
A: On CPU: ~5-10 minutes. On GPU: <1 minute. If slower, check for inefficiencies.

**Q: Can I submit late?**
A: No. The autograder closes at the deadline. Plan for technical issues.

**Q: What if the autograder fails?**
A: Check CI logs in your PR for error messages. Fix and push again (counts as an attempt).

**Q: Can I delete my branch and resubmit?**
A: No. Attempt counter is per-student, not per-branch. Closing and reopening PRs doesn't reset attempts.

---

## Contact

**Instructor:** [Your Name]
**Email:** [email@innopolis.university]
**Office Hours:** [Schedule]
**Course Forum:** [Link]

**Deadline:** [Date and Time]

---

**Good luck! Remember: understanding GAT attention mechanisms deeply will serve you throughout your career in graph machine learning.**
