# Data & Knowledge Representation — Fall 2025

[![Autograde](https://github.com/Kamil-Sabbagh/Data-and-Knowledge-Representation-Fall-25/actions/workflows/grade.yml/badge.svg)](https://github.com/Kamil-Sabbagh/Data-and-Knowledge-Representation-Fall-25/actions/workflows/grade.yml)

Course repository for Data & Knowledge Representation (Fall 2025) at Innopolis University. This repository contains programming assignments with integrated autograding via GitHub Actions.

## 📚 Current Assignment

### Graph Attention Networks: Implementation and Evaluation

**Due Date:** [TBD]
**Assignment Brief:** [View PDF](assignment/DKR_HW_Combined_F25.pdf) | [View Markdown](assignment/DKR_HW_Combined_F25.md)

**Objectives:**
- Implement a custom Graph Attention Network (GAT) layer using PyTorch Geometric
- Build a complete GNN pipeline for node classification
- Conduct reproducible experiments on the Cora citation network
- Achieve ≥76% test accuracy

---

## 🚀 Quick Start for Students

### 1. Fork and Clone

```bash
# Fork this repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/Data-and-Knowledge-Representation-Fall-25.git
cd Data-and-Knowledge-Representation-Fall-25
```

### 2. Create Your Branch

```bash
git checkout -b submission/YOUR_NAME
```

### 3. Set Up Environment

```bash
cd assignment/starter

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Implement Your Solution

Edit the following files:
- `model.py` - Implement `GATLayer` and `GNNStack` classes
- `train.py` - Implement `train_step()` and `eval_step()` functions
- `run_experiment.py` - Complete the training loop and evaluation

**Important:** Do not modify function signatures or return formats!

### 5. Test Locally

```bash
# Run public smoke tests
python -m pytest tests_public/ -v

# Run your experiment
python run_experiment.py --seed 42 --epochs 200
```

### 6. Submit via Pull Request

```bash
# Commit your changes
git add assignment/starter/
git commit -m "Implement GAT assignment"

# Push to your fork
git push origin submission/YOUR_NAME
```

Then open a Pull Request on GitHub:
- Base repository: `Kamil-Sabbagh/Data-and-Knowledge-Representation-Fall-25`
- Base branch: `main`
- Compare branch: `YOUR_FORK:submission/YOUR_NAME`
- Title: `Submission: YOUR_NAME`

---

## 🤖 Autograding System

### How It Works

1. **Open a PR** from your fork to the main repository
2. **Autograder runs** automatically within 5 minutes
3. **Grade posted** as a comment on your PR
4. **Attempt counted** via labels (`attempt-1`, `attempt-2`, `attempt-3`)

### Submission Limits

- **Maximum attempts:** 3
- **What counts as an attempt:** Any push to your PR branch
- **After 3 attempts:** Further pushes will auto-fail
- **Override requests:** Comment on your PR and tag the instructor

### Grade Breakdown

| Task | Points | Description |
|------|--------|-------------|
| Task 1: GAT Layer | 40 | Custom attention mechanism implementation |
| Task 2: GNN Stack & API | 30 | Model architecture and training functions |
| Task 3: Experiment | 25 | Reproducible training pipeline and performance |
| Stretch Goal | +5 | Attention visualization (bonus) |
| **Total** | **100** (+5) | |

### Reading Your Results

The autograder comment will show:
- ✓ **Passed tests:** Green checkmarks with points awarded
- ✗ **Failed tests:** Red X with error messages
- **Total score:** Out of 100 points
- **Remaining attempts:** How many submissions you have left

Example:
```
✓ test_gat_forward_shape: +8 points
✗ test_attention_normalization: 0 points - AssertionError: Output contains NaN
```

---

## 📖 Assignment Structure

```
assignment/
├── DKR_HW_Combined_F25.md           # Assignment brief (Markdown source)
├── DKR_HW_Combined_F25.pdf          # Assignment brief (PDF)
└── starter/
    ├── __init__.py                  # Package init
    ├── data_utils.py                # Dataset loading helpers
    ├── model.py                     # GAT implementation (TODO)
    ├── train.py                     # Training/eval functions (TODO)
    ├── run_experiment.py            # Main experiment script (TODO)
    ├── requirements.txt             # Python dependencies
    └── tests_public/                # Local smoke tests
        ├── __init__.py
        └── test_imports.py
```

---

## 💡 Development Tips

### Testing Strategy

1. **Start small:** Test your `GATLayer` with toy data before full Cora
2. **Use debugger:** Add breakpoints in `message()` and `aggregate()`
3. **Check shapes:** Print tensor shapes at each step
4. **Verify attention:** Ensure weights sum to 1 per node

### Common Issues

**Problem:** `AttributeError: 'GATLayer' object has no attribute 'lin'`
**Solution:** Initialize all parameters in `__init__()` and call `reset_parameters()`

**Problem:** `RuntimeError: Expected shape [N, 7], got [N, 56]`
**Solution:** Check `concat` parameter in your final GAT layer (should be `False`)

**Problem:** Test accuracy is only ~30%
**Solution:** Verify you're using `F.log_softmax` in the final layer and `F.nll_loss` for training

**Problem:** Attention weights don't sum to 1
**Solution:** Use `torch_geometric.utils.softmax` with the `index` parameter, not regular softmax

### Hyperparameter Tuning

Default hyperparameters should achieve ≥76% accuracy:
```python
heads=8, hidden_dim=8, num_layers=2, lr=0.005, weight_decay=5e-4, dropout=0.6
```

If you need to tune:
- **Lower accuracy?** Increase `hidden_dim` or `num_layers`
- **Overfitting?** Increase `dropout` or `weight_decay`
- **Unstable training?** Decrease `lr`

---

## 🔒 Academic Integrity

### Allowed Resources
- ✅ PyTorch and PyTorch Geometric documentation
- ✅ GAT paper (Veličković et al., ICLR 2018)
- ✅ Course lecture notes and materials
- ✅ Discussing high-level concepts with classmates

### Prohibited Actions
- ❌ Copying code from other students
- ❌ Sharing your implementation
- ❌ Using AI tools (ChatGPT, Copilot) for full implementations
- ❌ Submitting code you don't understand

**Consequences:** 0 on assignment + academic integrity report

---

## 📞 Support

### Getting Help

1. **Read the assignment brief carefully** — most questions are answered there
2. **Check the FAQ** in the PDF
3. **Post on the course forum** — don't share code publicly
4. **Attend office hours** — [Schedule TBD]

### Reporting Issues

**Autograder bugs:**
Open an issue with the `autograder` label and include:
- Your PR link
- Error message from CI logs
- Steps to reproduce

**Assignment clarifications:**
Open an issue with the `question` label

---

## 📅 Important Dates

| Event | Date |
|-------|------|
| Assignment Released | [TBD] |
| Office Hours | [TBD] |
| Deadline | [TBD] 23:59 UTC |
| Late Submissions | Not accepted |

---

## 🏆 Leaderboard (Optional)

Top performers (by test accuracy) will be recognized:

| Rank | Name | Accuracy | Submission Date |
|------|------|----------|-----------------|
| 🥇 | — | — | — |
| 🥈 | — | — | — |
| 🥉 | — | — | — |

*(Participation is optional; email instructor to opt-in)*

---

## 📄 License

This repository is for educational purposes only. Code submissions remain the intellectual property of the student authors.

© 2025 Innopolis University

---

## 🤝 Contributing

This is a course assignment repository. Student submissions via PR only. Instructors and TAs can contribute via direct push to `main`.

---

**Questions? Email:** [instructor@innopolis.university]
**Course Page:** [Link TBD]