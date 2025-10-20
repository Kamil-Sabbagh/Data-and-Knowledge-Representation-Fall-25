# Instructor Guide: GAT Assignment Autograding System

## Overview

This document explains the complete autograding infrastructure for the Graph Attention Networks assignment. The system uses GitHub Actions to automatically grade student submissions via Pull Requests, with a 3-attempt limit and detailed feedback.

---

## System Architecture

### Components

1. **Public Repository** (`Data-and-Knowledge-Representation-Fall-25`)
   - Student-facing assignment materials
   - Starter code templates with TODOs
   - Public smoke tests
   - GitHub Actions workflows
   - README with submission instructions

2. **Private Grader** (`grader_private/` directory)
   - Reference solution implementation
   - Comprehensive test suite (pytest-based)
   - Grade computation logic
   - **Note:** In production, this should be in a separate private repository

3. **GitHub Actions CI**
   - Triggered on Pull Requests to `main`
   - Runs grading tests automatically
   - Posts results as PR comments
   - Manages attempt counting via labels

---

## Assignment Structure

### Task Breakdown

**Task 1: GAT Layer (40 points)**
- Custom attention mechanism using PyG's MessagePassing
- Multi-head attention support
- Proper normalization and gradient flow

**Task 2: GNN Stack & Training API (30 points)**
- Model architecture with multiple GAT layers
- Training step function with loss computation
- Evaluation step function with accuracy computation

**Task 3: Reproducible Experiment (25 points)**
- CLI for hyperparameter control
- Training loop with progress logging
- Machine-readable JSON output
- Performance threshold: ≥76% test accuracy

**Stretch Goal: Attention Visualization (+5 points bonus)**
- Export attention weights for analysis
- Optional but encouraged

---

## Autograding Workflow

### Student Submission Process

1. Student forks the repository
2. Creates a branch: `submission/STUDENT_NAME`
3. Implements solution in `assignment/starter/`
4. Pushes and opens a PR to upstream `main`
5. Autograder runs automatically

### Grading Pipeline

```
PR Opened/Updated
    ↓
Check Attempts (via labels)
    ↓
    ├─→ [Attempts < 3] → Run Grader
    │       ↓
    │   Install Dependencies
    │       ↓
    │   Fetch Private Grader
    │       ↓
    │   Run Test Suite
    │       ↓
    │   Compute Score
    │       ↓
    │   Post Comment with Grade
    │       ↓
    │   Update Attempt Label
    │
    └─→ [Attempts ≥ 3] → Post "Exhausted" Message
```

### Attempt Tracking

Attempts are tracked using GitHub labels:
- `attempt-1`: First submission
- `attempt-2`: Second submission
- `attempt-3`: Third/final submission
- `attempts-exhausted`: No more attempts allowed

**Override Mechanism:**
- Instructor can add `override` label to allow additional attempts
- Useful for technical issues or special circumstances

---

## Test Suite Details

### Test Categories

**1. API Compliance Tests (12 points)**
- Module imports work correctly
- Classes/functions exist with correct signatures
- Basic initialization succeeds

**2. Functional Tests (40 points)**
- Forward pass produces correct output shapes
- Attention weights are normalized (sum to 1)
- Multi-head concatenation/averaging works
- Gradient flow through network

**3. Integration Tests (23 points)**
- Training loop executes without errors
- Evaluation functions return correct format
- JSON output matches specification
- CLI accepts all required arguments

**4. Performance Test (10 points)**
- Model achieves ≥76% test accuracy on Cora
- Training completes within reasonable time (10 min)

### Grading Rubric

```python
{
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
```

Total: 95 points mapped to 100-point scale

---

## Setup Instructions

### Initial Repository Setup

1. **Create/Configure Main Repository**

```bash
# Already done - repository exists at:
# https://github.com/Kamil-Sabbagh/Data-and-Knowledge-Representation-Fall-25

# The assignment is on branch: assignment/combined-f25
# To merge to main:
git checkout main
git merge assignment/combined-f25
git push origin main
```

2. **Set Up Private Grader Repository** (Recommended for Production)

```bash
# Create a new private repository
gh repo create dkr-autograder-private --private

# Move grader files there
cd grader_private
git init
git add .
git commit -m "Initial grader implementation"
git remote add origin https://github.com/YOUR_ORG/dkr-autograder-private.git
git push -u origin main
```

3. **Configure GitHub Secrets**

Go to repository Settings → Secrets and variables → Actions:

- **`GRADER_TOKEN`**: Fine-grained PAT with read access to private grader repo
  - Generate at: https://github.com/settings/tokens
  - Permissions: `Contents: Read` on private grader repo
  - Add to public repo secrets

4. **Update Workflow to Use Private Grader**

Edit `.github/workflows/grade.yml`, replace:

```yaml
# - name: Fetch private grader
#   run: |
#     git clone https://${{ secrets.GRADER_TOKEN }}@github.com/YOUR_ORG/dkr-autograder-private.git grader_private
```

### Testing the Autograder

**Dry Run with Solution:**

1. Create a test branch with the solution:

```bash
cd assignment/starter
# Copy solution files from grader_private/solution_model.py
# Implement train.py and run_experiment.py
```

2. Open a PR from your test branch
3. Verify:
   - Workflow runs successfully
   - Grade comment appears
   - Attempt label is added
   - Score is 100/100

**Test Failure Cases:**

1. Test with incomplete implementation:
   - Should receive partial credit
   - Feedback should indicate failing tests

2. Test attempt limit:
   - Make 3 consecutive pushes to same PR
   - 4th push should trigger "exhausted" message

---

## Maintenance & Operations

### Common Tasks

**1. Reviewing Submissions**

```bash
# List all student PRs
gh pr list --label submission

# View specific PR
gh pr view 123

# Check CI logs
gh run view <run-id>
```

**2. Granting Override**

```bash
# Allow additional attempts
gh pr edit 123 --add-label override
```

**3. Updating Tests**

Edit `grader_private/test_grader.py`:

```python
def test_new_feature():
    """Test description (5 points)"""
    # Add test logic
    pass
```

Update `test_scores` dictionary with point values.

**4. Changing Performance Threshold**

Edit `test_performance_threshold` in `test_grader.py`:

```python
assert accuracy >= 0.76, f"Test accuracy {accuracy:.4f} below threshold 0.76"
#                  ^^^^  Change this value
```

### Monitoring

**Check Workflow Health:**

```bash
# View recent workflow runs
gh run list --workflow=grade.yml

# View failed runs
gh run list --workflow=grade.yml --status=failure
```

**Track Submission Stats:**

```python
# Script to analyze PR data
import requests

def get_submission_stats():
    prs = requests.get('https://api.github.com/repos/Kamil-Sabbagh/Data-and-Knowledge-Representation-Fall-25/pulls?state=all').json()

    total = len(prs)
    by_attempts = {1: 0, 2: 0, 3: 0}

    for pr in prs:
        labels = [l['name'] for l in pr['labels']]
        attempts = max([int(l.split('-')[1]) for l in labels if l.startswith('attempt-')], default=0)
        if attempts in by_attempts:
            by_attempts[attempts] += 1

    print(f"Total submissions: {total}")
    print(f"Distribution by attempts: {by_attempts}")

get_submission_stats()
```

---

## Troubleshooting

### Common Issues

**Problem:** Workflow fails with "Permission denied"
**Solution:** Check that `GRADER_TOKEN` is set correctly and has read access

**Problem:** Tests timeout on performance check
**Solution:** Increase timeout in workflow (currently 600s) or reduce epochs in test

**Problem:** Student complains about incorrect grading
**Solution:**
1. Check CI logs for actual error
2. Run grader locally on their submission
3. Verify test logic is correct
4. Grant override if warranted

**Problem:** Grader doesn't run on PR
**Solution:**
- Ensure PR targets `main` branch
- Check that workflow file is on `main` branch
- Verify PR is from a fork (not internal branch)

### Debugging Workflow

**Run grader locally:**

```bash
# Clone student submission
git clone <fork-url>
cd <fork-name>
git checkout <student-branch>

# Set up environment
cd assignment/starter
pip install -r requirements.txt

# Run grader
cd ../../grader_private
python test_grader.py
```

**View detailed logs:**

```bash
# Download workflow artifacts
gh run download <run-id>

# Or view logs directly
gh run view <run-id> --log
```

---

## Security Considerations

### What's Protected

- ✅ Reference solution (in private repo)
- ✅ Test cases and grading logic
- ✅ Grader token (in GitHub Secrets)

### What's Public

- Assignment brief and requirements
- Starter code templates
- Public smoke tests
- Grading rubric (point breakdown)
- Example outputs

### Best Practices

1. **Never commit secrets** to public repo
2. **Mask sensitive output** in workflow logs
3. **Limit token permissions** to read-only on private repo
4. **Review PRs** before merging to main
5. **Monitor for suspicious activity** (mass PR spam, etc.)

---

## Extensions & Customization

### Adding More Tests

1. Write new test in `test_grader.py`:

```python
def test_custom_feature(self):
    """Test custom feature (N points)"""
    # Test logic
    assert condition, "Failure message"
```

2. Add to `test_scores` dictionary:

```python
'test_custom_feature': 5,
```

3. Update max score if needed

### Changing Attempt Limit

Edit `.github/workflows/grade.yml`:

```yaml
# Line ~15
const exhausted = attempts >= 3;  # Change 3 to desired limit
```

### Adding Pre-Commit Checks

Create `.github/workflows/lint.yml`:

```yaml
name: Code Quality
on: pull_request

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: |
          pip install flake8 black
          flake8 assignment/starter/
          black --check assignment/starter/
```

---

## FAQ

**Q: Can students see the grader code?**
A: No, if it's in a private repository. The workflow fetches it using a secret token.

**Q: What if a student needs more than 3 attempts?**
A: Add the `override` label to their PR. The workflow will allow additional runs.

**Q: How do I update the assignment after students have started?**
A: Careful! Changes to test logic will affect fairness. If necessary:
1. Announce the change clearly
2. Re-run grading for all existing PRs
3. Allow appeals

**Q: Can students game the system?**
A: Unlikely. They can't see the grader code, and the performance threshold requires actual learning.

**Q: What's the cost of running CI?**
A: GitHub Actions is free for public repos (2000 minutes/month). This assignment uses ~5 min per run.

---

## Contact & Support

**Repository:** https://github.com/Kamil-Sabbagh/Data-and-Knowledge-Representation-Fall-25
**Issues:** Use GitHub Issues for bug reports
**Instructor Email:** [TBD]

---

## Appendix: File Manifest

### Public Files (in repo)
```
.github/workflows/grade.yml          # CI workflow
.gitignore                           # Git ignore rules
README.md                            # Student documentation
assignment/
  DKR_HW_Combined_F25.md             # Assignment brief (MD)
  starter/
    __init__.py
    data_utils.py                    # Dataset helpers
    model.py                         # GAT templates
    train.py                         # Training templates
    run_experiment.py                # Experiment CLI
    requirements.txt                 # Dependencies
    tests_public/
      test_imports.py                # Smoke tests
```

### Private Files (should be in separate repo)
```
grader_private/
  solution_model.py                  # Reference solution
  test_grader.py                     # Full test suite
  requirements.txt                   # Grader dependencies
```

---

**Version:** 1.0
**Last Updated:** October 2025
**Author:** Course TA Team
