# Machine Learning - Gradient Descent

A machine learning optimisation project focused on **gradient descent for linear regression** and a scaffold for **regularised least squares** experiments.

The main work is in `ml_optimisation.ipynb`.

---

## Notebook contents

### 1) Gradient Descent + Linear Regression (Diabetes dataset)
- Loads the **diabetes dataset** from `sklearn.datasets`
- Converts data to **PyTorch tensors**
- Sets up a train/test split
- Implements gradient descent training (explicit learning rate + epochs)
- Computes/compares errors against a closed-form least squares baseline (matrix form)

---

### 2) Regularised Least Squares (A.csv, b.csv)
- Reads `A.csv` and `b.csv` via **pandas**
- Converts them to **PyTorch tensors**
- Creates a random train/test split using `torch.randperm`
- Sets up the workflow for testing different regularisation strengths `λ` and plotting test error vs `λ`

> Some parts of this section are currently left as `...` placeholders (assignment scaffolding). The structure is there for implementing the full ridge/regularised solution and evaluation loop.

---

## Repository structure

