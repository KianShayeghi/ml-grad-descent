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

### 2) Regularised Least Squares
- Reads `A.csv` and `b.csv` via **pandas**
- Converts them to **PyTorch tensors**
- Creates a random train/test split using `torch.randperm`
- Sets up the workflow for testing different regularisation strengths `λ` and plotting test error vs `λ`

---

## Repository structure

    .
    ├── ml_optimisation.ipynb
    ├── A.csv
    ├── b.csv
    └── .gitignore

- **ml_optimisation.ipynb** – main notebook with gradient descent and regularised least squares setup  
- **A.csv** – design matrix for the regularised least squares experiment  
- **b.csv** – target vector  
- **.gitignore** – keeps environments, caches, and generated files out of version control

---

## Requirements

The notebook imports the following:
- Python 3.x
- numpy
- pandas
- matplotlib
- torch
- scikit-learn
- jupyter

Quick install:
```bash
pip install numpy pandas matplotlib torch scikit-learn jupyter
```

---

How to run
```bash
git clone https://github.com/KianShayeghi/ml-grad-descent.git
cd ml-grad-descent
jupyter notebook
```
Open ```ml_optimisation.ipynb``` and run the cells.

---

Notes

There is a Google Colab upload cell (```from google.colab import files```) intended for uploading ```A.csv``` / ```b.csv``` in Colab.

If you’re running locally, you can ignore that cell as long as the CSVs are already in the repo directory.
