# Linear Regression from Scratch (NumPy Only)

A clean, minimal implementation of **Ordinary Least Squares (OLS) Linear Regression** with optional **L2 (Ridge) Regularisation**, built entirely using NumPy.

---

## Table of Contents

- [Overview](#overview)
- [Math Behind It](#math-behind-it)
  - [Ordinary Least Squares](#ordinary-least-squares)
  - [L2 Regularisation (Ridge)](#l2-regularisation-ridge)
  - [Intercept via Mean-Centering](#intercept-via-mean-centering)
- [Class Reference](#class-reference)
  - [Parameters](#parameters)
  - [Attributes](#attributes)
  - [Methods](#methods)
- [Usage Examples](#usage-examples)
  - [Simple Linear Regression](#simple-linear-regression)
  - [Multiple Linear Regression](#multiple-linear-regression)
  - [Ridge Regression (L2)](#ridge-regression-l2)
  - [Polynomial Regression](#polynomial-regression)
- [Why Mean-Centering?](#why-mean-centering)
- [Why `pinv` instead of `inv`?](#why-pinv-instead-of-inv)

---

## Overview

This implementation uses the **closed-form (analytical) solution** to find optimal regression weights — meaning no gradient descent or iterative optimisation is needed. The solution is computed directly using matrix operations.

```
theta = (XᵀX)⁻¹ Xᵀy          ← OLS
theta = (XᵀX + αI)⁻¹ Xᵀy     ← Ridge (L2)
```

---

## Math Behind It

### Ordinary Least Squares

We want to find a weight vector **θ** that minimises the residual sum of squares:

```
Loss(θ) = ||y - Xθ||²
```

Setting the gradient to zero and solving gives the closed-form solution:

```
θ = (XᵀX)⁻¹ Xᵀy
```

### L2 Regularisation (Ridge)

Ridge regression adds a penalty term `α||θ||²` to the loss to prevent overfitting and stabilise the solution when features are correlated (multicollinearity). The modified closed-form becomes:

```
θ = (XᵀX + αI)⁻¹ Xᵀy
```

Where:
- `α` (alpha) controls regularisation strength — larger values shrink coefficients more aggressively
- `I` is the identity matrix of size `(n_features × n_features)`

### Intercept via Mean-Centering

Instead of appending a bias column of ones to `X`, this implementation uses **mean-centering** to separate the intercept calculation cleanly:

```
X_centered = X - X_mean
y_centered = y - y_mean

θ  = solve on centered data
b  = y_mean - X_mean @ θ
```

This is mathematically equivalent to including a bias term and is numerically cleaner.

---

## Class Reference

```python
class Linear_Regression:
    def __init__(self, regularisation_l2=False, alpha=1.0)
    def fit(self, X_, y_) -> self
    def predict(self, X) -> np.ndarray
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `regularisation_l2` | `bool` | `False` | Enable L2 (Ridge) regularisation |
| `alpha` | `float` | `1.0` | Regularisation strength. Only used when `regularisation_l2=True`. Higher = stronger penalty |

### Attributes

After calling `.fit()`, the following attributes are available:

| Attribute | Type | Description |
|---|---|---|
| `coeff` | `np.ndarray`, shape `(n_features,)` | Learned regression coefficients (weights) |
| `intercept` | `float` or `np.ndarray` | Learned bias/intercept term |

### Methods

#### `fit(X_, y_)`

Trains the model using the closed-form solution.

- `X_` : array-like of shape `(n_samples, n_features)` — Feature matrix
- `y_` : array-like of shape `(n_samples,)` or `(n_samples, n_outputs)` — Target values
- Returns `self` (for method chaining)

#### `predict(X)`

Generates predictions for new input data.

- `X` : array-like of shape `(n_samples, n_features)`
- Returns `np.ndarray` of predictions

---

## Usage Examples

### Setup

```python
import numpy as np

# Paste or import the Linear_Regression class here
```

---

### Simple Linear Regression

```python
# Single feature: y = 3x + 5
X = np.array([[1], [2], [3], [4], [5]], dtype=float)
y = np.array([8, 11, 14, 17, 20], dtype=float)

model = Linear_Regression()
model.fit(X, y)

print("Coefficient :", model.coeff)       # [3.]
print("Intercept   :", model.intercept)   # 5.

predictions = model.predict(np.array([[6], [7]]))
print("Predictions :", predictions)       # [23. 26.]
```

---

### Multiple Linear Regression

```python
# Two features
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6]
], dtype=float)
y = np.array([7, 10, 13, 16, 19], dtype=float)

model = Linear_Regression()
model.fit(X, y)

print("Coefficients:", model.coeff)
print("Intercept   :", model.intercept)
print("Predictions :", model.predict(X))
```

---

### Ridge Regression (L2)

Use when features are correlated or when the model overfits.

```python
np.random.seed(42)
X = np.random.randn(50, 5)
y = X @ np.array([1.5, -2.0, 0.5, 3.0, -1.0]) + np.random.randn(50) * 0.5

# Without regularisation
ols = Linear_Regression(regularisation_l2=False)
ols.fit(X, y)

# With L2 regularisation (alpha controls strength)
ridge = Linear_Regression(regularisation_l2=True, alpha=0.5)
ridge.fit(X, y)

print("OLS coeffs  :", ols.coeff)
print("Ridge coeffs:", ridge.coeff)  # Coefficients are shrunk toward zero
```

**Choosing `alpha`:** Start with `alpha=1.0` and tune using cross-validation. Higher values increase bias but reduce variance.

---

### Polynomial Regression

The `Linear_Regression` class can perform **polynomial regression** by manually engineering polynomial features before fitting. The model itself stays linear in the *transformed* feature space.

Even though this class is a linear model, it can fit non-linear data through feature engineering. The idea is to manually transform the raw input x into a matrix of polynomial terms — [x, x², x³, ..., xᵈ] — and pass that as the feature matrix X to .fit(). The model then treats each power as an independent feature and learns a coefficient for each, effectively fitting a polynomial curve while the underlying math stays identical.


**Higher degree example:**

```python
X_poly # Design Matrix having Polynomial features

# Add Ridge regularisation to avoid overfitting with high-degree features
model_ridge = Linear_Regression(regularisation_l2=True, alpha=0.1)
model_ridge.fit(X_poly4, y)

y_pred4 = model_ridge.predict(X_poly4)
```

For higher degrees (d ≥ 4), it is recommended to enable Ridge regularisation (regularisation_l2=True) to prevent overfitting, since high-degree polynomial features tend to cause the model to chase noise in the training data.

---

## Why Mean-Centering?

Mean-centering removes the need to augment `X` with a column of ones to learn the intercept. It is:

- **Numerically stable** — keeps the feature matrix well-conditioned
- **Mathematically equivalent** — produces the same result as the augmented approach
- **Cleaner separation** — coefficients and intercept are learned independently

---

## Why `pinv` instead of `inv`?

`np.linalg.pinv` (Moore-Penrose pseudoinverse) is used instead of `np.linalg.inv` because:

- It handles **rank-deficient** or **singular** matrices gracefully (e.g., perfectly collinear features)
- It works even when `XᵀX` is not invertible
- It is strictly more general — for full-rank matrices, it gives the same result as `inv`

---

## Dependencies

```
numpy
```

---

## HaHa

Completly enjoyed it
