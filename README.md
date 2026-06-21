# Supervised ML — From Scratch

A growing collection of supervised machine learning algorithms implemented from scratch — no high-level ML libraries, just math translated directly into code.

The goal isn't to reinvent the wheel. It's to actually understand what's inside it.

---

## Why From Scratch?

Calling `.fit()` is easy. Knowing *why* it works is the part that actually makes you better.

Every algorithm here is built by going back to the math — the loss functions, the linear algebra, the optimisation — and turning it into working code step by step. No black boxes.

---

## Repository Structure

```
Supervised_ML/
│
├── LinearRegression/
│   ├── linear_regression.py
│   └── README.md
│
└── ... 
```

Each algorithm lives in its own folder with its own implementation file and a dedicated README covering the math, class reference, and usage examples.

---

## Algorithms

### ✅ Implemented

| Algorithm | Type | Key Concepts | Folder |
|---|---|---|---|
| Linear Regression | Regression | Normal equation, Ridge (L2), Mean-centering, Pseudoinverse | [`LinearRegression/`](./LinearRegression/) |


---

## Current Implementation — Linear Regression

The first algorithm in the repo is a clean implementation of **Ordinary Least Squares Linear Regression** with optional **L2 (Ridge) Regularisation**.

Rather than using gradient descent, it solves directly for the optimal weights using the closed-form normal equation:

```
θ = (XᵀX)⁻¹ Xᵀy          ← OLS
θ = (XᵀX + αI)⁻¹ Xᵀy     ← Ridge (L2)
```

A few deliberate design choices in the implementation:

- **Mean-centering** instead of appending a bias column — the intercept falls out naturally, and the feature matrix stays cleaner
- **Moore-Penrose pseudoinverse** (`pinv`) instead of regular matrix inverse — handles correlated or rank-deficient features without breaking
- **Ridge regularisation** as an optional flag — adds α·I to XᵀX before inverting, stabilising the solution on noisy or collinear data
- **Polynomial regression support** — pass engineered polynomial features `[x, x², ..., xᵈ]` as input and the same class handles it out of the box

Validated on a 4000-sample regression dataset — R² of **0.9959**, matching scikit-learn's result exactly.

→ Full details in [`LinearRegression/README.md`](./LinearRegression/README.md)

---

## Dependencies

All implementations use only:

```
numpy
```
