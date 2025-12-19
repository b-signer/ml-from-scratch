# Logistic Regression from Scratch (L1 Regularization)

This project implements binary logistic regression **from scratch** using NumPy only.
No machine learning libraries, autograd frameworks, or pretrained models are used.

The goal is to demonstrate a clear understanding of:
- probabilistic classification
- convex optimization
- gradient-based learning
- sparsity via L1 regularization

---

## Problem Setting

Given a dataset of feature vectors $\( x \in \mathbb{R}^d \)$ and binary labels
$\( y \in \{0,1\} \)$, we model the conditional probability:

$$
P(y=1 \mid x) = \sigma(w^\top x + b)
$$

where $\( \sigma(z) = \frac{1}{1 + e^{-z}} \)$ is the sigmoid function.

---

## Loss Function

We minimize the regularized empirical risk:

$$
\mathcal{L}(w, b) =
-\frac{1}{n} \sum_{i=1}^n
\left[
y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)
\right]
+ \lambda \|w\|_1$$


where $\( \lambda \)$ controls sparsity.

---

## Optimization

Parameters are optimized using batch gradient descent.
Gradients are derived analytically and implemented explicitly.
