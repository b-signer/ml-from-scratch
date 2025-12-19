import numpy as np


class LogisticRegressionL1:
    """
    Binary logistic regression with L1 regularization.
    Implemented from scratch using NumPy only.
    """

    def __init__(self, n_features, reg_strength=0.0):
        """
        Parameters
        ----------
        n_features : int
            Number of input features.
        reg_strength : float
            L1 regularization coefficient (lambda).
        """
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lambda_ = reg_strength

    @staticmethod
    def sigmoid(z):
        """
        Numerically stable sigmoid.
        """
        return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(self, X):
        """
        Compute predicted probabilities P(y=1 | x).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        probs : ndarray of shape (n_samples,)
        """
        logits = X @ self.w + self.b
        return self.sigmoid(logits)

    def compute_loss(self, X, y):
        """
        Compute binary cross-entropy loss with L1 regularization.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)

        Returns
        -------
        loss : float
        """
        n = X.shape[0]
        probs = self.predict_proba(X)

        # Avoid log(0)
        eps = 1e-12
        probs = np.clip(probs, eps, 1 - eps)

        data_loss = -np.mean(
            y * np.log(probs) + (1 - y) * np.log(1 - probs)
        )

        reg_loss = self.lambda_ * np.linalg.norm(self.w, ord=1)

        return data_loss + reg_loss

    def compute_gradients(self, X, y):
        """
        Compute gradients of the loss w.r.t. parameters.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)

        Returns
        -------
        grad_w : ndarray of shape (n_features,)
        grad_b : float
        """
        n = X.shape[0]
        probs = self.predict_proba(X)

        error = probs - y

        grad_w_data = (1.0 / n) * (X.T @ error)
        grad_b = (1.0 / n) * np.sum(error)

        grad_w_reg = self.lambda_ * np.sign(self.w)

        grad_w = grad_w_data + grad_w_reg

        return grad_w, grad_b
