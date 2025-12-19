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
