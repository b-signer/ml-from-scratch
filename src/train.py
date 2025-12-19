import numpy as np
import matplotlib.pyplot as plt
from model import LogisticRegressionL1

def main():
    np.random.seed(42)

    # Synthetic binary classification dataset
    n_samples = 200
    n_features = 5

    X = np.random.randn(n_samples, n_features)
    true_w = np.array([1.5, -2.0, 0.0, 0.0, 0.5])
    logits = X @ true_w
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(int)

    # Train model
    model = LogisticRegressionL1(n_features=n_features, reg_strength=0.1)
    losses = model.fit(X, y, lr=0.04, epochs=1000, verbose=True)

    # Plot loss
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.show()

if __name__ == "__main__":
    main()