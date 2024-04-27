import numpy as np


class Perceptron:
    """
    Parameters
        eta: float
            Learning rate
        n_iter: int
            Passes over training dataset.
        random_state: int
            Random number generator seed for random weight initialization.
    Attributes
        w_: 1d-array
        b_: scalar
        errors_: list
            Number of misclassifications (updates in each epoch).
    """

    def __init__(self, eta: float = 0.01, n_iter: int = 50, random_state: int = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def net_input(self, X: np.ndarray) -> np.ndarray:
        """
        It uses the weights and biases to generate the prediction
        """
        return np.dot(X, self.w_) + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        It uses the prediction, with an activation function to forecast labels
        """
        return np.where(self.net_input(X) >= 0.0, 1, 0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """
        Parameters
            X: matrix, shape = [n_examples, n_features]
            y: vector, shape = [n_examples]
        """
        rgen = np.random.RandomState(self.random_state)  # seed usage for rgen creation
        self.w_ = rgen.normal(
            loc=0.0, scale=0.01, size=X.shape[1]
        )  # number of wheights corresponding to the number of features

        self.b_ = np.float_(0.0)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for x_i, target in zip(X, y):
                update = self.eta * (target - self.predict(x_i))
                self.w_ += (
                    update * x_i
                )  # the update it's applied at the same time on each weight
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
