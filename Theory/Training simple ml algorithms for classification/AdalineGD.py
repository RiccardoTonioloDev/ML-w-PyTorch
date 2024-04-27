import numpy as np


class AdalineGD:
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
        losses_: list
            Mean squared error loss function values in each epoch.
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
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

    def activation(self, X: np.ndarray) -> np.ndarray:
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdalineGD":
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
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self
