import numpy as np


class AdalineSGD:
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

    def __init__(
        self,
        eta: float = 0.01,
        n_iter: int = 50,
        shuffle: bool = True,
        random_state=None,
    ):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
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

    def _update_weights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w_ += self.eta * 2.0 * xi * (error)
        self.b_ += self.eta * 2.0 * (error)
        loss = error**2
        return loss

    def _initialize_weights(self, m: int):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float_(0.0)
        self.w_initialized = True

    def _shuffle(self, X: np.ndarray, y: np.ndarray):
        """Shuffles the training data (with corrisponding positions to the y)"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdalineSGD":
        """
        Parameters
            X: matrix, shape = [n_examples, n_features]
            y: vector, shape = [n_examples]
        """
        self._initialize_weights(X.shape[1])
        self.losses_ = []

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for xi, target in zip(X, y):
                losses.append(self._update_weights(xi, target))
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)
        return self
