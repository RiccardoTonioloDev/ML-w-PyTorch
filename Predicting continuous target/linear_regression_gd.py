import numpy as np


class LinearRegressionGD:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.array([0.0])
        self.losses_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = y - output
            self.w_ += (
                self.eta  # learning rate
                * 2.0  # constant multiplier
                * X.T.dot(errors)
                / X.shape[0]  # average of the errors scaled by their specific weight
            )
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()  # Mean Squared Error
            self.losses_.append(loss)
        return self

    def predict(self, X):
        return self.net_input(X)
