import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def int_to_onehot(y, num_labels):
    ary = np.zeros((y.shape[0], num_labels))

    for i, val in enumerate(y):
        ary[i, val] = 1

    return ary


class NeuralNetMLP:
    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        super().__init__()

        self.num_classes = num_classes

        rng = np.random.RandomState(random_seed)

        self.weight_h = rng.normal(loc=0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)

        self.weight_out = rng.normal(loc=0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)

    def forward(self, x):
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)

        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)

        return (
            a_h,  # We need the hidden layer activations in order to know how to optimize (in the backprop) the weight of the hidden layer
            a_out,  # The final class porbability prediction
        )

    def backward(self, x, a_h, a_out, y):
        # Here we are supposing that the loss function is the MSE

        y_onehot = int_to_onehot(y, self.num_classes)

        d_loss__d_a_out = (
            2.0 * (a_out - y_onehot) / y.shape[0]
        )  # derivative of L w.r.t. the output activations
        d_a_out__d_z_out = a_out * (1.0 - a_out)  # sigmoid derivative
        delta_out = (
            d_loss__d_a_out * d_a_out__d_z_out
        )  # chain rule: derivative of L w.r.t. the output layer (before the activations)

        d_z_out_dw_out = a_h  # derivative of the the output layer (before the activation) w.r.t. the output weights
        d_loss__dw_out = np.dot(
            delta_out.T,  # the error signal coming from the output computation
            d_z_out_dw_out,
        )  # chain rule: derivative of L w.r.t. the weights of the output

        #####################
        #        NOTE       #
        #####################
        # With the derivative of the loss w.r.t. the weights of the output, we want to calculate a matrix that has for each
        # element of the weight matrix between the hidden layer and the output layer, the single derivative of each of those
        # weights.
        # The dot product makes sense for two reasons:
        # - We achieve that derivative matrix with the right dimensions;
        # - We sum the contributions of each sample (remember that each original contribution is scaled by the number of
        # samples, look at row 49) in the batch to adjust each weight correctly.

        d_loss__db_out = np.sum(
            delta_out, axis=0
        )  # chain rule: derivative of L w.r.t. the output bias units (the derivative with respect to the bias unit is 1)

        d_z_out__a_h = (
            self.weight_out
        )  # derivative of the output layer w.r.t. the activation of the hidden layer
        d_loss__a_h = np.dot(
            delta_out, d_z_out__a_h
        )  # chain rule: derivative of L w.r.t. the activation of the hidden layer
        d_a_h__d_z_h = a_h * (
            1.0 - a_h
        )  # derivative of the sigmoid for the hidden layer
        d_z_h__d_w_h = x  # derivative of the hidden layer pre activation w.r.t. the weights of the hidden layer
        d_loss__d_w_h = np.dot(
            (d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h
        )  # the gradient of L w.r.t. the hidden layer weights
        d_loss__d_b_h = np.sum(
            (d_loss__a_h * d_a_h__d_z_h), axis=0
        )  # chain rule: derivative of L w.r.t. the hidden bias units

        return (d_loss__dw_out, d_loss__db_out, d_loss__d_w_h, d_loss__d_b_h)
