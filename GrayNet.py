import sys
import numpy as np
import math


class GrayNet:
    def __init__(self, layers, learning_rate=0.1):
        self.layers = layers
        self.learning_rate = learning_rate

        # initialize weights between all layers
        # except the last two
        weights = []
        for i in range(0, len(layers) - 2):
            # weight matrix
            # N rows and M columns
            # another weight for the bias (cool trick!)
            n = layers[i] + 1
            m = layers[i + 1] + 1
            layer_weights = np.random.randn(n, m)
            # avoid "vanishing gradient"
            layer_weights = layer_weights / math.sqrt(layers[i])
            weights.append(layer_weights)

        # handle weights between
        # the last two layers
        n = layers[-2] + 1
        m = layers[-1]
        layer_weights = np.random.randn(n, m)
        # avoid "vanishing gradient"
        layer_weights = layer_weights / math.sqrt(layers[-2])
        weights.append(layer_weights)

        self.weights = weights

    def activation(self, x):
        return 1 / (1 + np.exp(-x))

    def activation_derivative(self, x):
        # here, x is actually activation(x)
        return x * (1 - x)

        # i'd like to figure out how to use the real derivative without this pre-computed x thing

    def sse(self, x, t):
        # y needs to be a matrix with one row
        # i think matlab will handle this fine
        t = np.atleast_2d(t)
        p = self.predict(x)
        # sum squared error
        return np.sum((p - t) ** 2) / 2

    def train(self, x, t, epochs=1000):
        # insert 1's as last entry for bias
        # lets see how to do this in matlab later
        x = np.c_[x, np.ones((x.shape[0]))]
        print(x)

        # keep track of sum squared errors so you can graph it later
        sses = []

        for _ in range(0, epochs):
            for (inp, target) in zip(x, t):
                self.fit(inp, target)

            sses.append(self.sse(x, t))

        # return the sum squared error at each epoch
        return sses

    def predict(self, x):
        # needs to be a matrix, again
        x = np.atleast_2d(x)

        for i in range(0, len(self.weights)):
            x = self.activation(np.dot(x, self.weights[i]))

        sys.exit(0)
        return x

    def fit(self, x, t):
        ################
        # FORWARD PASS #
        ################

        # we will remember the outputs of each layer
        # the first outputs are just the inputs themselves
        outputs = []
        outputs.append(np.atleast_2d(x))

        # iterate over layers (actually, the weights)
        for i in range(0, len(self.weights)):
            # sum of a bunch of products?
            # thats just the dot product!
            net = outputs[i].dot(self.weights[i])

            # apply the activation function to the raw output of the weights
            net = self.activation(net)

            # remember it
            outputs.append(net)

        ###################
        # BACKPROPAGATION #
        ###################

        cost = outputs[-1] - t

        # these deltas go from last layer to first layer
        # but they appear in this list left to right
        # a little confusing
        deltas = []

        # actual chain rule here ->

        # the first deltas are just
        # the cost times the derivative of
        # the activation function evaluated at
        # the outputs of the final layer
        deltas.append(
            cost * self.activation_derivative(
                outputs[-1])
        )

        # the subsequent deltas depend on the most previously computer delta
        for i in range(len(outputs) - 2, 0, -1):
            delta = deltas[-1].dot(self.weights[i].T) * \
                self.activation_derivative(outputs[i])
            deltas.append(delta)

        # <- end chain rule

        # since the deltas are in reverse order, flip it
        deltas = deltas[::-1]

        # update those weights
        for i in range(0, len(self.weights)):
            self.weights[i] += -self.learning_rate * \
                outputs[i].transpose().dot(deltas[i])
