__author__ = 'Justin Niosi'

import numpy as np

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def sigmoid_derivative(t):
    return np.exp(t) / np.square(np.exp(t) - 1)

def quadratic_cost(goal, actual):
    return np.sum(np.square(np.linalg.norm(goal - actual, axis=1))) / (2 * len(goal))

def quadratic_cost_derivative(goal, actual):
    return actual - goal

class NeuralNetwork:

    def __init__(self, layers, max_neurons, rate, out_func=sigmoid, out_derivative=sigmoid_derivative,
                 cost_func=quadratic_cost, cost_derivative=quadratic_cost_derivative):
        self.layers = layers
        self.maxNeurons = max_neurons
        self.rate = rate
        self.weights = np.zeros((self.layers, self.maxNeurons, self.maxNeurons))
        self.biases = np.zeros((self.layers, self.maxNeurons))
        self.outFunc = out_func
        self.outDerivative = out_derivative
        self.costFunc = cost_func
        self.costDerivative = cost_derivative

    def feed_forward(self, inputs):
        values = np.zeros((self.layers, self.maxNeurons, 3))
        z_values = np.zeros((self.layers, self.maxNeurons))
        values[0] = np.expand_dims(inputs, 1)
        for i in xrange(1, self.layers):
            z_values[i] = np.sum(np.multiply(values[i-1], self.weights[i]), axis=0)
            values[i] = np.expand_dims(self.outFunc(z_values[i] - self.biases[i]), 1)
        return np.array([np.transpose(values, axes=(2, 0, 1))[0], z_values])

    def cost(self, goal, actual):
        return self.costFunc(goal, actual)

    def output_error(self, goal, actual, z_values):
        return self.costDerivative(goal, actual) * self.outDerivative(z_values)

    def error_from_next_layer(self, next_error, next_weights, z_values):
        return np.multiply(next_weights * next_error, z_values)

    def weight_cost_derivatives(self, goal, values, z_values):
        o_error = self.output_error(goal, values[self.layers - 1], z_values[self.layers - 1])
        error = np.zeros(self.biases.shape)
        error[self.layers - 1] = o_error
        for i in xrange(self.layers - 1, 0, -1):  # TODO:double check that this works correctly
            error[i] = self.error_from_next_layer(error[i+1], self.weights[i+1], z_values[i + 1])
        return np.multiply(error, values)

    def run_test(self, input_values, goal):
        results = self.feed_forward(input_values)
        gradient = self.weight_cost_derivatives(goal, results[0], results[1])
        return np.array([results, gradient])
