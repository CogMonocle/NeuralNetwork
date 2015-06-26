__author__ = 'Justin Niosi'

import numpy as np


def sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))


def sigmoid_derivative(t):
    # return np.exp(t) / np.square(np.exp(t) + 1)
    return sigmoid(t) * (1 - sigmoid(t))


def quadratic_cost(goal, actual):
    return np.sum(np.square(np.linalg.norm(goal - actual))) / (2 * len(goal))


def quadratic_cost_derivative(goal, actual):
    return actual[:len(goal)] - goal


class NeuralNetwork:

    def __init__(self, layers, max_neurons, rate, out_func=sigmoid, out_derivative=sigmoid_derivative,
                 cost_func=quadratic_cost, cost_derivative=quadratic_cost_derivative):
        self.layers = layers
        self.maxneurons = max_neurons
        self.rate = rate
        self.weights = np.zeros((self.layers, self.maxneurons, self.maxneurons))
        self.biases = np.zeros((self.layers, self.maxneurons))
        self.outfunc = out_func
        self.outderivative = out_derivative
        self.costfunc = cost_func
        self.costderivative = cost_derivative

    def feed_forward(self, inputs):
        values = np.zeros((self.layers, self.maxneurons))
        z_values = np.zeros((self.layers, self.maxneurons))
        values[0, :len(inputs)] = inputs
        for i in xrange(1, self.layers):
            z_values[i] = np.dot(values[i-1], self.weights[i])
            values[i] = self.outfunc(z_values[i] + self.biases[i])
        return np.array([values, z_values])

    def cost(self, goal, actual):
        return self.costfunc(goal, actual)

    def output_error(self, goal, actual, z_values):
        return self.costderivative(goal, actual) * self.outderivative(z_values[:len(goal)])

    def backpropagate(self, goal, feed_forward_output):
        output = np.zeros(self.biases.shape)
        cost_derivative = self.costderivative(goal, feed_forward_output[0][-1])
        output_derivative = self.outderivative(feed_forward_output[1][-1])
        output_error = cost_derivative * output_derivative
        output[-1] = output_error
        for i in xrange(self.layers - 1, 0, -1):
            weight_times_error = np.dot(self.weights[i].transpose(), output[i])
            out_derivative_z = self.outderivative(feed_forward_output[1][i - 1])
            output[i - 1] = np.dot(weight_times_error, out_derivative_z)
        return output

    def update_parameters(self, test_output):
        values, error = test_output
        self.biases -= self.rate * error
        values = np.roll(values, 1, axis=0).repeat(self.maxneurons, axis=0).reshape(self.weights.shape).transpose(0, 2, 1)
        error = error.repeat(self.maxneurons, axis=0).reshape(self.weights.shape)
        self.weights -= self.rate * (error + values)

    def run_test(self, input_values, goal):
        results = self.feed_forward(input_values)
        gradient = self.backpropagate(goal, results)
        return np.array([results, gradient])
