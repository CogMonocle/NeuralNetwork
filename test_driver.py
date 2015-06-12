__author__ = 'Justin Niosi'

import sys
import getopt
import network
import math
import numpy as np

def goal_function(x, y, z):
    return np.array([math.ln(x * y), 1 / (y * z), np.exp2(x * z)])

def test_network(layers, max_neurons, rate):
    neural_network = network.NeuralNetwork(layers, max_neurons, rate)
    neural_network.weights = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                       [[1, 2, 3], [4, 5, 6], [-1, -2, -3]],
                                       [[1, 2, 1], [2, 0, -1], [4, -2, 1]]])
    neural_network.biases = np.array([[0, 0, 0], [2, 1, -2], [3, 1, 0]])
    return neural_network

def main(argv):
    layers = 4
    max_neurons = 20
    rate = 1
    if len(argv) > 0:
        try:
            opts = getopt.getopt(argv, "l:m:r:")
        except getopt.GetoptError:
            print 'Network.py -n <network shape> -r <learning rate>'
            sys.exit(2)
        for opt, arg in opts:
            if opt == 'l':
                layers = arg
            if opt == 'm':
                max_neurons = arg
            if opt == 'r':
                rate = arg
    neural_network = test_network(layers, max_neurons, rate)
    goal_function_network = network.NeuralNetwork(layers, max_neurons, rate)
    weights = np.zeros((layers, max_neurons, max_neurons))
    weights[1, :, :3] = np.random.rand(20, 3)
    weights[2] = np.random.rand(20, 20)
    weights[3, :3] = np.random.rand(3, 20)
    goal_function_network.weights = weights
    biases = np.zeros(layers, max_neurons)
    biases[1] = np.random(20)
    biases[2] = np.random(20)
    biases[3] = np.random(3)
    goal_function_network.biases = biases

if __name__ == "__main__":
    main(sys.argv[1:])
