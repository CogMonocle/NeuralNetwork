__author__ = 'Justin Niosi'

import sys
import getopt
import network
import network2
import numpy as np


def goal_function(inputs):
    total = np.sum(inputs)
    output = np.zeros(3)
    if total > 20:
        output[0] = 1
    elif total < -20:
        output[1] = 1
    else:
        output[2] = 1
    return output

def test_network():
    layers = 3
    max_neurons = 3
    learning_rate = 0.1
    weights = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                       [[2, -3, 4], [-5, 3, 1], [4, -6, 5]],
                       [[7, -8, 4], [2, 8, 15], [7, 2, 8]]])
    biases = np.array([[0, 0, 0], [3, -1, 10], [5, 20, 6]])
    inputs = np.array([1, 2, 3])
    goal = goal_function(inputs)
    neural_network = network.NeuralNetwork(layers, max_neurons, learning_rate)
    neural_network.weights = weights.transpose((0, 2, 1))
    neural_network.biases = biases
    desired_values = np.zeros((layers, max_neurons))
    desired_z_values = np.zeros(desired_values.shape)
    desired_values[0] = inputs
    desired_z_values[1] = np.array([np.dot(desired_values[0], weights[1][0]),
                                    np.dot(desired_values[0], weights[1][1]),
                                    np.dot(desired_values[0], weights[1][2])])
    desired_values[1] = np.array(neural_network.outFunc(desired_z_values[1] - biases[1]))
    desired_z_values[2] = np.array([np.dot(desired_values[1], weights[2][0]),
                                    np.dot(desired_values[1], weights[2][1]),
                                    np.dot(desired_values[1], weights[2][2])])
    desired_values[2] = np.array(neural_network.outFunc(desired_z_values[2] - biases[2]))
    feed_forward_results = neural_network.feed_forward(inputs)
    print("z value test: " + str(np.allclose(desired_z_values, feed_forward_results[1])))
    print("a value test: " + str(np.allclose(desired_values, feed_forward_results[0])))
    cost = neural_network.costFunc(goal, feed_forward_results[0][layers - 1])
    cost_derivative = neural_network.costDerivative(goal, feed_forward_results[0][layers - 1])
    output_derivative = neural_network.outDerivative(feed_forward_results[1][layers - 1])
    output_error = np.multiply(cost_derivative, output_derivative)
    print(goal)
    print(desired_values[layers - 1])
    print(cost)
    print(cost_derivative)
    print(output_derivative)
    print(output_error)
    return neural_network

def weight_update_test():
    neural_network = network.NeuralNetwork(3, 3, 1)
    val = np.array(range(9)).reshape((3, 3))
    err = np.array(range(9)).reshape((3, 3)) * 10
    print(val)
    print(err)
    neural_network.update_parameters([val, err])
    print(neural_network.weights)

def learning_test(layers, max_neurons, rate, test_size, test_num):
    neural_network = network.NeuralNetwork(layers, max_neurons, rate)
    neural_network.weights = np.random.random_sample(neural_network.weights.shape)
    neural_network.biases = np.random.random_sample(neural_network.biases.shape)
    for i in xrange(test_num):
        total_output = np.zeros(neural_network.biases.shape)
        total_gradient = np.zeros(neural_network.biases.shape)
        for j in range(test_size):
            inputs = np.random.random_sample(max_neurons) * 20
            results = neural_network.run_test(inputs, goal_function(inputs))
            total_output += results[0][1]
            total_gradient += results[1]
        neural_network.update_parameters([total_output / test_size, total_gradient / test_size])
    print(neural_network.feed_forward([2, 2, 10])[0])

def learning_test2(shape, epochs, mini_batch_size, rate, data):
    training_data, validation_data, test_data = data
    net = network2.NeuralNetwork(shape)
    net.SGD(training_data, epochs, mini_batch_size, rate, test_data=test_data)

def main(argv):
    layers = 5
    max_neurons = 3
    rate = 0.5
    test_size = 1000
    test_num = 10
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
    #test_network()
    #weight_update_test()
    #learning_test(layers, max_neurons, rate, test_size, test_num)
    training_size = 5000
    validation_size = 1000
    test_size = 1000
    mini_batch_size = 100
    num_epochs = training_size / mini_batch_size
    shape = [3, 3, 3]
    training_data = ([([np.reshape(x, (len(x), 1)), goal_function(x)]) for x in np.random.randn(training_size, 3) * 20])
    validation_data = ([([np.reshape(x, (len(x), 1)), goal_function(x)]) for x in np.random.randn(validation_size, 3) * 20])
    test_data = ([([np.reshape(x, (len(x), 1)), goal_function(x)]) for x in np.random.randn(test_size, 3) * 20])
    data_array = [training_data, validation_data, test_data]
    #print(np.array(data_array))
    learning_test2(shape, num_epochs, mini_batch_size, rate, data_array)


if __name__ == "__main__":
    main(sys.argv[1:])
