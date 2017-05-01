"""
This module implements the neural network class and all components necessary to
modularize the build/use process of the neural network.
"""

import numpy as np
import pickle

class NNetwork:
    """
    The representation of a feed forward neural network with a bias in every
    layer (excluding output layer obviously).
    """

    def __init__(self, layer_sizes, activation_funcs, bias_neuron = False):
        """
        Creates a 'NNetwork'. 'layer_sizes' provides information about the
        number of neurons in each layer, as well as the total number of layers
        in the neural network. 'activation_funcs' provides information about the
        activation functions to use on each respective hidden layers and output
        layer. This means that the length of 'activation_funcs' is always one
        less than the length of 'layer_sizes'.
        """
        assert(len(layer_sizes) >= 2)
        assert(len(layer_sizes) - 1 == len(activation_funcs))
        assert(min(layer_sizes) >= 1)
        self.layers = []
        self.connections = []

        # Initialize layers.
        for i in range(len(layer_sizes)):
            # Input layer.
            if i == 0:
                self.layers.append(Layer(layer_sizes[i], None, bias_neuron))
            # Hidden layer.
            elif i < len(layer_sizes) - 1:
                self.layers.append(Layer(layer_sizes[i], activation_funcs[i - 1], bias_neuron))
            # Output layer.
            else:
                self.layers.append(Layer(layer_sizes[i], activation_funcs[i - 1]))

        # Initialize connections.
        num_connections = len(layer_sizes) - 1
        for i in range(num_connections):
            self.connections.append(Connection(self.layers[i], self.layers[i + 1]))

    def feed_forward(self, data, one_hot_encoding = True):
        """
        Feeds given data through neural network and stores output in output
        layer's data field. Output can optionally be one-hot encoded.
        """
        if self.layers[0].HAS_BIAS_NEURON:
            assert(len(data) == self.layers[0].SIZE - 1)
            self.layers[0].data = data
            self.layers[0].data.append(1)
        else:
            assert(len(data) == self.layers[0].SIZE)
            self.layers[0].data = data
        for i in range(len(self.connections)):
            self.connections[i].TO.data = np.dot(self.layers[i].data, self.connections[i].weights)
            self.connections[i].TO.activate()
        if one_hot_encoding:
            this_data = self.layers[len(self.layers) - 1].data
            MAX = max(this_data)
            for i in range(len(this_data)):
                if this_data[i] == MAX:
                    this_data[i] = 1
                else:
                    this_data[i] = 0

    def output(self):
        """
        Retrieves data in output layer.
        """
        return self.layers[len(self.layers) - 1].data

class Layer:
    """
    The representation of a layer in a neural network. Used as a medium for
    passing data through the network in an efficent manner.
    """

    def __init__(self, num_neurons, activation_func, bias_neuron = False):
        """
        Creates a 'Layer' with 'num_neurons' and an additional (optional) bias
        neuron (which always has a value of '1'). The layer will utilize the
        'activation_func' during activation.
        """
        assert(num_neurons > 0)
        self.ACTIVATION_FUNC = activation_func
        self.HAS_BIAS_NEURON = bias_neuron
        if bias_neuron:
            self.SIZE = num_neurons + 1
            self.data = np.array([0] * num_neurons + [1])
        else:
            self.SIZE = num_neurons
            self.data = np.array([0] * num_neurons)

    def activate(self):
        """
        Calls activation function on layer's data.
        """
        if self.ACTIVATION_FUNC != None:
            self.ACTIVATION_FUNC(self.data)

class Connection:
    """
    The representation of a connection between layers in a neural network.
    """

    def __init__(self, layer_from, layer_to):
        """"
        Creates a 'Connection' between 'layer_from' and 'layer_to' that contains
        all required weights, which are randomly initialized with random numbers
        in a guassian distribution of mean '0' and standard deviation '1'.
        """
        self.FROM = layer_from
        self.TO = layer_to
        self.weights = np.zeros((layer_from.SIZE, layer_to.SIZE))
        for i in range(layer_from.SIZE):
            for j in range(layer_to.SIZE):
                self.weights[i][j] = np.random.standard_normal()

def sigmoid(data):
    """
    Uses sigmoid transformation on given data. This is an activation function.
    """
    for i in range(len(data)):
        data[i] = 1 / (1 + np.exp(-data[i]))

def softmax(data):
    """
    Uses softmax transformation on given data. This is an activation function.
    """
    sum = 0.0
    for i in range(len(data)):
        sum += np.exp(data[i])
    for i in range(len(data)):
        data[i] = np.exp(data[i]) / sum
