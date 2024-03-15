# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn')
plt.figure(figsize=(8,4))
def SigmoidBinary(t):
    return 1/(1+np.exp(-t))
t = np.linspace(-5, 5)
plt.plot(t, SigmoidBinary(t))
plt.title('Binary Sigmoid Activation Function')
plt.show()


from matplotlib import pyplot
from sklearn.neural_network import MLPClassifier
from math import cos, sin, atan
import numpy as np


class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)


model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)


class Layer():
    def __init__(self, network, number_of_neurons, weights):
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.weights = weights

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return horizontal_distance_between_neurons * (number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, linewidth):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = neuron_radius * sin(angle)
        y_adjustment = neuron_radius * cos(angle)
        line_x_data = (neuron1.x - x_adjustment, neuron2.x + x_adjustment)
        line_y_data = (neuron1.y - y_adjustment, neuron2.y + y_adjustment)
        line = pyplot.Line2D(line_x_data, line_y_data, linewidth=linewidth)
        pyplot.gca().add_line(line)

    def draw(self):
        for this_layer_neuron_index in range(len(self.neurons)):
            neuron = self.neurons[this_layer_neuron_index]
            neuron.draw()
            if self.previous_layer:
                for previous_layer_neuron_index in range(len(self.previous_layer.neurons)):
                    previous_layer_neuron = self.previous_layer.neurons[previous_layer_neuron_index]
                    weight = self.previous_layer.weights[this_layer_neuron_index, previous_layer_neuron_index]
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, weight)


class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add_layer(self, number_of_neurons, weights=None):
        layer = Layer(self, number_of_neurons, weights)
        self.layers.append(layer)

    def draw(self):
        for layer in self.layers:
            layer.draw()
        pyplot.axis('scaled')
        pyplot.show()

model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)



if __name__ == "__main__":
    vertical_distance_between_layers = 20
    horizontal_distance_between_neurons = 3
    neuron_radius = 0.5
    number_of_neurons_in_widest_layer = 20
    network = NeuralNetwork()
    # weights to convert from 100 outputs to 100 (decimal digits to their binary representation)
    weights1 = np.array([\
                         [0,0,0,0,0,0,0,0,1,1],\
                         [0,0,0,0,1,1,1,1,0,0],\
                         [0,0,1,1,0,0,1,1,0,0],\
                         [0,1,0,1,0,1,0,1,0,1]])
    network.add_layer(10, weights1)
    network.add_layer(4)
    network.draw()

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CombinedModel(BaseEstimator, TransformerMixin):
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    def fit(self, X, y=None):
        self.model1.fit(X, y)
        self.model2.fit(X, y)
        return self

    def transform(self, X):
        predictions1 = self.model1.predict(X)
        predictions2 = self.model2.predict(X)
        return np.c_[predictions1, predictions2]

from matplotlib import pyplot
from math import cos, sin, atan
import numpy as np


class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)


class Layer():
    def __init__(self, network, number_of_neurons, weights):
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.weights = weights

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return horizontal_distance_between_neurons * (number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, linewidth):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = neuron_radius * sin(angle)
        y_adjustment = neuron_radius * cos(angle)
        line_x_data = (neuron1.x - x_adjustment, neuron2.x + x_adjustment)
        line_y_data = (neuron1.y - y_adjustment, neuron2.y + y_adjustment)
        line = pyplot.Line2D(line_x_data, line_y_data, linewidth=linewidth)
        pyplot.gca().add_line(line)

    def draw(self):
        for this_layer_neuron_index in range(len(self.neurons)):
            neuron = self.neurons[this_layer_neuron_index]
            neuron.draw()
            if self.previous_layer:
                for previous_layer_neuron_index in range(len(self.previous_layer.neurons)):
                    previous_layer_neuron = self.previous_layer.neurons[previous_layer_neuron_index]
                    weight = self.previous_layer.weights[this_layer_neuron_index, previous_layer_neuron_index]
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, weight)


class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add_layer(self, number_of_neurons, weights=None):
        layer = Layer(self, number_of_neurons, weights)
        self.layers.append(layer)

    def draw(self):
        for layer in self.layers:
            layer.draw()
        pyplot.axis('scaled')
        pyplot.show()
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)


if __name__ == "__main__":
    vertical_distance_between_layers = 20
    horizontal_distance_between_neurons = 3
    neuron_radius = 0.5
    number_of_neurons_in_widest_layer = 20
    network = NeuralNetwork()
    # weights to convert from 100 outputs to 100 (decimal digits to their binary representation)
    weights1 = np.array([\
                         [0,0,0,0,0,0,0,0,1,1],\
                         [0,0,0,0,1,1,1,1,0,0],\
                         [0,0,1,1,0,0,1,1,0,0],\
                         [0,1,0,1,0,1,0,1,0,1]])
    network.add_layer(10, weights1)
    network.add_layer(4)
    network.draw()

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CombinedModel(BaseEstimator, TransformerMixin):
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    def fit(self, X, y=None):
        self.model1.fit(X, y)
        self.model2.fit(X, y)
        return self

    def transform(self, X):
        predictions1 = self.model1.predict(X)
        predictions2 = self.model2.predict(X)
        return np.c_[predictions1, predictions2]

from matplotlib import pyplot
from math import cos, sin, atan
import numpy as np


class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)


class Layer():
    def __init__(self, network, number_of_neurons, weights):
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.weights = weights

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return horizontal_distance_between_neurons * (number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, linewidth):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = neuron_radius * sin(angle)
        y_adjustment = neuron_radius * cos(angle)
        line_x_data = (neuron1.x - x_adjustment, neuron2.x + x_adjustment)
        line_y_data = (neuron1.y - y_adjustment, neuron2.y + y_adjustment)
        line = pyplot.Line2D(line_x_data, line_y_data, linewidth=linewidth)
        pyplot.gca().add_line(line)

    def draw(self):
        for this_layer_neuron_index in range(len(self.neurons)):
            neuron = self.neurons[this_layer_neuron_index]
            neuron.draw()
            if self.previous_layer:
                for previous_layer_neuron_index in range(len(self.previous_layer.neurons)):
                    previous_layer_neuron = self.previous_layer.neurons[previous_layer_neuron_index]
                    weight = self.previous_layer.weights[this_layer_neuron_index, previous_layer_neuron_index]
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, weight)


class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add_layer(self, number_of_neurons, weights=None):
        layer = Layer(self, number_of_neurons, weights)
        self.layers.append(layer)

    def draw(self):
        for layer in self.layers:
            layer.draw()
        pyplot.axis('scaled')
        pyplot.show()


if __name__ == "__main__":
    vertical_distance_between_layers = 20
    horizontal_distance_between_neurons = 3
    neuron_radius = 0.5
    number_of_neurons_in_widest_layer = 20
    network = NeuralNetwork()
    # weights to convert from 100 outputs to 100 (decimal digits to their binary representation)
    weights1 = np.array([\
                         [0,0,0,0,0,0,0,0,1,1],\
                         [0,0,0,0,1,1,1,1,0,0],\
                         [0,0,1,1,0,0,1,1,0,0],\
                         [0,1,0,1,0,1,0,1,0,1]])
    network.add_layer(10, weights1)
    network.add_layer(4)
    network.draw()

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CombinedModel(BaseEstimator, TransformerMixin):
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    def fit(self, X, y=None):
        self.model1.fit(X, y)
        self.model2.fit(X, y)
        return self

    def transform(self, X):
        predictions1 = self.model1.predict(X)
        predictions2 = self.model2.predict(X)
        return np.c_[predictions1, predictions2]

from matplotlib import pyplot
from math import cos, sin, atan
import numpy as np


class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)


class Layer():
    def __init__(self, network, number_of_neurons, weights):
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.weights = weights

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return horizontal_distance_between_neurons * (number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, linewidth):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = neuron_radius * sin(angle)
        y_adjustment = neuron_radius * cos(angle)
        line_x_data = (neuron1.x - x_adjustment, neuron2.x + x_adjustment)
        line_y_data = (neuron1.y - y_adjustment, neuron2.y + y_adjustment)
        line = pyplot.Line2D(line_x_data, line_y_data, linewidth=linewidth)
        pyplot.gca().add_line(line)

    def draw(self):
        for this_layer_neuron_index in range(len(self.neurons)):
            neuron = self.neurons[this_layer_neuron_index]
            neuron.draw()
            if self.previous_layer:
                for previous_layer_neuron_index in range(len(self.previous_layer.neurons)):
                    previous_layer_neuron = self.previous_layer.neurons[previous_layer_neuron_index]
                    weight = self.previous_layer.weights[this_layer_neuron_index, previous_layer_neuron_index]
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, weight)


class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add_layer(self, number_of_neurons, weights=None):
        layer = Layer(self, number_of_neurons, weights)
        self.layers.append(layer)

    def draw(self):
        for layer in self.layers:
            layer.draw()
        pyplot.axis('scaled')
        pyplot.show()


if __name__ == "__main__":
    vertical_distance_between_layers = 20
    horizontal_distance_between_neurons = 3
    neuron_radius = 0.5
    number_of_neurons_in_widest_layer = 20
    network = NeuralNetwork()
    # weights to convert from 100 outputs to 100 (decimal digits to their binary representation)
    weights1 = np.array([\
                         [0,0,0,0,0,0,0,0,1,1],\
                         [0,0,0,0,1,1,1,1,0,0],\
                         [0,0,1,1,0,0,1,1,0,0],\
                         [0,1,0,1,0,1,0,1,0,1]])
    network.add_layer(10, weights1)
    network.add_layer(4)
    network.draw()

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CombinedModel(BaseEstimator, TransformerMixin):
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    def fit(self, X, y=None):
        self.model1.fit(X, y)
        self.model2.fit(X, y)
        return self

    def transform(self, X):
        predictions1 = self.model1.predict(X)
        predictions2 = self.model2.predict(X)
        return np.c_[predictions1, predictions2]

from matplotlib import pyplot
from math import cos, sin, atan
import numpy as np


class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)


class Layer():
    def __init__(self, network, number_of_neurons, weights):
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.weights = weights

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return horizontal_distance_between_neurons * (number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, linewidth):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = neuron_radius * sin(angle)
        y_adjustment = neuron_radius * cos(angle)
        line_x_data = (neuron1.x - x_adjustment, neuron2.x + x_adjustment)
        line_y_data = (neuron1.y - y_adjustment, neuron2.y + y_adjustment)
        line = pyplot.Line2D(line_x_data, line_y_data, linewidth=linewidth)
        pyplot.gca().add_line(line)

    def draw(self):
        for this_layer_neuron_index in range(len(self.neurons)):
            neuron = self.neurons[this_layer_neuron_index]
            neuron.draw()
            if self.previous_layer:
                for previous_layer_neuron_index in range(len(self.previous_layer.neurons)):
                    previous_layer_neuron = self.previous_layer.neurons[previous_layer_neuron_index]
                    weight = self.previous_layer.weights[this_layer_neuron_index, previous_layer_neuron_index]
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, weight)


class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add_layer(self, number_of_neurons, weights=None):
        layer = Layer(self, number_of_neurons, weights)
        self.layers.append(layer)

    def draw(self):
        for layer in self.layers:
            layer.draw()
        pyplot.axis('scaled')
        pyplot.show()


if __name__ == "__main__":
    vertical_distance_between_layers = 20
    horizontal_distance_between_neurons = 3
    neuron_radius = 0.5
    number_of_neurons_in_widest_layer = 20
    network = NeuralNetwork()
    # weights to convert from 100 outputs to 100 (decimal digits to their binary representation)
    weights1 = np.array([\
                         [0,0,0,0,0,0,0,0,1,1],\
                         [0,0,0,0,1,1,1,1,0,0],\
                         [0,0,1,1,0,0,1,1,0,0],\
                         [0,1,0,1,0,1,0,1,0,1]])
    model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
    network.add_layer(10, weights1)
    network.add_layer(4)
    network.draw()
    model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=64000000000000000000000000000, max_iter=2000000000000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)

from projectq import MainEngine
from projectq.ops import All, CNOT, H, Measure, X, Z

quantum_engine = MainEngine()


def create_bell_pair(quantum_engine):
    # Newly created Qubits are in the base state of 0,
    qubit_one = quantum_engine.allocate_qubit()
    qubit_two = quantum_engine.allocate_qubit()
    H | qubit_one
    # Measure | qubit_one
    # qubit_one_val = int(qubit_one)

    CNOT | (qubit_one, qubit_two)
    # Measure | qubit_two
    # cnot_val = int(qubit_two)

    return qubit_one, qubit_two

def create_message(quantum_engine='', qubit_one='', message_value=0):
    qubit_to_send = quantum_engine.allocate_qubit()
    if message_value == 1:
        '''
        setting the qubit to positive if message_value is 1
        by flipping the base state with a Pauli-X gate.
        '''
        X | qubit_to_send

    # entangle the original qubit with the message qubit
    CNOT | (qubit_to_send, qubit_one)

    '''
    1 - Put the message qubit in superposition
    2 - Measure out the two values to get the classical bit value
        by collapsing the state.
    '''
    H | qubit_to_send
    Measure | qubit_to_send
    Measure | qubit_one

    # The qubits are now turned into normal bits we can send through classical channels
    classical_encoded_message = [int(qubit_to_send), int(qubit_one)]

    return classical_encoded_message


def message_reciever(quantum_engine, message, qubit_two):
    '''
    Pauli-X and/or Pauli-Z gates are applied to the Qubit,
    conditionally on the values in the message.
    '''
    if message[1] == 1:
        X | qubit_two
    if message[0] == 1:
        Z | qubit_two

    '''
    Measuring the Qubit and collapsing the state down to either 1 or 0
    '''
    Measure | qubit_two

    quantum_engine.flush()

    received_bit = int(qubit_two)
    return received_bit


qubit_one, qubit_two = create_bell_pair(quantum_engine)
classical_encoded_message = create_message(
    quantum_engine=quantum_engine, qubit_one=qubit_one, message_value=0)

print('classical_encoded_message = ', classical_encoded_message)

received_bit = message_reciever(
    quantum_engine=quantum_engine, message=classical_encoded_message, qubit_two=qubit_two)

print('received_bit = ', str(received_bit))


def send_receive(bit=0, quantum_engine=''):
    # Create bell pair
    qubit_one, qubit_two = create_bell_pair(quantum_engine)
    # entangle the bit with the first qubit
    classical_encoded_message = create_message(
        quantum_engine=quantum_engine, qubit_one=qubit_one, message_value=bit)
    # print('send_bit = ', classical_encoded_message)
    # Teleport the bit and return it back
    received_bit = message_reciever(
        quantum_engine, classical_encoded_message, qubit_two)
    # print('received_bit = ', received_bit)
    return received_bit


message = 'HelloWorld'
binary_encoded_message = [bin(ord(x))[2:].zfill(8) for x in message]
print('Message to send: ', message)
print('Binary message to send: ', binary_encoded_message)

received_bytes_list = []
for letter in binary_encoded_message:
    received_bits = ''
    for bit in letter:
        received_bits = received_bits + \
            str(send_receive(int(bit), quantum_engine))
    received_bytes_list.append(received_bits)

binary_to_string = ''.join([chr(int(x, 2)) for x in received_bytes_list])
print('Received Binary message: ', received_bytes_list)
print('Received message: ', binary_to_string)

quantum_engine.flush()


# bin_mess = 'a'
# print(ord(bin_mess))
# print(bin(ord(bin_mess)))
# print(bin(ord(bin_mess))[2:])
# print(bin(ord(bin_mess))[2:].zfill(8))

# bin_result = bin(ord(bin_mess))[2:].zfill(8)
# print(chr(int(bin_result, 2)))

from projectq import MainEngine
from projectq.ops import All, CNOT, H, Measure, X, Z

quantum_engine = MainEngine()


def create_bell_pair(quantum_engine):
    # Newly created Qubits are in the base state of 0,
    qubit_one = quantum_engine.allocate_qubit()
    qubit_two = quantum_engine.allocate_qubit()
    H | qubit_one
    # Measure | qubit_one
    # qubit_one_val = int(qubit_one)

    CNOT | (qubit_one, qubit_two)
    # Measure | qubit_two
    # cnot_val = int(qubit_two)

    return qubit_one, qubit_two

def create_message(quantum_engine='', qubit_one='', message_value=0):
    qubit_to_send = quantum_engine.allocate_qubit()
    if message_value == 1:
        '''
        setting the qubit to positive if message_value is 1
        by flipping the base state with a Pauli-X gate.
        '''
        X | qubit_to_send

    # entangle the original qubit with the message qubit
    CNOT | (qubit_to_send, qubit_one)

    '''
    1 - Put the message qubit in superposition
    2 - Measure out the two values to get the classical bit value
        by collapsing the state.
    '''
    H | qubit_to_send
    Measure | qubit_to_send
    Measure | qubit_one

    # The qubits are now turned into normal bits we can send through classical channels
    classical_encoded_message = [int(qubit_to_send), int(qubit_one)]

    return classical_encoded_message


def message_reciever(quantum_engine, message, qubit_two):
    '''
    Pauli-X and/or Pauli-Z gates are applied to the Qubit,
    conditionally on the values in the message.
    '''
    if message[1] == 1:
        X | qubit_two
    if message[0] == 1:
        Z | qubit_two

    '''
    Measuring the Qubit and collapsing the state down to either 1 or 0
    '''
    Measure | qubit_two

    quantum_engine.flush()

    received_bit = int(qubit_two)
    return received_bit


qubit_one, qubit_two = create_bell_pair(quantum_engine)
classical_encoded_message = create_message(
    quantum_engine=quantum_engine, qubit_one=qubit_one, message_value=0)

print('classical_encoded_message = ', classical_encoded_message)

received_bit = message_reciever(
    quantum_engine=quantum_engine, message=classical_encoded_message, qubit_two=qubit_two)

print('received_bit = ', str(received_bit))


def send_receive(bit=0, quantum_engine=''):
    # Create bell pair
    qubit_one, qubit_two = create_bell_pair(quantum_engine)
    # entangle the bit with the first qubit
    classical_encoded_message = create_message(
        quantum_engine=quantum_engine, qubit_one=qubit_one, message_value=bit)
    # print('send_bit = ', classical_encoded_message)
    # Teleport the bit and return it back
    received_bit = message_reciever(
        quantum_engine, classical_encoded_message, qubit_two)
    # print('received_bit = ', received_bit)
    return received_bit


message = 'HelloWorld'
binary_encoded_message = [bin(ord(x))[2:].zfill(8) for x in message]
print('Message to send: ', message)
print('Binary message to send: ', binary_encoded_message)

received_bytes_list = []
for letter in binary_encoded_message:
    received_bits = ''
    for bit in letter:
        received_bits = received_bits + \
            str(send_receive(int(bit), quantum_engine))
    received_bytes_list.append(received_bits)

binary_to_string = ''.join([chr(int(x, 2)) for x in received_bytes_list])
print('Received Binary message: ', received_bytes_list)
print('Received message: ', binary_to_string)

quantum_engine.flush()


# bin_mess = 'a'
# print(ord(bin_mess))
# print(bin(ord(bin_mess)))
# print(bin(ord(bin_mess))[2:])
# print(bin(ord(bin_mess))[2:].zfill(8))

# bin_result = bin(ord(bin_mess))[2:].zfill(8)
# print(chr(int(bin_result, 2)))

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
# Generate a random quantum state vector
num_qubits = 1  # Number of qubits
state = rand_ket(2 ** num_qubits)

# Visualize the state using a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sphere = Bloch(axes=ax)
sphere.add_states(state)
sphere.show()
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
# Generate a random quantum state vector
num_qubits = 1  # Number of qubits
state = rand_ket(2 ** num_qubits)

# Alternatively, visualize the state using a Bloch sphere representation
bloch = Bloch()
bloch.add_states(state)
bloch.show()

# Show the plots
plt.show()

# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
...

...
# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

...
# shape
print(dataset.shape)

# Authors:
# Mathieu Blondel <mathieu@mblondel.org>
# Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
# Balazs Kegl <balazs.kegl@gmail.com>
# Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# License: BSD Style.

import numpy as np

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

n_samples = 50000
n_bins = 3  # use 3 bins for calibration_curve as we have 3 clusters here

# Generate 3 blobs with 2 classes where the second blob contains
# half positive samples and half negative samples. Probability in this
# blob is therefore 0.5.
centers = [(-5, -5), (0, 0), (5, 5)]
X, y = make_blobs(n_samples=n_samples, centers=centers, shuffle=False, random_state=42)

y[: n_samples // 2] = 0
y[n_samples // 2 :] = 1
sample_weight = np.random.RandomState(42).rand(y.shape[0])

# split train, test for calibration
X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
    X, y, sample_weight, test_size=0.9, random_state=42
)

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.naive_bayes import GaussianNB

# With no calibration
clf = GaussianNB()
clf.fit(X_train, y_train)  # GaussianNB itself does not support sample-weights
prob_pos_clf = clf.predict_proba(X_test)[:, 1]

# With isotonic calibration
clf_isotonic = CalibratedClassifierCV(clf, cv=2, method="isotonic")
clf_isotonic.fit(X_train, y_train, sample_weight=sw_train)
prob_pos_isotonic = clf_isotonic.predict_proba(X_test)[:, 1]

# With sigmoid calibration
clf_sigmoid = CalibratedClassifierCV(clf, cv=2, method="sigmoid")
clf_sigmoid.fit(X_train, y_train, sample_weight=sw_train)
prob_pos_sigmoid = clf_sigmoid.predict_proba(X_test)[:, 1]

print("Brier score losses: (the smaller the better)")

clf_score = brier_score_loss(y_test, prob_pos_clf, sample_weight=sw_test)
print("No calibration: %1.3f" % clf_score)

clf_isotonic_score = brier_score_loss(y_test, prob_pos_isotonic, sample_weight=sw_test)
print("With isotonic calibration: %1.3f" % clf_isotonic_score)

clf_sigmoid_score = brier_score_loss(y_test, prob_pos_sigmoid, sample_weight=sw_test)
print("With sigmoid calibration: %1.3f" % clf_sigmoid_score)

import matplotlib.pyplot as plt
from matplotlib import cm

plt.figure()
y_unique = np.unique(y)
colors = cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))
for this_y, color in zip(y_unique, colors):
    this_X = X_train[y_train == this_y]
    this_sw = sw_train[y_train == this_y]
    plt.scatter(
        this_X[:, 0],
        this_X[:, 1],
        s=this_sw * 50,
        c=color[np.newaxis, :],
        alpha=0.5,
        edgecolor="k",
        label="Class %s" % this_y,
    )
plt.legend(loc="best")
plt.title("Data")

plt.figure()

order = np.lexsort((prob_pos_clf,))
plt.plot(prob_pos_clf[order], "r", label="No calibration (%1.3f)" % clf_score)
plt.plot(
    prob_pos_isotonic[order],
    "g",
    linewidth=3,
    label="Isotonic calibration (%1.3f)" % clf_isotonic_score,
)
plt.plot(
    prob_pos_sigmoid[order],
    "b",
    linewidth=3,
    label="Sigmoid calibration (%1.3f)" % clf_sigmoid_score,
)
plt.plot(
    np.linspace(0, y_test.size, 51)[1::2],
    y_test[order].reshape(25, -1).mean(1),
    "k",
    linewidth=3,
    label=r"Empirical",
)
plt.ylim([-0.05, 1.05])
plt.xlabel("Instances sorted according to predicted probability (uncalibrated GNB)")
plt.ylabel("P(y=1)")
plt.legend(loc="upper left")
plt.title("Gaussian naive Bayes probabilities")

plt.show()

import os
import sys
import requests
from tqdm import tqdm

subdir = 'data'
if not os.path.exists(subdir):
    os.makedirs(subdir)
subdir = subdir.replace('\\','/') # needed for Windows

for ds in [
    'webtext',
    'small-117M',  'small-117M-k40',
    'medium-345M', 'medium-345M-k40',
    'large-762M',  'large-762M-k40',
    'xl-1542M',    'xl-1542M-k40',
]:
    for split in ['train', 'valid', 'test']:
        filename = ds + "." + split + '.jsonl'
        r = requests.get("https://openaipublic.azureedge.net/gpt-2/output-dataset/v1/" + filename, stream=True)

        with open(os.path.join(subdir, filename), 'wb') as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)

...
# shape
print(dataset.shape)
...
# head
print(dataset.head(20))

...
# descriptions
print(dataset.describe())

...
# class distribution
print(dataset.groupby('class').size())

...
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

...
# histograms
dataset.hist()
plt.show()

from matplotlib import pyplot
from math import cos, sin, atan
import numpy as np


class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)


class Layer():
    def __init__(self, network, number_of_neurons, weights):
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.weights = weights

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return horizontal_distance_between_neurons * (number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, linewidth):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = neuron_radius * sin(angle)
        y_adjustment = neuron_radius * cos(angle)
        line_x_data = (neuron1.x - x_adjustment, neuron2.x + x_adjustment)
        line_y_data = (neuron1.y - y_adjustment, neuron2.y + y_adjustment)
        line = pyplot.Line2D(line_x_data, line_y_data, linewidth=linewidth)
        pyplot.gca().add_line(line)

    def draw(self):
        for this_layer_neuron_index in range(len(self.neurons)):
            neuron = self.neurons[this_layer_neuron_index]
            neuron.draw()
            if self.previous_layer:
                for previous_layer_neuron_index in range(len(self.previous_layer.neurons)):
                    previous_layer_neuron = self.previous_layer.neurons[previous_layer_neuron_index]
                    weight = self.previous_layer.weights[this_layer_neuron_index, previous_layer_neuron_index]
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, weight)


class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add_layer(self, number_of_neurons, weights=None):
        layer = Layer(self, number_of_neurons, weights)
        self.layers.append(layer)

    def draw(self):
        for layer in self.layers:
            layer.draw()
        pyplot.axis('scaled')
        pyplot.show()


if __name__ == "__main__":
    vertical_distance_between_layers = 20
    horizontal_distance_between_neurons = 3
    neuron_radius = 0.5
    number_of_neurons_in_widest_layer = 20
    network = NeuralNetwork()
    # weights to convert from 100 outputs to 100 (decimal digits to their binary representation)
    weights1 = np.array([\
                         [0,0,0,0,0,0,0,0,1,1],\
                         [0,0,0,0,1,1,1,1,0,0],\
                         [0,0,1,1,0,0,1,1,0,0],\
                         [0,1,0,1,0,1,0,1,0,1]])
    network.add_layer(10, weights1)
    network.add_layer(4)
    network.draw()

...
# scatter plot matrix
scatter_matrix(dataset)
plt.show()

# visualize the data
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
# histograms
dataset.hist()
plt.show()
# scatter plot matrix
scatter_matrix(dataset)
plt.show()

...
# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

...
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

...
# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()

# compare algorithms
from pandas import read_csv
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()

...
# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# make predictions
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

from random import seed
from random import random

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

seed(1)
network = initialize_network(2, 1, 2)
for layer in network:
	print(layer)

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

from math import exp

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# test forward propagation
network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
		[{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]]
row = [1, 0, None]
output = forward_propagate(network, row)
print(output)

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(neuron['output'] - expected[j])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(neuron['output'] - expected[j])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# test backpropagation of error
network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
		[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
expected = [0, 1]
backward_propagate_error(network, expected)
for layer in network:
	print(layer)

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] -= l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

from math import exp
from random import seed
from random import random

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(neuron['output'] - expected[j])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] -= l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

# Test training backprop algorithm
seed(1)
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, 2, n_outputs)
train_network(network, dataset, 0.5, 500000, n_outputs)
for layer in network:
	print(layer)

# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

from math import exp

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

# Test making predictions with the network
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
network = [[{'weights': [-1.482313569067226, 1.8308790073202204, 1.078381922048799]}, {'weights': [0.23244990332399884, 0.3621998343835864, 0.40289821191094327]}],
	[{'weights': [2.5001872433501404, 0.7887233511355132, -1.1026649757805829]}, {'weights': [-2.429350576245497, 0.8357651039198697, 1.0699217181280656]}]]
for row in dataset:
	prediction = predict(network, row)
	print('Expected=%d, Got=%d' % (row[-1], prediction))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CombinedModel(BaseEstimator, TransformerMixin):
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    def fit(self, X, y=None):
        self.model1.fit(X, y)
        self.model2.fit(X, y)
        return self

    def transform(self, X):
        predictions1 = self.model1.predict(X)
        predictions2 = self.model2.predict(X)
        return np.c_[predictions1, predictions2]

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn')
plt.figure(figsize=(8,4))
def SigmoidBinary(t):
    return 1/(1+np.exp(-t))
t = np.linspace(-5, 5)
plt.plot(t, SigmoidBinary(t))
plt.title('Binary Sigmoid Activation Function')
plt.show()


from matplotlib import pyplot
from sklearn.neural_network import MLPClassifier
from math import cos, sin, atan
import numpy as np


class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)


model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)


class Layer():
    def __init__(self, network, number_of_neurons, weights):
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.weights = weights

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return horizontal_distance_between_neurons * (number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, linewidth):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = neuron_radius * sin(angle)
        y_adjustment = neuron_radius * cos(angle)
        line_x_data = (neuron1.x - x_adjustment, neuron2.x + x_adjustment)
        line_y_data = (neuron1.y - y_adjustment, neuron2.y + y_adjustment)
        line = pyplot.Line2D(line_x_data, line_y_data, linewidth=linewidth)
        pyplot.gca().add_line(line)

    def draw(self):
        for this_layer_neuron_index in range(len(self.neurons)):
            neuron = self.neurons[this_layer_neuron_index]
            neuron.draw()
            if self.previous_layer:
                for previous_layer_neuron_index in range(len(self.previous_layer.neurons)):
                    previous_layer_neuron = self.previous_layer.neurons[previous_layer_neuron_index]
                    weight = self.previous_layer.weights[this_layer_neuron_index, previous_layer_neuron_index]
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, weight)


class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add_layer(self, number_of_neurons, weights=None):
        layer = Layer(self, number_of_neurons, weights)
        self.layers.append(layer)

    def draw(self):
        for layer in self.layers:
            layer.draw()
        pyplot.axis('scaled')
        pyplot.show()

model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)



if __name__ == "__main__":
    vertical_distance_between_layers = 20
    horizontal_distance_between_neurons = 3
    neuron_radius = 0.5
    number_of_neurons_in_widest_layer = 20
    network = NeuralNetwork()
    # weights to convert from 100 outputs to 100 (decimal digits to their binary representation)
    weights1 = np.array([\
                         [0,0,0,0,0,0,0,0,1,1],\
                         [0,0,0,0,1,1,1,1,0,0],\
                         [0,0,1,1,0,0,1,1,0,0],\
                         [0,1,0,1,0,1,0,1,0,1]])
    network.add_layer(10, weights1)
    network.add_layer(4)
    network.draw()

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CombinedModel(BaseEstimator, TransformerMixin):
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    def fit(self, X, y=None):
        self.model1.fit(X, y)
        self.model2.fit(X, y)
        return self

    def transform(self, X):
        predictions1 = self.model1.predict(X)
        predictions2 = self.model2.predict(X)
        return np.c_[predictions1, predictions2]

from matplotlib import pyplot
from math import cos, sin, atan
import numpy as np


class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)


class Layer():
    def __init__(self, network, number_of_neurons, weights):
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.weights = weights

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return horizontal_distance_between_neurons * (number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, linewidth):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = neuron_radius * sin(angle)
        y_adjustment = neuron_radius * cos(angle)
        line_x_data = (neuron1.x - x_adjustment, neuron2.x + x_adjustment)
        line_y_data = (neuron1.y - y_adjustment, neuron2.y + y_adjustment)
        line = pyplot.Line2D(line_x_data, line_y_data, linewidth=linewidth)
        pyplot.gca().add_line(line)

    def draw(self):
        for this_layer_neuron_index in range(len(self.neurons)):
            neuron = self.neurons[this_layer_neuron_index]
            neuron.draw()
            if self.previous_layer:
                for previous_layer_neuron_index in range(len(self.previous_layer.neurons)):
                    previous_layer_neuron = self.previous_layer.neurons[previous_layer_neuron_index]
                    weight = self.previous_layer.weights[this_layer_neuron_index, previous_layer_neuron_index]
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, weight)


class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add_layer(self, number_of_neurons, weights=None):
        layer = Layer(self, number_of_neurons, weights)
        self.layers.append(layer)

    def draw(self):
        for layer in self.layers:
            layer.draw()
        pyplot.axis('scaled')
        pyplot.show()
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)


if __name__ == "__main__":
    vertical_distance_between_layers = 20
    horizontal_distance_between_neurons = 3
    neuron_radius = 0.5
    number_of_neurons_in_widest_layer = 20
    network = NeuralNetwork()
    # weights to convert from 100 outputs to 100 (decimal digits to their binary representation)
    weights1 = np.array([\
                         [0,0,0,0,0,0,0,0,1,1],\
                         [0,0,0,0,1,1,1,1,0,0],\
                         [0,0,1,1,0,0,1,1,0,0],\
                         [0,1,0,1,0,1,0,1,0,1]])
    network.add_layer(10, weights1)
    network.add_layer(4)
    network.draw()

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CombinedModel(BaseEstimator, TransformerMixin):
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    def fit(self, X, y=None):
        self.model1.fit(X, y)
        self.model2.fit(X, y)
        return self

    def transform(self, X):
        predictions1 = self.model1.predict(X)
        predictions2 = self.model2.predict(X)
        return np.c_[predictions1, predictions2]

from matplotlib import pyplot
from math import cos, sin, atan
import numpy as np


class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)


class Layer():
    def __init__(self, network, number_of_neurons, weights):
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.weights = weights

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return horizontal_distance_between_neurons * (number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, linewidth):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = neuron_radius * sin(angle)
        y_adjustment = neuron_radius * cos(angle)
        line_x_data = (neuron1.x - x_adjustment, neuron2.x + x_adjustment)
        line_y_data = (neuron1.y - y_adjustment, neuron2.y + y_adjustment)
        line = pyplot.Line2D(line_x_data, line_y_data, linewidth=linewidth)
        pyplot.gca().add_line(line)

    def draw(self):
        for this_layer_neuron_index in range(len(self.neurons)):
            neuron = self.neurons[this_layer_neuron_index]
            neuron.draw()
            if self.previous_layer:
                for previous_layer_neuron_index in range(len(self.previous_layer.neurons)):
                    previous_layer_neuron = self.previous_layer.neurons[previous_layer_neuron_index]
                    weight = self.previous_layer.weights[this_layer_neuron_index, previous_layer_neuron_index]
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, weight)


class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add_layer(self, number_of_neurons, weights=None):
        layer = Layer(self, number_of_neurons, weights)
        self.layers.append(layer)

    def draw(self):
        for layer in self.layers:
            layer.draw()
        pyplot.axis('scaled')
        pyplot.show()


if __name__ == "__main__":
    vertical_distance_between_layers = 20
    horizontal_distance_between_neurons = 3
    neuron_radius = 0.5
    number_of_neurons_in_widest_layer = 20
    network = NeuralNetwork()
    # weights to convert from 100 outputs to 100 (decimal digits to their binary representation)
    weights1 = np.array([\
                         [0,0,0,0,0,0,0,0,1,1],\
                         [0,0,0,0,1,1,1,1,0,0],\
                         [0,0,1,1,0,0,1,1,0,0],\
                         [0,1,0,1,0,1,0,1,0,1]])
    network.add_layer(10, weights1)
    network.add_layer(4)
    network.draw()

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CombinedModel(BaseEstimator, TransformerMixin):
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    def fit(self, X, y=None):
        self.model1.fit(X, y)
        self.model2.fit(X, y)
        return self

    def transform(self, X):
        predictions1 = self.model1.predict(X)
        predictions2 = self.model2.predict(X)
        return np.c_[predictions1, predictions2]

from matplotlib import pyplot
from math import cos, sin, atan
import numpy as np


class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)


class Layer():
    def __init__(self, network, number_of_neurons, weights):
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.weights = weights

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return horizontal_distance_between_neurons * (number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, linewidth):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = neuron_radius * sin(angle)
        y_adjustment = neuron_radius * cos(angle)
        line_x_data = (neuron1.x - x_adjustment, neuron2.x + x_adjustment)
        line_y_data = (neuron1.y - y_adjustment, neuron2.y + y_adjustment)
        line = pyplot.Line2D(line_x_data, line_y_data, linewidth=linewidth)
        pyplot.gca().add_line(line)

    def draw(self):
        for this_layer_neuron_index in range(len(self.neurons)):
            neuron = self.neurons[this_layer_neuron_index]
            neuron.draw()
            if self.previous_layer:
                for previous_layer_neuron_index in range(len(self.previous_layer.neurons)):
                    previous_layer_neuron = self.previous_layer.neurons[previous_layer_neuron_index]
                    weight = self.previous_layer.weights[this_layer_neuron_index, previous_layer_neuron_index]
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, weight)


class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add_layer(self, number_of_neurons, weights=None):
        layer = Layer(self, number_of_neurons, weights)
        self.layers.append(layer)

    def draw(self):
        for layer in self.layers:
            layer.draw()
        pyplot.axis('scaled')
        pyplot.show()


if __name__ == "__main__":
    vertical_distance_between_layers = 20
    horizontal_distance_between_neurons = 3
    neuron_radius = 0.5
    number_of_neurons_in_widest_layer = 20
    network = NeuralNetwork()
    # weights to convert from 100 outputs to 100 (decimal digits to their binary representation)
    weights1 = np.array([\
                         [0,0,0,0,0,0,0,0,1,1],\
                         [0,0,0,0,1,1,1,1,0,0],\
                         [0,0,1,1,0,0,1,1,0,0],\
                         [0,1,0,1,0,1,0,1,0,1]])
    network.add_layer(10, weights1)
    network.add_layer(4)
    network.draw()

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CombinedModel(BaseEstimator, TransformerMixin):
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    def fit(self, X, y=None):
        self.model1.fit(X, y)
        self.model2.fit(X, y)
        return self

    def transform(self, X):
        predictions1 = self.model1.predict(X)
        predictions2 = self.model2.predict(X)
        return np.c_[predictions1, predictions2]

from matplotlib import pyplot
from math import cos, sin, atan
import numpy as np


class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)


class Layer():
    def __init__(self, network, number_of_neurons, weights):
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.weights = weights

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return horizontal_distance_between_neurons * (number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, linewidth):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = neuron_radius * sin(angle)
        y_adjustment = neuron_radius * cos(angle)
        line_x_data = (neuron1.x - x_adjustment, neuron2.x + x_adjustment)
        line_y_data = (neuron1.y - y_adjustment, neuron2.y + y_adjustment)
        line = pyplot.Line2D(line_x_data, line_y_data, linewidth=linewidth)
        pyplot.gca().add_line(line)

    def draw(self):
        for this_layer_neuron_index in range(len(self.neurons)):
            neuron = self.neurons[this_layer_neuron_index]
            neuron.draw()
            if self.previous_layer:
                for previous_layer_neuron_index in range(len(self.previous_layer.neurons)):
                    previous_layer_neuron = self.previous_layer.neurons[previous_layer_neuron_index]
                    weight = self.previous_layer.weights[this_layer_neuron_index, previous_layer_neuron_index]
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, weight)


class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add_layer(self, number_of_neurons, weights=None):
        layer = Layer(self, number_of_neurons, weights)
        self.layers.append(layer)

    def draw(self):
        for layer in self.layers:
            layer.draw()
        pyplot.axis('scaled')
        pyplot.show()


if __name__ == "__main__":
    vertical_distance_between_layers = 20
    horizontal_distance_between_neurons = 3
    neuron_radius = 0.5
    number_of_neurons_in_widest_layer = 20
    network = NeuralNetwork()
    # weights to convert from 100 outputs to 100 (decimal digits to their binary representation)
    weights1 = np.array([\
                         [0,0,0,0,0,0,0,0,1,1],\
                         [0,0,0,0,1,1,1,1,0,0],\
                         [0,0,1,1,0,0,1,1,0,0],\
                         [0,1,0,1,0,1,0,1,0,1]])
    model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
    network.add_layer(10, weights1)
    network.add_layer(4)
    network.draw()
    model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=64000000000000000000000000000, max_iter=2000000000000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)
model = MLPClassifier(hidden_layer_sizes=6400, max_iter=2000)

from projectq import MainEngine
from projectq.ops import All, CNOT, H, Measure, X, Z

quantum_engine = MainEngine()


def create_bell_pair(quantum_engine):
    # Newly created Qubits are in the base state of 0,
    qubit_one = quantum_engine.allocate_qubit()
    qubit_two = quantum_engine.allocate_qubit()
    H | qubit_one
    # Measure | qubit_one
    # qubit_one_val = int(qubit_one)

    CNOT | (qubit_one, qubit_two)
    # Measure | qubit_two
    # cnot_val = int(qubit_two)

    return qubit_one, qubit_two

def create_message(quantum_engine='', qubit_one='', message_value=0):
    qubit_to_send = quantum_engine.allocate_qubit()
    if message_value == 1:
        '''
        setting the qubit to positive if message_value is 1
        by flipping the base state with a Pauli-X gate.
        '''
        X | qubit_to_send

    # entangle the original qubit with the message qubit
    CNOT | (qubit_to_send, qubit_one)

    '''
    1 - Put the message qubit in superposition
    2 - Measure out the two values to get the classical bit value
        by collapsing the state.
    '''
    H | qubit_to_send
    Measure | qubit_to_send
    Measure | qubit_one

    # The qubits are now turned into normal bits we can send through classical channels
    classical_encoded_message = [int(qubit_to_send), int(qubit_one)]

    return classical_encoded_message


def message_reciever(quantum_engine, message, qubit_two):
    '''
    Pauli-X and/or Pauli-Z gates are applied to the Qubit,
    conditionally on the values in the message.
    '''
    if message[1] == 1:
        X | qubit_two
    if message[0] == 1:
        Z | qubit_two

    '''
    Measuring the Qubit and collapsing the state down to either 1 or 0
    '''
    Measure | qubit_two

    quantum_engine.flush()

    received_bit = int(qubit_two)
    return received_bit


qubit_one, qubit_two = create_bell_pair(quantum_engine)
classical_encoded_message = create_message(
    quantum_engine=quantum_engine, qubit_one=qubit_one, message_value=0)

print('classical_encoded_message = ', classical_encoded_message)

received_bit = message_reciever(
    quantum_engine=quantum_engine, message=classical_encoded_message, qubit_two=qubit_two)

print('received_bit = ', str(received_bit))


def send_receive(bit=0, quantum_engine=''):
    # Create bell pair
    qubit_one, qubit_two = create_bell_pair(quantum_engine)
    # entangle the bit with the first qubit
    classical_encoded_message = create_message(
        quantum_engine=quantum_engine, qubit_one=qubit_one, message_value=bit)
    # print('send_bit = ', classical_encoded_message)
    # Teleport the bit and return it back
    received_bit = message_reciever(
        quantum_engine, classical_encoded_message, qubit_two)
    # print('received_bit = ', received_bit)
    return received_bit


message = 'HelloWorld'
binary_encoded_message = [bin(ord(x))[2:].zfill(8) for x in message]
print('Message to send: ', message)
print('Binary message to send: ', binary_encoded_message)

received_bytes_list = []
for letter in binary_encoded_message:
    received_bits = ''
    for bit in letter:
        received_bits = received_bits + \
            str(send_receive(int(bit), quantum_engine))
    received_bytes_list.append(received_bits)

binary_to_string = ''.join([chr(int(x, 2)) for x in received_bytes_list])
print('Received Binary message: ', received_bytes_list)
print('Received message: ', binary_to_string)

quantum_engine.flush()


# bin_mess = 'a'
# print(ord(bin_mess))
# print(bin(ord(bin_mess)))
# print(bin(ord(bin_mess))[2:])
# print(bin(ord(bin_mess))[2:].zfill(8))

# bin_result = bin(ord(bin_mess))[2:].zfill(8)
# print(chr(int(bin_result, 2)))

from projectq import MainEngine
from projectq.ops import All, CNOT, H, Measure, X, Z

quantum_engine = MainEngine()


def create_bell_pair(quantum_engine):
    # Newly created Qubits are in the base state of 0,
    qubit_one = quantum_engine.allocate_qubit()
    qubit_two = quantum_engine.allocate_qubit()
    H | qubit_one
    # Measure | qubit_one
    # qubit_one_val = int(qubit_one)

    CNOT | (qubit_one, qubit_two)
    # Measure | qubit_two
    # cnot_val = int(qubit_two)

    return qubit_one, qubit_two

def create_message(quantum_engine='', qubit_one='', message_value=0):
    qubit_to_send = quantum_engine.allocate_qubit()
    if message_value == 1:
        '''
        setting the qubit to positive if message_value is 1
        by flipping the base state with a Pauli-X gate.
        '''
        X | qubit_to_send

    # entangle the original qubit with the message qubit
    CNOT | (qubit_to_send, qubit_one)

    '''
    1 - Put the message qubit in superposition
    2 - Measure out the two values to get the classical bit value
        by collapsing the state.
    '''
    H | qubit_to_send
    Measure | qubit_to_send
    Measure | qubit_one

    # The qubits are now turned into normal bits we can send through classical channels
    classical_encoded_message = [int(qubit_to_send), int(qubit_one)]

    return classical_encoded_message


def message_reciever(quantum_engine, message, qubit_two):
    '''
    Pauli-X and/or Pauli-Z gates are applied to the Qubit,
    conditionally on the values in the message.
    '''
    if message[1] == 1:
        X | qubit_two
    if message[0] == 1:
        Z | qubit_two

    '''
    Measuring the Qubit and collapsing the state down to either 1 or 0
    '''
    Measure | qubit_two

    quantum_engine.flush()

    received_bit = int(qubit_two)
    return received_bit


qubit_one, qubit_two = create_bell_pair(quantum_engine)
classical_encoded_message = create_message(
    quantum_engine=quantum_engine, qubit_one=qubit_one, message_value=0)

print('classical_encoded_message = ', classical_encoded_message)

received_bit = message_reciever(
    quantum_engine=quantum_engine, message=classical_encoded_message, qubit_two=qubit_two)

print('received_bit = ', str(received_bit))


def send_receive(bit=0, quantum_engine=''):
    # Create bell pair
    qubit_one, qubit_two = create_bell_pair(quantum_engine)
    # entangle the bit with the first qubit
    classical_encoded_message = create_message(
        quantum_engine=quantum_engine, qubit_one=qubit_one, message_value=bit)
    # print('send_bit = ', classical_encoded_message)
    # Teleport the bit and return it back
    received_bit = message_reciever(
        quantum_engine, classical_encoded_message, qubit_two)
    # print('received_bit = ', received_bit)
    return received_bit


message = 'HelloWorld'
binary_encoded_message = [bin(ord(x))[2:].zfill(8) for x in message]
print('Message to send: ', message)
print('Binary message to send: ', binary_encoded_message)

received_bytes_list = []
for letter in binary_encoded_message:
    received_bits = ''
    for bit in letter:
        received_bits = received_bits + \
            str(send_receive(int(bit), quantum_engine))
    received_bytes_list.append(received_bits)

binary_to_string = ''.join([chr(int(x, 2)) for x in received_bytes_list])
print('Received Binary message: ', received_bytes_list)
print('Received message: ', binary_to_string)

quantum_engine.flush()


# bin_mess = 'a'
# print(ord(bin_mess))
# print(bin(ord(bin_mess)))
# print(bin(ord(bin_mess))[2:])
# print(bin(ord(bin_mess))[2:].zfill(8))

# bin_result = bin(ord(bin_mess))[2:].zfill(8)
# print(chr(int(bin_result, 2)))

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
# Generate a random quantum state vector
num_qubits = 1  # Number of qubits
state = rand_ket(2 ** num_qubits)

# Visualize the state using a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sphere = Bloch(axes=ax)
sphere.add_states(state)
sphere.show()
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
# Generate a random quantum state vector
num_qubits = 1  # Number of qubits
state = rand_ket(2 ** num_qubits)

# Alternatively, visualize the state using a Bloch sphere representation
bloch = Bloch()
bloch.add_states(state)
bloch.show()

# Show the plots
plt.show()

# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
...

...
# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

...
# shape
print(dataset.shape)

import qiskit as q
from qiskit.tools.visualization import plot_bloch_multivector
from qiskit.visualization import plot_histogram
from matplotlib import style
#style.use("dark_background") # I am using dark mode notebook, so I use this to see the chart.
# to use dark mode:
# edited '/usr/local/lib/python3.7/dist-packages/qiskit/visualization/bloch.py line 177 self.font_color = 'white'
# edited '/usr/local/lib/python3.7/dist-packages/qiskit/visualization/counts_visualization.py line 206     ax.set_facecolor('#000000')



statevec_simulator = q.Aer.get_backend("statevector_simulator")
qasm_sim = q.Aer.get_backend('qasm_simulator')

def do_job(circuit):
    job = q.execute(circuit, backend=statevec_simulator)
    result = job.result()
    statevec = result.get_statevector()
    
    n_qubits = circuit.n_qubits
    circuit.measure([i for i in range(n_qubits)], [i for i in range(n_qubits)])
    
    qasm_job = q.execute(circuit, backend=qasm_sim, shots=1024).result()
    counts = qasm_job.get_counts()
    
    return statevec, counts

circuit = q.QuantumCircuit(2,2)  # 2 qubits, 2 classical bits 
circuit.h(0)  # hadamard gate on qubit0
statevec, counts = do_job(circuit)
plot_bloch_multivector(statevec)

circuit = q.QuantumCircuit(2,2)  # 2 qubits, 2 classical bits 
circuit.h(0)  # hadamard gate on qubit0
circuit.cx(0,1)  # controlled not control: 0 target: 1
statevec, counts = do_job(circuit)
plot_bloch_multivector(statevec)
plot_histogram([counts], legend=['output'])

circuit = q.QuantumCircuit(3,3)  # 2 qubits, 2 classical bits 
circuit.h(0)
circuit.h(1)
circuit.cx(0,2)
circuit.cx(1,2)
circuit.draw()

statevec, counts = do_job(circuit)
plot_bloch_multivector(statevec)

plot_histogram([counts], legend=['output'])

circuit = q.QuantumCircuit(3,3)  # 3 qubits, 3 classical bits 
circuit.h(0)
circuit.h(1)
circuit.ccx(0,1,2)
circuit.draw()

statevec, counts = do_job(circuit)
plot_bloch_multivector(statevec)
from projectq import MainEngine
from projectq.ops import All, CNOT, H, Measure, X, Z

quantum_engine = MainEngine()


def create_bell_pair(quantum_engine):
    # Newly created Qubits are in the base state of 0,
    qubit_one = quantum_engine.allocate_qubit()
    qubit_two = quantum_engine.allocate_qubit()
    H | qubit_one
    # Measure | qubit_one
    # qubit_one_val = int(qubit_one)

    CNOT | (qubit_one, qubit_two)
    # Measure | qubit_two
    # cnot_val = int(qubit_two)

    return qubit_one, qubit_two

def create_message(quantum_engine='', qubit_one='', message_value=0):
    qubit_to_send = quantum_engine.allocate_qubit()
    if message_value == 1:
        '''
        setting the qubit to positive if message_value is 1
        by flipping the base state with a Pauli-X gate.
        '''
        X | qubit_to_send

    # entangle the original qubit with the message qubit
    CNOT | (qubit_to_send, qubit_one)

    '''
    1 - Put the message qubit in superposition
    2 - Measure out the two values to get the classical bit value
        by collapsing the state.
    '''
    H | qubit_to_send
    Measure | qubit_to_send
    Measure | qubit_one

    # The qubits are now turned into normal bits we can send through classical channels
    classical_encoded_message = [int(qubit_to_send), int(qubit_one)]

    return classical_encoded_message


def message_reciever(quantum_engine, message, qubit_two):
    '''
    Pauli-X and/or Pauli-Z gates are applied to the Qubit,
    conditionally on the values in the message.
    '''
    if message[1] == 1:
        X | qubit_two
    if message[0] == 1:
        Z | qubit_two

    '''
    Measuring the Qubit and collapsing the state down to either 1 or 0
    '''
    Measure | qubit_two

    quantum_engine.flush()

    received_bit = int(qubit_two)
    return received_bit


qubit_one, qubit_two = create_bell_pair(quantum_engine)
classical_encoded_message = create_message(
    quantum_engine=quantum_engine, qubit_one=qubit_one, message_value=0)

print('classical_encoded_message = ', classical_encoded_message)

received_bit = message_reciever(
    quantum_engine=quantum_engine, message=classical_encoded_message, qubit_two=qubit_two)

print('received_bit = ', str(received_bit))


def send_receive(bit=0, quantum_engine=''):
    # Create bell pair
    qubit_one, qubit_two = create_bell_pair(quantum_engine)
    # entangle the bit with the first qubit
    classical_encoded_message = create_message(
        quantum_engine=quantum_engine, qubit_one=qubit_one, message_value=bit)
    # print('send_bit = ', classical_encoded_message)
    # Teleport the bit and return it back
    received_bit = message_reciever(
        quantum_engine, classical_encoded_message, qubit_two)
    # print('received_bit = ', received_bit)
    return received_bit


message = 'HelloWorld'
binary_encoded_message = [bin(ord(x))[2:].zfill(8) for x in message]
print('Message to send: ', message)
print('Binary message to send: ', binary_encoded_message)

received_bytes_list = []
for letter in binary_encoded_message:
    received_bits = ''
    for bit in letter:
        received_bits = received_bits + \
            str(send_receive(int(bit), quantum_engine))
    received_bytes_list.append(received_bits)

binary_to_string = ''.join([chr(int(x, 2)) for x in received_bytes_list])
print('Received Binary message: ', received_bytes_list)
print('Received message: ', binary_to_string)

quantum_engine.flush()


# bin_mess = 'a'
# print(ord(bin_mess))
# print(bin(ord(bin_mess)))
# print(bin(ord(bin_mess))[2:])
# print(bin(ord(bin_mess))[2:].zfill(8))

# bin_result = bin(ord(bin_mess))[2:].zfill(8)
# print(chr(int(bin_result, 2)))

from projectq import MainEngine
from projectq.ops import All, CNOT, H, Measure, X, Z

quantum_engine = MainEngine()


def create_bell_pair(quantum_engine):
    # Newly created Qubits are in the base state of 0,
    qubit_one = quantum_engine.allocate_qubit()
    qubit_two = quantum_engine.allocate_qubit()
    H | qubit_one
    # Measure | qubit_one
    # qubit_one_val = int(qubit_one)

    CNOT | (qubit_one, qubit_two)
    # Measure | qubit_two
    # cnot_val = int(qubit_two)

    return qubit_one, qubit_two

def create_message(quantum_engine='', qubit_one='', message_value=0):
    qubit_to_send = quantum_engine.allocate_qubit()
    if message_value == 1:
        '''
        setting the qubit to positive if message_value is 1
        by flipping the base state with a Pauli-X gate.
        '''
        X | qubit_to_send

    # entangle the original qubit with the message qubit
    CNOT | (qubit_to_send, qubit_one)

    '''
    1 - Put the message qubit in superposition
    2 - Measure out the two values to get the classical bit value
        by collapsing the state.
    '''
    H | qubit_to_send
    Measure | qubit_to_send
    Measure | qubit_one

    # The qubits are now turned into normal bits we can send through classical channels
    classical_encoded_message = [int(qubit_to_send), int(qubit_one)]

    return classical_encoded_message


def message_reciever(quantum_engine, message, qubit_two):
    '''
    Pauli-X and/or Pauli-Z gates are applied to the Qubit,
    conditionally on the values in the message.
    '''
    if message[1] == 1:
        X | qubit_two
    if message[0] == 1:
        Z | qubit_two

    '''
    Measuring the Qubit and collapsing the state down to either 1 or 0
    '''
    Measure | qubit_two

    quantum_engine.flush()

    received_bit = int(qubit_two)
    return received_bit


qubit_one, qubit_two = create_bell_pair(quantum_engine)
classical_encoded_message = create_message(
    quantum_engine=quantum_engine, qubit_one=qubit_one, message_value=0)

print('classical_encoded_message = ', classical_encoded_message)

received_bit = message_reciever(
    quantum_engine=quantum_engine, message=classical_encoded_message, qubit_two=qubit_two)

print('received_bit = ', str(received_bit))


def send_receive(bit=0, quantum_engine=''):
    # Create bell pair
    qubit_one, qubit_two = create_bell_pair(quantum_engine)
    # entangle the bit with the first qubit
    classical_encoded_message = create_message(
        quantum_engine=quantum_engine, qubit_one=qubit_one, message_value=bit)
    # print('send_bit = ', classical_encoded_message)
    # Teleport the bit and return it back
    received_bit = message_reciever(
        quantum_engine, classical_encoded_message, qubit_two)
    # print('received_bit = ', received_bit)
    return received_bit


message = 'HelloWorld'
binary_encoded_message = [bin(ord(x))[2:].zfill(8) for x in message]
print('Message to send: ', message)
print('Binary message to send: ', binary_encoded_message)

received_bytes_list = []
for letter in binary_encoded_message:
    received_bits = ''
    for bit in letter:
        received_bits = received_bits + \
            str(send_receive(int(bit), quantum_engine))
    received_bytes_list.append(received_bits)

binary_to_string = ''.join([chr(int(x, 2)) for x in received_bytes_list])
print('Received Binary message: ', received_bytes_list)
print('Received message: ', binary_to_string)

quantum_engine.flush()


# bin_mess = 'a'
# print(ord(bin_mess))
# print(bin(ord(bin_mess)))
# print(bin(ord(bin_mess))[2:])
# print(bin(ord(bin_mess))[2:].zfill(8))

# bin_result = bin(ord(bin_mess))[2:].zfill(8)
# print(chr(int(bin_result, 2)))



import numpy as np
import matplotlib.pyplot as plt
from qutip import *
# Generate a random quantum state vector
num_qubits = 1  # Number of qubits
state = rand_ket(2 ** num_qubits)

# Visualize the state using a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sphere = Bloch(axes=ax)
sphere.add_states(state)
sphere.show()
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
# Generate a random quantum state vector
num_qubits = 1  # Number of qubits
state = rand_ket(2 ** num_qubits)

# Alternatively, visualize the state using a Bloch sphere representation
bloch = Bloch()
bloch.add_states(state)
bloch.show()

# Show the plots
plt.show()

...
# head
print(dataset.head(20))

...
# descriptions
print(dataset.describe())

