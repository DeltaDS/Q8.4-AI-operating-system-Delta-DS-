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

from projectq.ops import H, Measure
from projectq import MainEngine

# initialises a new quantum backend
quantum_engine = MainEngine()

# Create Quibit
qubit = quantum_engine.allocate_qubit()

# Using Hadamard gate put it in superposition
H | qubit

#  Measure Quibit
Measure | qubit

# print(int(qubit))
random_number = int(qubit)
print(random_number)

# Flushes the quantum engine from memory
quantum_engine.flush()

from projectq import MainEngine
from projectq.ops import All, CNOT, H, Measure, X, Z
from collections import OrderedDict

quantum_engine = MainEngine()
od = OrderedDict()

control = quantum_engine.allocate_qubit()
target = quantum_engine.allocate_qubit()

H | control
Measure | control
od['Control'] = int(control)

H | target
Measure | target
od['Target'] = int(target)

CNOT | (control, target)
Measure | target
od['CNOT'] = int(target)

quantum_engine.flush()


for key, value in od.items():
    print(key, value)

from projectq import MainEngine
from projectq.ops import All, CNOT, H, Measure, X, Z

quantum_engine = MainEngine()

def entangle(quantum_engine):

    control = quantum_engine.allocate_qubit()
    target = quantum_engine.allocate_qubit()
    H | control
    Measure | control
    control_val = int(control)

    CNOT | (control, target)
    Measure | target
    target_cnot_val = int(target)

    return control_val, target_cnot_val


bell_pair_list = []
for i in range(10):
    bell_pair_list.append(entangle(quantum_engine))
quantum_engine.flush()
print(bell_pair_list)

bell_pair_list = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Create the mesh in polar coordinates and compute corresponding Z.
r = np.linspace(0, 1.25, 50)
p = np.linspace(0, 2*np.pi, 50)
R, P = np.meshgrid(r, p)
Z = ((R**2 - 1)**2)

# Express the mesh in the cartesian system.
X, Y = R*np.cos(P), R*np.sin(P)

# Plot the surface.
ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)

# Tweak the limits and add latex math labels.
ax.set_zlim(0, 1)
ax.set_xlabel(r'$\phi_\mathrm{real}$')
ax.set_ylabel(r'$\phi_\mathrm{im}$')
ax.set_zlabel(r'$V(\phi)$')

plt.show()

import matplotlib.pyplot as plt

plt.figure()
plt.subplot(projection="aitoff")
plt.title("Aitoff")
plt.grid(True)

plt.figure()
plt.subplot(projection="hammer")
plt.title("Hammer")
plt.grid(True)

plt.figure()
plt.subplot(projection="lambert")
plt.title("Lambert")
plt.grid(True)

plt.figure()
plt.subplot(projection="mollweide")
plt.title("Mollweide")
plt.grid(True)

plt.show()


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



