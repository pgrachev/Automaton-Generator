import random
import numpy as np

DICTIONARY = ['a', 'b', 'c']
HIDDEN_LAYER_SIZE = 10
NUMBER_OF_POSITIONS = 3

def create_network ():
    input_layer_size = NUMBER_OF_POSITIONS + len(DICTIONARY)
    output_layer_size = NUMBER_OF_POSITIONS
    w1 = []
    w2 = []
    for i in range(input_layer_size):
        w1.append([2 * random.random() - 1 for j in range(HIDDEN_LAYER_SIZE + 1)])
    for i in range(HIDDEN_LAYER_SIZE):
        w2.append([2 * random.random() - 1 for j in range(output_layer_size + 1)])
    return [w1, w2]

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def normalize(lst):
    result = [0.0] * len(lst)
    sum_exp = 0.0
    for i in range(len(lst)):
        result[i] = np.exp(lst[i])
        sum_exp += result[i]
    for i in range(len(lst)):
        result[i] /= sum_exp
    return result

def count_level(network, char_index, positions):
    first_layer_input = [0.0] * (char_index) + [1.0] + [0.0] * (len(DICTIONARY) - char_index - 1) + positions
    hidden_layer_input = [0.0] * HIDDEN_LAYER_SIZE
    new_positions = [0.0] * NUMBER_OF_POSITIONS

    for j in range(len(first_layer_input)):
        bias = network[0][j][HIDDEN_LAYER_SIZE]
        for i in range(HIDDEN_LAYER_SIZE):
            hidden_layer_input[i] += sigmoid(first_layer_input[j] * network[0][j][i] + bias)

    for j in range(HIDDEN_LAYER_SIZE):
        bias = network[1][j][NUMBER_OF_POSITIONS]
        for i in range(len(positions)):
            new_positions[i] += sigmoid(hidden_layer_input[j] * network[1][j][i])

    return normalize(new_positions)

def initialize(network, word):
    position = [1.0] + [0.0] * (NUMBER_OF_POSITIONS - 1)
    for i in range(len(word)):
        position = count_level(network, DICTIONARY.index(word[i]), position)
    return position


nn = create_network()
print initialize(nn, "abacaba")

