import random
import numpy as np

DICTIONARY = ['a', 'b', 'c']
NUMBER_OF_POSITIONS = 6
NU = 4.0

def create_network ():
    input_layer_size = NUMBER_OF_POSITIONS + len(DICTIONARY)
    output_layer_size = NUMBER_OF_POSITIONS
    w = []
    for i in range(input_layer_size):
        w.append([(2 * random.random() - 1) / (2 * NUMBER_OF_POSITIONS) for j in range(output_layer_size)])
    return w

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def normalize(lst):
    result = [0.0] * len(lst)
    xmin = 100.0
    xmax = -100.0
    for i in range(len(lst)):
        if(xmin > lst[i]):
            xmin = lst[i]
        if(xmax < lst[i]):
            xmax = lst[i]
    for i in range(len(lst)):
        result[i] = (lst[i] - xmin) / (xmax - xmin)
    return result / (sum(result))


def train(network, word, result):
    layers = [[0.0 for x in range(NUMBER_OF_POSITIONS)] for y in range(len(word) + 1)]
    error = [[0.0 for x in range(NUMBER_OF_POSITIONS)] for y in range(len(word) + 1)]
    outputs = [[0.0 for x in range(NUMBER_OF_POSITIONS)] for y in range(len(word) + 1)]
    input_layer = [[0.0 for x in range(NUMBER_OF_POSITIONS + len(DICTIONARY))] for y in range(len(word) + 1)]
    layers[0] = [1.0] + [0.0] * (NUMBER_OF_POSITIONS - 1)
    for k in range(len(word)):
        char_index = DICTIONARY.index(word[k])
        input_layer[k][char_index] = 1.0
        for i in range(len(layers[k])):
            input_layer[k][i + len(DICTIONARY)] = layers[k][i]
        for j in range(len(input_layer[k])):
            for i in range(NUMBER_OF_POSITIONS):
                outputs[k + 1][i] += input_layer[k][j] * network[j][i]
        for i in range(NUMBER_OF_POSITIONS):
            layers[k + 1][i] = sigmoid(outputs[k + 1][i])
        layers[k + 1] = normalize(layers[k + 1]);
    for j in range(len(layers[-1])):
        error[-1][j] = layers[-1][j] - result[j]
    for k in range(len(word) - 1, -1, -1):
        for fr in range(len(input_layer[k])):
            for to in range(len(network[j])):
                network[fr][to] -= NU * (error[k + 1][to]) * sigmoid_derivative(outputs[k + 1][to]) * input_layer[k][fr]
        for i in range(len(error[k])):
            for to in range(len(network[j])):
                error[k][i] += error[k + 1][to] * sigmoid_derivative(outputs[k + 1][to]) * network[i][to]
    return sum(map(lambda x: x * x, error[-1]))


def initialize (network, word):
    position = [1.0] + [0.0] * (NUMBER_OF_POSITIONS - 1);
    for k in range(len(word)):
        input_layer = [0.0] * (NUMBER_OF_POSITIONS + len(DICTIONARY))
        char_index = DICTIONARY.index(word[k])
        input_layer[char_index] = 1.0
        for i in range(len(position)):
            input_layer[i + len(DICTIONARY)] = position[i]
        output_layer = [0.0] * NUMBER_OF_POSITIONS
        for j in range(len(input_layer)):
            for i in range(NUMBER_OF_POSITIONS):
                output_layer[i] += input_layer[j] * network[j][i]
        for i in range(NUMBER_OF_POSITIONS):
            output_layer[i] = sigmoid(output_layer[i])
        position = normalize(output_layer)
    print position

def cost_function(exp, real):
    result = 0.0
    for i in range(len(exp)):
        result += (exp[i] - real[i])**2
    return result



nn = create_network()

f = open('dataset.txt', 'r')
for line in f:
    arr = line.split(' ')
    res_mas = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    if arr[1][0] == '0':
        res_mas = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    train(nn, arr[0], res_mas)

initialize(nn, "a")
initialize(nn, "b")
initialize(nn, "c")
initialize(nn, "aacaa")
initialize(nn, "abbba")
initialize(nn, "aabaa")
initialize(nn, "abacaba")
