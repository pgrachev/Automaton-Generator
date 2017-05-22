import random
import numpy as np

DICTIONARY = ['a', 'b', 'c']
NUMBER_OF_POSITIONS = 4
NUMBER_OF_CHARS = len(DICTIONARY)
PERCENT_OF_TESTS = 0.1
PERCENT_OF_BATCH = 0.05
MAX_RANDOM_VALUE = 0.1
EPS = 0.01
NU = 0.4

class NeuralNetwork:
    tensor = np.empty(NUMBER_OF_POSITIONS, NUMBER_OF_CHARS, NUMBER_OF_POSITIONS)
    adder = np.empty(NUMBER_OF_POSITIONS)
    def __init__(self):
        self.tensor = np.random.rand(NUMBER_OF_POSITIONS, NUMBER_OF_CHARS, NUMBER_OF_POSITIONS) * MAX_RANDOM_VALUE
        self.adder = np.random.rand(NUMBER_OF_POSITIONS) * ()

def softmax(z):
    t = np.exp(z)
    return t / np.sum(t)

def softmax_derrivative(z):
    t = [0.0 for x in range(len(z))]
    for i in range(len(z)):
        for j in range(len(z)):
            if (i == j):
                delta = z[i] * (1 - z[i])
            else:
                delta = - z[i] * z[j]
            t[i] += delta
    return t


def lastsum(nn, x):
    return np.dot(nn.adder, x)

def lastsum_derrivative(nn, dz):
    return np.dot(dz, nn.adder)

def lastsum_derrivative_adder(np, dz, inp):
    return np.dot()

def W1_derrivative_w(z, inp):
    n = len(nn[1])
    m = len(nn[1][0])
    w = np.zeros[n, m]
    for i in range(n):
        for j in range(m):
            w[i][j] = z[j] * inp[i]
    return w

def W0_derrivative_w(z, inp):
    n = len(z)
    m = len(inp)
    w = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            w[i][j] = z[i] * inp[j]
    return w

def train(network, word, isOk):
    layers = [[0.0 for x in range(NUMBER_OF_POSITIONS + 1)] for y in range(len(word) + 1)]
    error1 = [[0.0 for x in range(HIDDEN_LAYER_SIZE)] for y in range(len(word) + 1)]
    error2 = [[0.0 for x in range(NUMBER_OF_POSITIONS)] for y in range(len(word) + 1)]
    error = [error1, error2]
    local_NU = NU
    outputs = [[0.0 for x in range(NUMBER_OF_POSITIONS + 1)] for y in range(len(word) + 1)]
    hidden_layer = [[0.0 for x in range(HIDDEN_LAYER_SIZE + 1)] for y in range(len(word) + 1)]
    hidden_output = [[0.0 for x in range(HIDDEN_LAYER_SIZE + 1)] for y in range(len(word) + 1)]
    input_layer = [[0.0 for x in range(NUMBER_OF_POSITIONS + len(DICTIONARY) + 1)] for y in range(len(word) + 1)]
    output_layer = [[0.0 for x in range(NUMBER_OF_POSITIONS + 1)] for y in range(len(word) + 1)]
    final_output = 0.0
    layers[0] = [1.0] + [0.0] * (NUMBER_OF_POSITIONS)
    for k in range(len(word)):
        char_index = DICTIONARY.index(word[k])
        input_layer[k][char_index] = 1.0
        for i in range(len(layers[k])):
            input_layer[k][i + len(DICTIONARY)] = layers[k][i]
        input_layer[k][-1] = 1.0 #bias

        hidden_layer[k] = sigmoid(W0(input_layer[k]))
        hidden_layer[k].append(1.0)


        if (k == len(word) - 1):
            final_output = unary_sigmoid(lastsum(hidden_layer[k]))
        else:
            output_layer[k] = softmax(W1(hidden_layer[k]))

    final_error = (final_output - 1.0 * int(isOk == True))
    grad = final_error
    print "grad0" + str(grad)
    error = final_error ** 2

    dw0 = np.zeros([HIDDEN_LAYER_SIZE, NUMBER_OF_POSITIONS + len(DICTIONARY) + 1])
    dw1 = np.zeros([NUMBER_OF_POSITIONS + 1, HIDDEN_LAYER_SIZE + 1])
    for k in range(len(word) - 1, -1, -1):
         if(k + 1 == len(word)):
             grad = unary_sigmoid_derrative(grad)
             print "grad1" + str(grad)
             dw = lastsum_derrivative_w(grad, hidden_layer[k])
             for fr in range(HIDDEN_LAYER_SIZE + 1):
                 dw1[-1][fr] += dw[fr];
             grad = lastsum_derrivative(grad)
         else:
             grad = softmax_derrivative(grad)
             dw = W1_derrivative_w(grad, hidden_layer[k])
             for fr in range(HIDDEN_LAYER_SIZE + 1):
                 for to in range(NUMBER_OF_POSITIONS):
                    dw1[fr][to] += dw[fr][to]
             grad = W1_derrivative(grad)
         grad = np.delete(grad, -1)
         print "grraaaaaaaaaaaaaaad: " + str(grad)
         grad = sigmoid_derrative(grad)
         print "grraaaaaaaaaaaaaaad: " + str(grad)
         dw = W0_derrivative_w(grad, input_layer[k])
         for fr in range(NUMBER_OF_POSITIONS + len(DICTIONARY) + 1):
             for to in range(HIDDEN_LAYER_SIZE):
                 dw0[to][fr] += dw[to][fr]
         grad = W0_derrivative(grad)

    print dw0
    for fr in range(NUMBER_OF_POSITIONS + len(DICTIONARY) + 1):
        for to in range(HIDDEN_LAYER_SIZE):
         #   print str(fr) + " " + str(to) + " " + str(dw0[fr][to])
            nn[0][to][fr] -= local_NU * dw0[to][fr]
    print dw1
    for fr in range(HIDDEN_LAYER_SIZE + 1):
        for to in range(NUMBER_OF_POSITIONS + 1):
            nn[1][to][fr] -= local_NU * dw1[to][fr]


    return error


def check (network, word, isOk):
    position = [1.0] + [0.0] * (NUMBER_OF_POSITIONS - 1);
    error = 0.0
    res = 0.0
    for k in range(len(word)):
        input_layer = [0.0] * (NUMBER_OF_POSITIONS + len(DICTIONARY) + 1)
        output_layer = [0.0] * (NUMBER_OF_POSITIONS + 1)
        char_index = DICTIONARY.index(word[k])
        input_layer[char_index] = 1.0
        for i in range(len(position)):
            input_layer[i + len(DICTIONARY)] = position[i]
        input_layer[-1] = 1.0 #bias

        hidden_layer = sigmoid(W0(input_layer))
        print hidden_layer
        hidden_layer.append(1.0)

        if(k == len(word) - 1):
            res = unary_sigmoid(lastsum(hidden_layer))
        else:
            output_layer = softmax(W1(hidden_layer))
            print output_layer
            position = output_layer[:-1]

    if(isOk):
        error = (1 - res) ** 2
    else:
        error = res ** 2
    return error



print train(nn, "a", True)
print train(nn, "a", True)
print train(nn, "a", True)
print train(nn, "a", True)
print train(nn, "a", True)
print train(nn, "a", True)
print train(nn, "a", True)
print train(nn, "a", True)
print train(nn, "a", True)
print train(nn, "a", True)
print train(nn, "a", True)
print train(nn, "a", True)
print check(nn, "a", True)


#f = open('dataset.txt', 'r')
#cnt = 0
#mid = 0.0
#dataset = []
#for line in f:
#    arr = line.split(' ')
#    isOk = True
#    if arr[1][0] == '0':
#        isOk = False
#    dataset.append([arr[0], isOk])

#average_error = 1.0
#epoch_number = 0
#while (average_error > EPS):
 #   random.shuffle(dataset)
 #   epoch_number += 1
 #   print 'Epoch #' + str(epoch_number)
 #   number_of_trains = int((1.0 - PERCENT_OF_TESTS) * len(dataset))
 #   for i in range(number_of_trains):
 #       train(nn, dataset[i][0], dataset[i][1])
 #   average_error = 0.0
 #   for i in range(number_of_trains, len(dataset) - 1):
 #       average_error += check(nn, dataset[i][0], dataset[i][1]) / (len(dataset) - number_of_trains - 1)
 #   check(nn, dataset[-1][0], dataset[-1][1], flag=True)
 #   print "Average error: " + str(average_error)


#f = open('checkset.txt', 'r')
#cnt = 0
#mid = 0.0
#dataset = []
#for line in f:
#    arr = line.split(' ')
#    isOk = True
#    if arr[1][0] == '0':
#        isOk = False
#    dataset.append([arr[0], isOk])

#for i in range(len(dataset)):
#    print "Test #" + str(i + 1) + ", word: " + str(dataset[i][0]) + ", error = " + str(check(nn, dataset[i][0], dataset[i][1]))
