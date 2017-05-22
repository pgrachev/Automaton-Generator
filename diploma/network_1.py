import random
import numpy as np

DICTIONARY = ['a', 'b', 'c']
NUMBER_OF_POSITIONS = 5
HIDDEN_LAYER_SIZE = 5
PERCENT_OF_TESTS = 0.1
PERCENT_OF_BATCH = 0.05
EPS = 0.01

NU = 0.5

def create_network ():
    input_layer_size = NUMBER_OF_POSITIONS + len(DICTIONARY) + 1
    output_layer_size = NUMBER_OF_POSITIONS + 1
    w1 = []
    w2 = []
    for i in range(input_layer_size):
        w1.append([(2 * random.random() - 1) / (2 * HIDDEN_LAYER_SIZE) for j in range(HIDDEN_LAYER_SIZE)])
    for i in range(HIDDEN_LAYER_SIZE + 1):
        w2.append([(2 * random.random() - 1) / (2 * output_layer_size) for j in range(output_layer_size)])
    return [w1, w2]

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def softmax(lst):
    t = []
    for i in range(len(lst)):
        t.append(np.exp(lst[i]))
    return t / np.sum(t)

def softmax_derivative(lst, fr, to):
    t = softmax(lst)
    if (fr == to):
        result = t[fr] * (1 - t[fr])
    else:
        result = -1.0 * t[fr] * t[to]
    return result

def train(network, word, isOk):
  #  print 'train:'
   # print word
    layers = [[0.0 for x in range(NUMBER_OF_POSITIONS + 1)] for y in range(len(word) + 1)]
    error1 = [[0.0 for x in range(HIDDEN_LAYER_SIZE)] for y in range(len(word) + 1)]
    error2 = [[0.0 for x in range(NUMBER_OF_POSITIONS)] for y in range(len(word) + 1)]
    error = [error1, error2]
    local_NU = NU
    outputs = [[0.0 for x in range(NUMBER_OF_POSITIONS + 1)] for y in range(len(word) + 1)]
    hidden_layer = [[0.0 for x in range(HIDDEN_LAYER_SIZE)] for y in range(len(word) + 1)]
    hidden_output = [[0.0 for x in range(HIDDEN_LAYER_SIZE + 1)] for y in range(len(word) + 1)]
    input_layer = [[0.0 for x in range(NUMBER_OF_POSITIONS + len(DICTIONARY) + 1)] for y in range(len(word) + 1)]
    layers[0] = [1.0] + [0.0] * (NUMBER_OF_POSITIONS)
    for k in range(len(word)):
    #    print word[k]
        char_index = DICTIONARY.index(word[k])
        input_layer[k][char_index] = 1.0
        for i in range(len(layers[k])):
            input_layer[k][i + len(DICTIONARY)] = layers[k][i]
        input_layer[k][-1] = 1.0 #bias

        for j in range(len(input_layer[k])):
            for i in range(HIDDEN_LAYER_SIZE):
                hidden_layer[k + 1][i] += input_layer[k][j] * network[0][j][i]
        for i in range(HIDDEN_LAYER_SIZE):
            hidden_output[k + 1][i] = sigmoid(hidden_layer[k + 1][i])
        hidden_output[k + 1][-1] = 1.0 #bias
     #  print hidden_output[k + 1]
     #   print network[1]
        for j in range(HIDDEN_LAYER_SIZE + 1):
            for i in range(NUMBER_OF_POSITIONS + 1):
                outputs[k + 1][i] += hidden_output[k + 1][j] * network[1][j][i]
      #          print 'hl: ' + str(hidden_output[k + 1][j]) + ', nw: ' + str(network[1][j][i])
        tmp = softmax(outputs[k + 1][:-1])
        for i in range(len(tmp)):
            layers[k + 1][i] = tmp[i]
        layers[k + 1][-1] = outputs[k + 1][-1]

    if (isOk):
        final_error = layers[-1][-1] - 1
    else:
        final_error = layers[-1][-1] - 0

    for k in range(len(word) - 1, -1, -1):

        if(k + 1 == len(word)):
            for fr in range(HIDDEN_LAYER_SIZE + 1):
                network[1][fr][-1] -= local_NU * final_error * hidden_output[k + 1][fr]
            for i in range(len(error[0][k + 1])):
                error[0][k + 1][i] += final_error * network[1][i][-1]
        else:
            for to in range(NUMBER_OF_POSITIONS):
                err = error[1][k + 1][to]
                for fr in range(HIDDEN_LAYER_SIZE + 1):
                    for to2 in range(NUMBER_OF_POSITIONS):
                        network[1][fr][to2] -= local_NU * err * softmax_derivative(outputs[k + 1][:-1], to2, to) * hidden_output[k + 1][fr]
            for fr in range(HIDDEN_LAYER_SIZE + 1):
                for to in range(NUMBER_OF_POSITIONS):
                    for to2 in range(NUMBER_OF_POSITIONS):
                        error[0][k + 1][to2] += error[1][k + 1][to] * softmax_derivative(outputs[k + 1][:-1], to2, to) * network[1][fr][to]
        for fr in range(len(input_layer[k])):
            for to in range(HIDDEN_LAYER_SIZE):
                network[0][fr][to] -= local_NU * (error[0][k + 1][to]) * sigmoid_derivative(hidden_layer[k + 1][to]) * input_layer[k][fr]
    #    print 'step' + str(k)
    #    print 'final_error = ' + str(final_error)
        for i in range(len(error[1][k])):
            for to in range(HIDDEN_LAYER_SIZE):
            #    print str(i) + ":"
            #    print error[0][k + 1][i]
            #    print sigmoid_derivative(hidden_layer[k + 1][to])
            #    print network[0][i][to]
                error[1][k][i] += error[0][k + 1][i] * sigmoid_derivative(hidden_layer[k + 1][to]) * network[0][i][to]


    return final_error ** 2


def check (network, word, isOk, flag=False):
  #  print word
    position = [1.0] + [0.0] * (NUMBER_OF_POSITIONS - 1);
    for k in range(len(word)):
  #      print word[k]
        input_layer = [0.0] * (NUMBER_OF_POSITIONS + len(DICTIONARY) + 1)
        output_layer = [0.0] * (NUMBER_OF_POSITIONS + 1)
        char_index = DICTIONARY.index(word[k])
        input_layer[char_index] = 1.0
        for i in range(len(position)):
            input_layer[i + len(DICTIONARY)] = position[i]
        input_layer[-1] = 1.0

        hidden_layer = [0.0] * (HIDDEN_LAYER_SIZE + 1)
        for j in range(len(input_layer)):
            for i in range(HIDDEN_LAYER_SIZE):
                hidden_layer[i] += input_layer[j] * network[0][j][i]
        for i in range(HIDDEN_LAYER_SIZE):
            hidden_layer[i] = sigmoid(hidden_layer[i])
        hidden_layer[-1] = 1.0
        for j in range(HIDDEN_LAYER_SIZE + 1):
            for i in range(NUMBER_OF_POSITIONS + 1):
                output_layer[i] += hidden_layer[j] * network[1][j][i]
        position = softmax(output_layer[:-1])
        if(flag):
            res = 0.0
            res += (position[0] - 1) ** 2
            for i in range(len(position) - 1):
                res += position[i + 1] ** 2
            print res
            return res

     #   print output_layer[-1]
    if(isOk):
        final_error = (1 - output_layer[-1]) ** 2
    else:
        final_error = output_layer[-1] ** 2
    return final_error


def cost_function(exp, real):
    result = 0.0
    for i in range(len(exp)):
        result += (exp[i] - real[i])**2
    return result



nn = create_network()

delta = 0.01
x1 = check(nn, "a", True, flag=True)
nn[1][1][2] += delta
x2 = check(nn, "a", True, flag=True)
print (x2 - x1)
print (x2 - x1) / delta

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
