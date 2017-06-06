import random
import numpy as np

DICTIONARY = ['a', 'b', 'c']
NUMBER_OF_POSITIONS = 4
START_POSITION = np.zeros(NUMBER_OF_POSITIONS)
START_POSITION[0] = 1
NUMBER_OF_CHARS = len(DICTIONARY)
PERCENT_OF_TESTS = 0.1
TenzToAdd = 2.0
EPS = 0.01
NU = 0.7
NU_ADDER = 0.3

class NeuralNetwork:

    tensor = np.zeros([NUMBER_OF_POSITIONS, NUMBER_OF_CHARS, NUMBER_OF_POSITIONS])
    adder = np.zeros(NUMBER_OF_POSITIONS)

    def __init__(self):
        self.tensor = np.zeros([NUMBER_OF_POSITIONS, NUMBER_OF_CHARS, NUMBER_OF_POSITIONS])
        for fr in range(NUMBER_OF_POSITIONS):
            for ch in range(NUMBER_OF_CHARS):
                z = np.random.rand(NUMBER_OF_POSITIONS)
                z = normalize(z)
                for to in range(NUMBER_OF_POSITIONS):
                    self.tensor[to][ch][fr] = z[to]
        self.adder = np.zeros(NUMBER_OF_POSITIONS)
        for to in range(NUMBER_OF_POSITIONS):
            self.adder[to] = 1.0 * (to + 1) / (NUMBER_OF_POSITIONS)


    def check(self, word):
        curr_pos = START_POSITION
        for k in range(len(word)):
            curr_word = char_to_vector(word[k])
            curr_pos = match(self, curr_word, curr_pos)
            curr_pos = normalize(curr_pos)
        return lastsum(self, curr_pos)

    def train_online(self, dataset):
        average_error = 1.0
        epoch_number = 0
        n = len(dataset)
        tests_size = int(PERCENT_OF_TESTS * n)
        while (average_error > EPS):
            random.shuffle(dataset)
            cases_left = len(dataset)
            epoch_number += 1
            print 'Epoch #' + str(epoch_number)
            while(cases_left > tests_size):
                self.train(dataset[cases_left - 1][0], dataset[cases_left - 1][1])
                cases_left -= 1
            average_error = 0.0
            for i in range(cases_left):
                average_error += cost_function(dataset[i][1], self.check(dataset[i][0]))
            average_error /= cases_left
            print "Average error: " + str(average_error)

    def train(self, word, exp):
        cut_v = np.vectorize(cut)
        word_length = len(word)
        positions = np.zeros([word_length + 1, NUMBER_OF_POSITIONS])
        before_normalize = np.zeros([word_length, NUMBER_OF_POSITIONS])
        d_tensor = np.zeros([NUMBER_OF_POSITIONS, NUMBER_OF_CHARS, NUMBER_OF_POSITIONS])
        d_adder = np.zeros(NUMBER_OF_POSITIONS)
        positions[0] = START_POSITION

        for k in range(word_length):
            curr_word = char_to_vector(word[k])
            before_normalize[k] = match(self, curr_word, positions[k])
            positions[k + 1] = normalize(before_normalize[k])
        answer = lastsum(self, positions[-1])
        error = cost_function(exp, answer)


        gradient = cost_function_derrivative(exp, answer)
        d_adder += lastsum_derrivative_adder(self, gradient, positions[-1])
        gradient = lastsum_derrivative(self, gradient)
        first_gradient = sum(abs(gradient)) * TenzToAdd
        for k in range(word_length - 1, -1, -1):
            curr_grad = sum(abs(gradient))
            if (curr_grad < 0.001):
                koef = 1.0
            else:
                koef = first_gradient / sum(abs(gradient))
            gradient *= koef
            curr_word = char_to_vector(word[k])
            gradient = normalize_derrivative(gradient, before_normalize[k])
            d_tensor += match_derrivative_tensor(gradient, curr_word, positions[k])
            gradient = match_derrivative(self, gradient, curr_word)

        d_tensor /= word_length
        self.tensor = cut_v(self.tensor - NU * d_tensor)
        self.adder = cut_v(self.adder - NU * NU_ADDER * d_adder)
        return error

    def get_automaton(self):
        for j in range(NUMBER_OF_POSITIONS):
            for i in range(NUMBER_OF_CHARS):
                max_ind = 0
                for k in range(1, NUMBER_OF_POSITIONS):
                    if (nn.tensor[k][i][j] > nn.tensor[max_ind][i][j]):
                        max_ind = k
                print str(j) + "--" + str(DICTIONARY[i]) + '-->' + str(max_ind)
        for k in range(NUMBER_OF_POSITIONS):
            if (nn.adder[k] > 0.5):
                print str(k) + " is terminal"

def cost_function(exp, res):
    return (res - exp) ** 2

def cost_function_derrivative(exp, res):
    return res - exp

def match(nn, ch, pos):
    new_pos = np.zeros(NUMBER_OF_POSITIONS)
    for k in range(NUMBER_OF_POSITIONS):
        for i in range (NUMBER_OF_CHARS):
            for j in range (NUMBER_OF_POSITIONS):
                new_pos[k] += nn.tensor[k][i][j] * ch[i] * pos[j]
    return new_pos

def match_derrivative(nn, dz, ch):
    derrivative = np.zeros([NUMBER_OF_POSITIONS, NUMBER_OF_POSITIONS])
    for i in range(NUMBER_OF_POSITIONS):
        for k in range(NUMBER_OF_POSITIONS):
            for j in range (NUMBER_OF_CHARS):
                derrivative[k][i] += nn.tensor[k][j][i] * ch[j]
    return np.dot(dz, derrivative)


def match_derrivative_tensor(dz, ch, pos):
    sample_matrix = np.zeros([NUMBER_OF_CHARS, NUMBER_OF_POSITIONS])
    for i in range(NUMBER_OF_CHARS):
        for j in range(NUMBER_OF_POSITIONS):
            sample_matrix[i][j] = ch[i] * pos[j]
    derrivative = np.zeros([NUMBER_OF_POSITIONS, NUMBER_OF_CHARS, NUMBER_OF_POSITIONS])
    for i in range(NUMBER_OF_POSITIONS):
        derrivative[i] = dz[i] * sample_matrix
    return derrivative

def normalize(t):
    return t / np.sum(t)

def normalize_derrivative(dz, inp):
    derrivative = np.zeros([NUMBER_OF_POSITIONS, NUMBER_OF_POSITIONS])
    sum = np.sum(inp)
    for i in range(NUMBER_OF_POSITIONS):
        for j in range(NUMBER_OF_POSITIONS):
            if(i == j):
                derrivative[i][j] = (sum - inp[i]) / (sum ** 2)
            else:
                derrivative[i][j] = -inp[i] / (sum ** 2)
    return np.dot(dz, derrivative)

def lastsum(nn, x):
    return np.dot(nn.adder, x)

def lastsum_derrivative(nn, dz):
    return np.dot(dz, nn.adder)

def lastsum_derrivative_adder(nn, dz, inp):
    derrivative = np.multiply(inp, nn.adder)
    return np.dot(dz, derrivative)

def char_to_vector(ch):
    index = DICTIONARY.index(ch)
    vec = np.zeros(NUMBER_OF_CHARS)
    vec[index] = 1.0
    return vec

def cut(x):
    if (x > 1.0):
        return 1.0
    if (x < 0.0):
        return 0.0
    return x


f = open('dataset.txt', 'r')
dataset = []
for line in f:
    arr = line.split(' ')
    isOk = 1.0
    if arr[1][0] == '0':
        isOk = 0.0
    dataset.append([arr[0], isOk])


nn = NeuralNetwork();
nn.train_online(dataset)
nn.get_automaton()

