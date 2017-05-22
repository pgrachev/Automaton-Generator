import random
import numpy as np

DICTIONARY = ['a', 'b']
NUMBER_OF_POSITIONS = 3
NUMBER_OF_CHARS = len(DICTIONARY)
PERCENT_OF_TESTS = 0.1
PERCENT_OF_BATCH = 0.05
MAX_RANDOM_VALUE = 0.1
EPS = 0.01
NU = 0.8

class NeuralNetwork:
    tensor = np.zeros([NUMBER_OF_POSITIONS, NUMBER_OF_CHARS, NUMBER_OF_POSITIONS])
    adder = np.zeros(NUMBER_OF_POSITIONS)
    start_position = np.zeros(NUMBER_OF_POSITIONS)

    def __init__(self):
        self.tensor = np.random.rand(NUMBER_OF_POSITIONS, NUMBER_OF_CHARS, NUMBER_OF_POSITIONS) * MAX_RANDOM_VALUE
        self.adder = np.full(NUMBER_OF_POSITIONS, 0.5) + np.random.rand(NUMBER_OF_POSITIONS) * MAX_RANDOM_VALUE - np.full(NUMBER_OF_POSITIONS, 0.5 * MAX_RANDOM_VALUE)
        self.adder[0] = 0.0
        self.adder[-1] = 1.0
        self.start_position[0] = 1.0

    def check(self, word):
        curr_pos = self.start_position
        for k in range(len(word)):
            curr_word = char_to_vector(word[k])
            curr_pos = match(self, curr_word, curr_pos)
            curr_pos = softmax(curr_pos)
        return lastsum(self, curr_pos)

    def train(self, word, exp):
        word_length = len(word)
        positions = np.zeros([word_length + 1, NUMBER_OF_POSITIONS])
        d_tensor = np.zeros([NUMBER_OF_POSITIONS, NUMBER_OF_CHARS, NUMBER_OF_POSITIONS])
        d_adder = np.zeros(NUMBER_OF_POSITIONS)
        positions[0] = self.start_position;

        for k in range(word_length):
            curr_word = char_to_vector(word[k])
            positions[k + 1] = softmax(match(self, curr_word, positions[k]))
        answer = lastsum(self, positions[-1])
        error = cost_function(exp, answer)

        gradient = cost_function_derrivative(exp, answer)
        d_adder += lastsum_derrivative_adder(gradient, positions[-1])
        gradient = lastsum_derrivative(self, gradient)
        for k in range(word_length - 1, -1, -1):
            curr_word = char_to_vector(word[k])
            gradient = softmax_derrivative(gradient, positions[k + 1])
            d_tensor += match_derrivative_tensor(gradient, curr_word, positions[k])
            gradient = match_derrivative(self, gradient, curr_word)
        self.tensor -= NU * d_tensor
        self.adder -= NU * d_adder

        return error

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

def softmax(z):
    t = np.exp(z)
    return t / np.sum(t)

def softmax_derrivative(dz, softmaxes):
    derrivative = np.zeros([NUMBER_OF_POSITIONS, NUMBER_OF_POSITIONS])
    for i in range(NUMBER_OF_POSITIONS):
        for j in range(NUMBER_OF_POSITIONS):
            if(i == j):
                derrivative[i][j] = softmaxes[i] * (1 - softmaxes[j])
            else:
                derrivative[i][j] = -1.0 * softmaxes[i] * softmaxes[j]
    return np.dot(dz, derrivative)

def lastsum(nn, x):
    return np.dot(nn.adder, x)

def lastsum_derrivative(nn, dz):
    return np.dot(dz, nn.adder)

def lastsum_derrivative_adder(dz, inp):
    return np.dot(dz, inp)

def char_to_vector(ch):
    index = DICTIONARY.index(ch)
    vec = np.zeros(NUMBER_OF_CHARS)
    vec[index] = 1.0
    return vec

def naive_convergence_test(nn, steps):
    for i in range(steps):
        print 'step ' + str(i)
        print 'error a: ' + str(nn.train("a", 0.0))
        print 'error b: ' + str(nn.train("b", 1.0))

nn = NeuralNetwork();
naive_convergence_test(nn, 11)