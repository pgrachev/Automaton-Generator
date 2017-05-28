import random
import numpy as np

DICTIONARY = ['a', 'b']
NUMBER_OF_POSITIONS = 4
NUMBER_OF_CHARS = len(DICTIONARY)
PERCENT_OF_TESTS = 0.1
PERCENT_OF_BATCH = 0.01
MAX_RANDOM_VALUE = 1.0
GRAD_CONST = 1.0
EPS = 0.01
NU = 0.2



class NeuralNetwork:
    tensor = np.zeros([NUMBER_OF_POSITIONS, NUMBER_OF_CHARS, NUMBER_OF_POSITIONS])
    adder = np.zeros(NUMBER_OF_POSITIONS)
    start_position = np.zeros(NUMBER_OF_POSITIONS)

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
            self.adder[to] = 1.0

        self.start_position[0] = 1.0


    def check(self, word):
        curr_pos = self.start_position
        for k in range(len(word)):
            curr_word = char_to_vector(word[k])
            curr_pos = match(self, curr_word, curr_pos)
            curr_pos = normalize(curr_pos)
        return lastsum(self, curr_pos)

    def train_online(self, dataset):
        average_error = 1.0
        epoch_number = 0
        n = len(dataset)
        train_size = int((1.0 - PERCENT_OF_TESTS) * n)
        tests_size = int(PERCENT_OF_TESTS * n)
        while (average_error > EPS):
            random.shuffle(dataset)
            left_set = dataset
            epoch_number += 1
            print 'Epoch #' + str(epoch_number)
            while(len(left_set) > tests_size):
                train_case, left_set = left_set[0], left_set[1:]
                self.train(train_case[0], train_case[1], online=True)
            average_error = 0.0
            for i in range(len(left_set)):
                average_error += cost_function(left_set[i][1], self.check(left_set[i][0]))
            average_error /= len(left_set)
            print "Average error: " + str(average_error)
        #    print self.tensor
            print self.adder

    def train_dataset(self, dataset):
        average_error = 1.0
        epoch_number = 0
        n = len(dataset)
        batch_size = int(PERCENT_OF_BATCH * n)
        tests_size = int(PERCENT_OF_TESTS * n)
        while (average_error > EPS):
            random.shuffle(dataset)
            left_set = dataset
            epoch_number += 1
            print 'Epoch #' + str(epoch_number)
            while(len(left_set) >= tests_size + batch_size):
                current_batch, left_set = left_set[:batch_size], left_set[batch_size:]
                self.train_batch(current_batch)
            average_error = 0.0
            for i in range(len(left_set)):
                average_error += cost_function(left_set[i][1], self.check(left_set[i][0]))
            average_error /= len(left_set)
            print "Average error: " + str(average_error)
        #    print self.tensor
        #    print self.adder

    def train_batch(self, batch):
        d_tensor = np.zeros([NUMBER_OF_POSITIONS, NUMBER_OF_CHARS, NUMBER_OF_POSITIONS])
        d_adder = np.zeros(NUMBER_OF_POSITIONS)
        cut_v = np.vectorize(cut)
        error = 0.0
        batch_size = len(batch)
        for i in range(batch_size):
            result = self.train(batch[i][0], batch[i][1])
            d_tensor += result[0]
            d_adder += result[1]
            error += result[2]
        self.tensor = cut_v(self.tensor - NU * (d_tensor / batch_size))
        self.adder = cut_v(self.adder - NU * (d_adder / batch_size))
        return error / batch_size



    def train(self, word, exp, online = False):
        cut_v = np.vectorize(cut)
        word_length = len(word)
        positions = np.zeros([word_length + 1, NUMBER_OF_POSITIONS])
        before_normalize = np.zeros([word_length, NUMBER_OF_POSITIONS])
        d_tensor = np.zeros([NUMBER_OF_POSITIONS, NUMBER_OF_CHARS, NUMBER_OF_POSITIONS])
        d_adder = np.zeros(NUMBER_OF_POSITIONS)
        positions[0] = self.start_position;

        for k in range(word_length):
            curr_word = char_to_vector(word[k])
            before_normalize[k] = match(self, curr_word, positions[k])
            positions[k + 1] = normalize(before_normalize[k])
        answer = lastsum(self, positions[-1])
        error = cost_function(exp, answer)

        gradient = cost_function_derrivative(exp, answer)
        d_adder += lastsum_derrivative_adder(gradient, positions[-1])
        gradient = lastsum_derrivative(self, gradient)
        print 'grad: ' + str(sum(abs(gradient)))
        print '-----------------'
        for k in range(word_length - 1, -1, -1):
            curr_word = char_to_vector(word[k])
            gradient = GRAD_CONST * normalize_derrivative(gradient, before_normalize[k])

            print 'grad1: ' + str(sum(abs(gradient)))
            d_tensor += match_derrivative_tensor(gradient, curr_word, positions[k])
            print d_tensor[1][0][1]
            gradient = match_derrivative(self, gradient, curr_word)
      #      print gradient
            print 'grad2: ' + str(sum(abs(gradient)))
        if(online):
            self.tensor = cut_v(self.tensor - NU * d_tensor)
            self.adder = cut_v(self.adder - NU * d_adder)
  #          print 'grad2: ' + str(gradient)

     #   print self.tensor
     #   print self.adder
     #   print error
     #   print d_tensor
     #   print d_adder
        return d_tensor, d_adder, error

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
  #  print 'inp' + str(inp)
    sum = np.sum(inp)
    for i in range(NUMBER_OF_POSITIONS):
        for j in range(NUMBER_OF_POSITIONS):
            if(i == j):
                derrivative[i][j] = (sum - inp[i]) / (sum ** 2)
            else:
                derrivative[i][j] = -inp[i] / (sum ** 2)
  #  print 'derr!' + str(derrivative)
  #  print dz
  #  print np.dot(dz, derrivative)
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

def cut(x):
    if (x > 1.0):
        return 1.0
    if (x < 0.0):
        return 0.0
    return x

def naive_convergence_test(nn, steps):
    for i in range(steps):
        print 'step ' + str(i)
        print 'error a: ' + str(nn.train("a", 0.0))
        print 'error b: ' + str(nn.train("b", 1.0))

nn = NeuralNetwork();
f = open('dataset.txt', 'r')
cnt = 0
mid = 0.0
dataset = []
for line in f:
    arr = line.split(' ')
    isOk = 1.0
    if arr[1][0] == '0':
        isOk = 0.0
    dataset.append([arr[0], isOk])

nn.train_online(dataset)

f = open('checkset.txt', 'r')
cnt = 0
mid = 0.0
dataset = []
for line in f:
    arr = line.split(' ')
    isOk = 1.0
    if arr[1][0] == '0':
        isOk = 0.0
    dataset.append([arr[0], isOk])


for i in range(len(dataset)):
    print "Test #" + str(i + 1) + ", word: " + str(dataset[i][0]) + ", error = " + str(cost_function(dataset[i][1], nn.check(dataset[i][0])))

nn.get_automaton()