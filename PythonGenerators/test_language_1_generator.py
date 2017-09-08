import random

NUMBER_OF_SAMPLES_1 = 10
NUMBER_OF_SAMPLES_2 = 200
NUMBER_OF_SAMPLES_3 = 200
NUMBER_OF_SAMPLES_4 = 200
NUMBER_OF_TESTS_1 = 5
NUMBER_OF_TESTS_2 = 5
MAX_STR_LENGTH_1 = 3
MAX_STR_LENGTH_2 = 8
MAX_STR_LENGTH_3 = 8
MAX_STR_LENGTH_4 = 8
TEST_STR_LENGTH_RANGE = [5, 8]

def generate_random_string(length, posibility_of_b):
    s = ''
    for x in range(length):
        random_float_value = random.random()
        if (random_float_value < posibility_of_b):
            s += 'b'
        else:
            s += chr(97 + 2 * random.randint(0, 1))
    return s

def is_ok(string):
    result = True
    for x in range(len(string)):
        if(string[x] == 'b'):
            result = False
            break
    return result

f = open('dataset.txt', 'w')
for x in range(NUMBER_OF_SAMPLES_1):
    l = random.randint(1, MAX_STR_LENGTH_1)
    s = generate_random_string(l, 1.0 / (l * 3.0));
    targetValue = int(is_ok(s))
    f.write(s + ' ' + str(targetValue) + '\n')

for x in range(NUMBER_OF_SAMPLES_2):
    l = random.randint(MAX_STR_LENGTH_1 + 1, MAX_STR_LENGTH_2)
    s = generate_random_string(l, 1.0 / (l * 3.0))
    targetValue = int(is_ok(s))
    f.write(s + ' ' + str(targetValue) + '\n')

for x in range(NUMBER_OF_SAMPLES_3):
    l = random.randint(MAX_STR_LENGTH_1 + 1, MAX_STR_LENGTH_3)
    s = generate_random_string(l, 0.0)
    targetValue = int(is_ok(s))
    f.write(s + ' ' + str(targetValue) + '\n')

for x in range(NUMBER_OF_SAMPLES_4):
    l = random.randint(MAX_STR_LENGTH_1 + 1, MAX_STR_LENGTH_4)
    s = generate_random_string(l, 0.0)
    v = random.randint(0, len(s) - 1)
    s = s[:v] + 'b' + s[v:]
    targetValue = int(is_ok(s))
    f.write(s + ' ' + str(targetValue) + '\n')

f1 = open('checkset.txt', 'w')
for x in range(NUMBER_OF_TESTS_1):
    l = random.randint(TEST_STR_LENGTH_RANGE[0], TEST_STR_LENGTH_RANGE[1])
    s = generate_random_string(l, 0.0);
    targetValue = int(is_ok(s))
    f1.write(s + ' ' + str(targetValue) + '\n')

for x in range(NUMBER_OF_TESTS_2):
    l = random.randint(TEST_STR_LENGTH_RANGE[0], TEST_STR_LENGTH_RANGE[1])
    s = generate_random_string(l, 1.0 / (l * 3.0));
    while (is_ok(s)):
        s = generate_random_string(l, 1.0 / 3.0);
    targetValue = int(is_ok(s))
    f1.write(s + ' ' + str(targetValue) + '\n')