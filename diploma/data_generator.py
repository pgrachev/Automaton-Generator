import random

NUMBER_OF_SAMPLES_1 = 2000
NUMBER_OF_SAMPLES_2 = 2000
MAX_STR_LENGTH_1 = 10
MAX_STR_LENGTH_2 = 20

def generate_random_string(length):
    s = ''
    posibility_of_b = 1.0 / (3 * length)
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
    randomString = generate_random_string(l)
    targetValue = int(is_ok(randomString))
    f.write(randomString + ' ' + str(targetValue) + '\n')

for x in range(NUMBER_OF_SAMPLES_2):
    l = random.randint(MAX_STR_LENGTH_1, MAX_STR_LENGTH_2)
    randomString = generate_random_string(l)
    targetValue = int(is_ok(randomString))
    f.write(randomString + ' ' + str(targetValue) + '\n')

