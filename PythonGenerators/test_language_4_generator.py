import random

NUMBER_OF_SAMPLES_1 = 500
NUMBER_OF_SAMPLES_2 = 500
MAX_STR_LENGTH_1 = 5
MAX_STR_LENGTH_2 = 15

def generate_random_string(length):
    s = ''
    for x in range(length):
        s += chr(97 + random.randint(0, 2))
    return s


def is_ok(string):
    return (len(string) % 3 == 0)

f = open('dataset.txt', 'w')
for x in range(NUMBER_OF_SAMPLES_1):
    l = random.randint(1, MAX_STR_LENGTH_1)
    s = generate_random_string(l)
    targetValue = int(is_ok(s))
    f.write(s + ' ' + str(targetValue) + '\n')

for x in range(NUMBER_OF_SAMPLES_2):
    l = random.randint(MAX_STR_LENGTH_1 + 1, MAX_STR_LENGTH_2)
    s = generate_random_string(l)
    targetValue = int(is_ok(s))
    f.write(s + ' ' + str(targetValue) + '\n')
