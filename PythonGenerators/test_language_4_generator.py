import random

NUMBER_OF_SAMPLES_1 = 150
NUMBER_OF_SAMPLES_2 = 300
NUMBER_OF_SAMPLES_3 = 550
MAX_STR_LENGTH_1 = 5
MAX_STR_LENGTH_2 = 12
MAX_STR_LENGTH_3 = 12

def generate_random_string(length):
    s = ''
    for x in range(length):
        s += chr(97 + random.randint(0, 2))
    return s

def generate_right_queue(length):
    s = generate_random_string(length - 3)
    j = random.randint(0, length - 4)
    return s[:j] + 'bac' + s[j:]


def is_ok(string):
    result = False
    for x in range(len(string) - 2):
        if(string[x] == 'b' and string[x + 1] == 'a' and string[x + 2] == 'c'):
            result = True
            break
    return result

f = open('tl4_dataset.txt', 'w')
for x in range(NUMBER_OF_SAMPLES_1):
    l = random.randint(1, MAX_STR_LENGTH_1)
    s = generate_random_string(l);
    targetValue = int(is_ok(s))
    f.write(s + ' ' + str(targetValue) + '\n')

for x in range(NUMBER_OF_SAMPLES_2):
    l = random.randint(4, MAX_STR_LENGTH_2)
    s = generate_right_queue(l)
    targetValue = int(is_ok(s))
    f.write(s + ' ' + str(targetValue) + '\n')

for x in range(NUMBER_OF_SAMPLES_3):
    l = random.randint(MAX_STR_LENGTH_1 + 1, MAX_STR_LENGTH_3)
    s = generate_random_string(l)
    targetValue = int(is_ok(s))
    f.write(s + ' ' + str(targetValue) + '\n')
