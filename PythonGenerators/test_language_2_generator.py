import random

NUMBER_OF_SAMPLES_1 = 50
NUMBER_OF_SAMPLES_2 = 750
NUMBER_OF_SAMPLES_3 = 200
NUMBER_OF_SAMPLES_4 = 500
MAX_STR_LENGTH_1 = 3
MAX_STR_LENGTH_2 = 10
MAX_STR_LENGTH_3 = 10
MAX_STR_LENGTH_4 = 10

def generate_random_string(length):
    s = ''
    for x in range(length):
        s += chr(97 + random.randint(0, 2))
    return s

def generate_right_queue(length):
    s = ''
    x = random.random()
    DICT = ['a', 'b', 'c']
    last = ''
    s = ''
    if (x > 0.66):
        s += 'c'
        last = 'c'
    else:
        if (x > 0.33):
            s += 'b'
            last = 'b'
        else:
            s += 'a'
            last = 'a'
    while(len(s) != length):
        local_dict = ['a', 'b', 'c']
        local_dict.remove(last)
        x = random.random()
        if (x > 0.5):
            s += local_dict[1]
            last = local_dict[1]
        else:
            s += local_dict[0]
            last = local_dict[0]
    return s

def generate_quasiright_queue(length):
    s = generate_right_queue(length - 1)
    j = random.randint(0, length - 2)
    return s[:j] + s[j] + s[j:]

def is_ok(string):
    result = True
    for x in range(len(string) - 1):
        if(string[x] == string[x + 1]):
            result = False
            break
    return result

f = open('dataset.txt', 'w')
for x in range(NUMBER_OF_SAMPLES_1):
    l = random.randint(1, MAX_STR_LENGTH_1)
    s = generate_random_string(l);
    targetValue = int(is_ok(s))
    f.write(s + ' ' + str(targetValue) + '\n')

for x in range(NUMBER_OF_SAMPLES_2):
    l = random.randint(MAX_STR_LENGTH_1 + 1, MAX_STR_LENGTH_2)
    s = generate_right_queue(l)
    targetValue = int(is_ok(s))
    f.write(s + ' ' + str(targetValue) + '\n')

for x in range(NUMBER_OF_SAMPLES_3):
    l = random.randint(MAX_STR_LENGTH_1 + 1, MAX_STR_LENGTH_3)
    s = generate_random_string(l)
    targetValue = int(is_ok(s))
    f.write(s + ' ' + str(targetValue) + '\n')

for x in range(NUMBER_OF_SAMPLES_4):
    l = random.randint(MAX_STR_LENGTH_1 + 1, MAX_STR_LENGTH_4)
    s = generate_quasiright_queue(l)
    targetValue = int(is_ok(s))
    f.write(s + ' ' + str(targetValue) + '\n')
