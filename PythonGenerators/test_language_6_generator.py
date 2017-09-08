import random

NUMBER_OF_SAMPLES_1 = 100
NUMBER_OF_SAMPLES_2 = 900
MAX_STR_LENGTH_1 = 5
MAX_STR_LENGTH_2 = 12
MAX_STR_LENGTH_3 = 12

def generate_random_string(length):
    s = ''
    for x in range(length):
        s += chr(97 + random.randint(0, 2))
    return s

def automat(pos, char):
    if (pos == 0):
        if(char == 'a'):
            return 1
        if(char == 'b'):
            return 2
        if(char == 'c'):
            return 2
    if (pos == 1):
        if(char == 'a'):
            return 3
        if(char == 'b'):
            return 2
        if(char == 'c'):
            return 0
    if (pos == 2):
        if(char == 'a'):
            return 3
        if(char == 'b'):
            return 2
        if(char == 'c'):
            return 1
    if (pos == 3):
        if(char == 'a'):
            return 0
        if(char == 'b'):
            return 3
        if(char == 'c'):
            return 3

def is_ok(string):
    curr_pos = 0
    for i in range(len(string)):
        curr_pos = automat(curr_pos, string[i])
    return (curr_pos == 1 or curr_pos == 2)



neededRes = True
f = open('dataset.txt', 'w')
for x in range(NUMBER_OF_SAMPLES_1):
    l = random.randint(1, MAX_STR_LENGTH_1)
    s = generate_random_string(l);
    while (is_ok(s) != neededRes):
        l = random.randint(1, MAX_STR_LENGTH_1)
        s = generate_random_string(l);
    targetValue = int(is_ok(s))
    f.write(s + ' ' + str(targetValue) + '\n')
    neededRes = not neededRes

for x in range(NUMBER_OF_SAMPLES_2):
    l = random.randint(MAX_STR_LENGTH_1 + 1, MAX_STR_LENGTH_2)
    s = generate_random_string(l);
    while (is_ok(s) != neededRes):
        l = random.randint(MAX_STR_LENGTH_1 + 1, MAX_STR_LENGTH_2)
        s = generate_random_string(l);
    targetValue = int(is_ok(s))
    f.write(s + ' ' + str(targetValue) + '\n')
    neededRes = not neededRes
