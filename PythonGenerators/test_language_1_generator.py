import random

NUMBER_OF_SAMPLES_1 = 50
NUMBER_OF_SAMPLES_2 = 475
NUMBER_OF_SAMPLES_3 = 475
NUMBER_OF_SAMPLES_4 = 500
MAX_STR_LENGTH_1 = 3
MAX_STR_LENGTH_2 = 10
MAX_STR_LENGTH_3 = 10
MAX_STR_LENGTH_4 = 10

def generate_random_string(length):
    s = ''
    for x in range(length):
        s += chr(97 + random.randint(0, 1))
    return s

def generate_right_queue(length):
    s = ''
    x = random.random()
    ch1 = 'a'
    ch2 = 'b'
    if(x > 0.5):
        ch1, ch2 = ch2, ch1
    cnt = 0;
    while (cnt != length):
        if(cnt % 2 == 0):
            s += ch1
        else:
            s += ch2
        cnt += 1
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

print generate_quasiright_queue(3)

f = open('tl1_dataset.txt', 'w')
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
