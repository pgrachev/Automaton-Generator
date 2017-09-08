import random

NUMBER_OF_SAMPLES_1 = 500
NUMBER_OF_SAMPLES_2 = 500
NUMBER_OF_SAMPLES_3 = 200
MAX_STR_LENGTH_1 = 8
MAX_STR_LENGTH_2 = 29
MAX_STR_LENGTH_3 = 29

def generate_random_string(length):
    s = ''
    for x in range(length):
        s += chr(97 + random.randint(0, 1))
    return s

def is_ok(string):
    return (len(string) % 2 == 0 or len(string) % 7 == 0)

f = open('tl6_dataset.txt', 'w')
for x in range(NUMBER_OF_SAMPLES_1):
    l = random.randint(1, MAX_STR_LENGTH_1)
    s = generate_random_string(l);
    targetValue = int(is_ok(s))
    f.write(s + ' ' + str(targetValue) + '\n')

for x in range(NUMBER_OF_SAMPLES_2):
    l = random.randint(1, MAX_STR_LENGTH_2)
    s = generate_random_string(l);
    targetValue = int(is_ok(s))
    f.write(s + ' ' + str(targetValue) + '\n')

for x in range(NUMBER_OF_SAMPLES_3):
    l = random.randint(1, MAX_STR_LENGTH_3)
    s = generate_random_string(l)
    while(is_ok(s)):
        l = random.randint(1, MAX_STR_LENGTH_3)
        s = generate_random_string(l);
    targetValue = int(is_ok(s))
    f.write(s + ' ' + str(targetValue) + '\n')
