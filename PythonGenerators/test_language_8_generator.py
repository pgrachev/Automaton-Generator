import random

NUMBER_OF_SAMPLES_1 = 650
NUMBER_OF_SAMPLES_2 = 350
MAX_STR_LENGTH_1 = 10
MAX_STR_LENGTH_2 = 10

def generate_random_string(length):
    s = ''
    for x in range(length):
        s += chr(97 + random.randint(0, 4))
    return s

def generate_right_queue(length):
    s = generate_random_string(length)
    return ''.join(sorted(s))

def w(char):
	if(char == 'a'):
		return 1
	if(char == 'b'):
		return 2
	if(char == 'c'):
		return 3
	if(char == 'd'):
		return 4
	if(char == 'e'):
		return 5

def is_ok(string):
    result = True
    for x in range(len(string) - 1):
        if(w(string[x]) > w(string[x + 1])):
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
    l = random.randint(1, MAX_STR_LENGTH_2)
    s = generate_right_queue(l)
    targetValue = int(is_ok(s))
    f.write(s + ' ' + str(targetValue) + '\n')

