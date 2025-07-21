###########################
#
#  Generate training data
#
###########################

# Packages
import os
import requests
import string
import random

# Download datasets
if not os.path.isfile('bad_words.txt'):
  print('Downloading datasets...')
  bad_words = requests.get('https://www.cs.cmu.edu/~biglou/resources/bad-words.txt').text
  good_words = requests.get('https://www.cs.cmu.edu/~biglou/resources/EN_SP_DICT.txt').text
  with open('bad_words.txt', 'w') as f: f.write(bad_words)
  with open('good_words.txt', 'w') as f: f.write(good_words)

# Training data
words = []

# Format training data
with open('bad_words.txt') as f: [words.append(w + ',1') for w in f.read().splitlines()]
with open('good_words.txt') as f: [words.append(w.split('||')[0] + ',0') for w in f.read().splitlines()]

words = [w for w in words if all(c in string.ascii_letters for c in w.split(',')[0])]
random.shuffle(words)

with open('input.txt', 'w') as f: f.write('\n'.join(words))
print('Created training data')
