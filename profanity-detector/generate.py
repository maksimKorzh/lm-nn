import requests
import random
import os

# Raw datasets
raw_good = []
raw_bad = []

# Filtered datasets
good_words = []
bad_words = []
all_words = []

# Filter words
strong_swear = [
 'fuck', 'cunt', 'shit', 'dick', 'wank', 'cum',
 'wtf', 'bollock', 'bitch', 'arse', 'ass', 'slut',
 'whore', 'cock', 'bollick', 'tit', 'twat', 'ball',
 'piss', 'suck', 'toss', 'pussy', 'anal', 'jiz',
 'jig', 'nigg', 'knob', 'jerk'
]

# Download datasets
print('Downloading raw datasets')
bads = requests.get('https://www.cs.cmu.edu/~biglou/resources/bad-words.txt').text
goods = requests.get('https://www.cs.cmu.edu/~biglou/resources/EN_SP_DICT.txt').text
with open('bad_words.txt', 'w') as f: f.write(bads)
with open('good_words.txt', 'w') as f: f.write(goods)

# Load raw data
with open('good_words.txt') as f: raw_good = [w.split('||')[0] for w in f.read().splitlines()]
with open('bad_words.txt') as f: raw_bad = f.read().splitlines()

# Filter bad words
for w in raw_bad:                                                                                                                                 
  for sw in strong_swear:                                                                                                                       
    if sw in w:                                                                                                                                 
      bad_words.append(w) 

bad_words = list(set(bad_words))
#for i in range(10): [bad_words.append(sw) for sw in strong_swear]

# Filter good words
for w in raw_good:
  if w not in bad_words and len(w.split(' ')) == 1:
    good_words.append(w)  

good_words = list(set(good_words))

# Build training dataset
#for w in good_words: all_words.append(w + ',0')
#for w in bad_words: all_words.append(w + ',1')
for i in range(len(bad_words)):
  all_words.append(bad_words[i] + ',1')
  all_words.append(good_words[i] + ',0')
  all_words.append(good_words[i+len(good_words)//len(bad_words)] + ',0')
print(f'{len(good_words)} good words, {len(bad_words)} bad words')
for i in range(3): random.shuffle(all_words)

# Save training data & clean up
with open('input.txt', 'w') as f: f.write('\n'.join(all_words))
print('Created training data')
os.remove('bad_words.txt')
os.remove('good_words.txt')
print('Cleaned up raw files')
