import torch

# Load data
with open('input.txt') as f: lines = f.read().splitlines()
words, labels = [], []
for line in lines:
  word, label = line.strip().split(',')
  words.append(word.lower())
  labels.append(int(label))

# Build voacabulary
chars = sorted(list(set(''.join(words))))

# Char encoding
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
