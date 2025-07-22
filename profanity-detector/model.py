#Packages
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_len = 22           # what is the longest word in dataset to encode?
batch_size = 64        # how many independent sequences will we process in parallel?
n_embd = 128           # embedding size per character
n_hidden = 64          # size of LSTM hidden state
n_layer = 1            # number of LSTM layers
max_iters = 2000       # total number of batches trained
eval_iters = 200       # number of iterations to evaluate
eval_interval = 100    # validation print interval
learning_rate = 0.0001 # by how much weights update each iteration?

# Same results across different platforms
torch.manual_seed(1337)

# Load data
with open('input.txt') as f: lines = f.read().splitlines()
words, labels = [], []
for line in lines:
  word, label = line.strip().split(',')
  words.append(word.lower())
  labels.append(int(label))

# Build voacabulary
chars = sorted(list(set(''.join(words))))
vocab_size = len(chars)+1
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Encode a single word
def encode_word(w):
  encoded = [stoi.get(c, 0) for c in w]
  if len(encoded) < max_len:
    encoded += [0] * (max_len - len(encoded))
  else:
    encoded = encoded[:max_len]
  return encoded

# Model
class LSTM(nn.Module):
    def __init__(self, vocab_size, n_embd, n_hidden, n_layer):
      super().__init__()
      self.embedding = nn.Embedding(vocab_size, n_embd, padding_idx=0)             # maps chars to vectors, ignores padding
      self.lstm = nn.LSTM(n_embd, n_hidden, num_layers=n_layer, batch_first=True)  # processes char sequences
      self.fc = nn.Linear(n_hidden, 1)                                             # outputs single logit

    def forward(self, x, targets=None):
      x = self.embedding(x)
      _, (hn, _) = self.lstm(x)
      logits = self.fc(hn[-1]).squeeze(1)
      loss = None
      if targets is not None:
          pos_weight = torch.tensor([20], device=logits.device)
          loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
      return logits, loss
