# Packages
import torch
import matplotlib.pyplot as plt

# Load data
words = open('names.txt').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

# Init statistical counts
N = torch.zeros((27, 27, 27), dtype=torch.int32)
for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    ix3 = stoi[ch3]
    N[ix1, ix2, ix3] += 1
    #print(ch1, ch2, ch3)

# Init probability distributions
P = (N+3).float()
P = P.view(27*27, 27)
P /= P.sum(1, keepdim=True)

# Preserve similar value across different systems
g = torch.Generator().manual_seed(2147483647)

# Sample data
for i in range(5):
  out = []
  ix1, ix2 = 0, 0
  while True:
    ix = ix1*27+ix2
    p = P[ix]
    ix3 = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix3])
    if ix3 == 0:
      break
    ix1, ix2 = ix2, ix3
  print(''.join(out))

# likelihood is a product of all probabilities
log_likelihood = 0.0
n = 0
P = P.view(27,27,27)
for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    ix3 = stoi[ch3]
    prob = P[ix1, ix2, ix3]
    logprob = torch.log(prob) # log to scale up probabilities
    log_likelihood += logprob
    n += 1
    #print(f'{ch1}{ch2}{ch3}: {prob:.4f} {logprob:.4f}')

print(f'{log_likelihood=}')
nll = -log_likelihood # negative log likelyhood
print(f'{nll=}')
print(f'{nll/n}') # normalized (average) negative log likely hood
