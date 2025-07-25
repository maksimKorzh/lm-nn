# Packages
import torch
import torch.nn.functional as F
import random

# Load data
words = open('names.txt').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# Create the datasets
random.shuffle(words)
xs, ys = [], []

for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    xs.append(ix1)
    ys.append(ix2)

trainlen = int(len(xs)*0.8)
devlen = int(len(xs)*0.1)

# Train dataset
X_train = torch.tensor(xs[:trainlen-1])
Y_train = torch.tensor(ys[:trainlen-1])
num = X_train.nelement()

# Validation dataset
X_dev = torch.tensor(xs[trainlen:-devlen])
Y_dev = torch.tensor(ys[trainlen:-devlen])

# Unseen data for testing
X_test = torch.tensor(xs[trainlen+devlen+1:])
Y_test = torch.tensor(ys[trainlen+devlen+1:])

print(f'X_train: {X_train.shape}')
print(f'X_dev: {X_dev.shape}')
print(f'X_test: {X_test.shape}')

# Gradient descent (training a net!)
g = torch.Generator().manual_seed(2147483647)

# Initialize the 'network'
W = torch.randn((27, 27), generator=g, requires_grad=True)

def train():

  for k in range(100):
    # Forward pass
    xenc = F.one_hot(X_train, num_classes=27).float() # input to the network: one-hot encoding
    logits = xenc @ W # predict log-counts
    counts = logits.exp() # counts, equivalent to N
    probs = counts / counts.sum(1, keepdim=True) # probabilities for next character
    # 0.01*(W**2).mean() is a loss regularization
    # 0.01 is regularization strength, it's like controlling N + how much in array mode
    # It's a gravity force that pushes W to be 0
    loss = -probs[torch.arange(num), Y_train].log().mean() + 0.01*(W**2).mean()
    print(f'Loss on train data: {loss.item()}')
    
    # Backward pass
    W.grad = None # set to zero the gradient
    loss.backward()
    
    # Update weights
    W.data += -50 * W.grad

# Sample from the 'neural net' model
def sample():
  g = torch.Generator().manual_seed(2147483647)
  
  for i in range(5):
    out = []
    ix = 0
    while True:
      xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
      logits = xenc @ W # predict log-counts
      counts = logits.exp() # counts, equivalent to N
      p = counts / counts.sum(1, keepdims=True) # probabilities for next character
      
      ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
      out.append(itos[ix])
      if ix == 0:
        break
    print(''.join(out))

def evaluate(xs, ys):
  num = xs.nelement()
  xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
  logits = xenc @ W # predict log-counts
  counts = logits.exp() # counts, equivalent to N
  probs = counts / counts.sum(1, keepdim=True) # probabilities for next character
  loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
  return loss.item()

train()
print(f'Loss on dev data: {evaluate(X_dev, Y_dev)}')
print(f'Loss on test data: {evaluate(X_test, Y_test)}')
