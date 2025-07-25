# Packages
import torch
import torch.nn.functional as F

# Load data
words = open('names.txt').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# Create the dataset
xs, ys = [], []
for w in words[:-1]:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    ix3 = stoi[ch3]
    xs.append((ix1, ix2))
    ys.append(ix3)

trainlen = int(len(xs)*0.8)
devlen = int(len(xs)*0.1)

# Train dataset
X_train = torch.tensor(xs[:trainlen])
Y_train = torch.tensor(ys[:trainlen])

# Validation dataset
X_dev = torch.tensor(xs[trainlen:trainlen+devlen])
Y_dev = torch.tensor(ys[trainlen:trainlen+devlen])

# Unseen data for testing
X_test = torch.tensor(xs[trainlen+devlen:])
Y_test = torch.tensor(ys[trainlen+devlen:])

print(f'X_train: {X_train.shape}')
print(f'X_dev: {X_dev.shape}')
print(f'X_test: {X_test.shape}')

# Initialize the 'network'
g = torch.Generator().manual_seed(2147483647)
W1 = torch.randn((27, 27), generator=g, requires_grad=True)
W2 = torch.randn((27, 27), generator=g, requires_grad=True)

# Gradient descent (training a net!)
def train():
  for k in range(100):
    ix1 = X_train[:, 0]
    ix2 = X_train[:, 1]
    x1 = W1[ix1]
    x2 = W2[ix2]
    logits = x1 + x2 # predict log-counts
    counts = logits.exp() # counts, equivalent to N
    probs = counts / counts.sum(1, keepdim=True) # probabilities for next character
    
    loss = F.cross_entropy(logits, Y_train)
    loss += 0.01 * (W1**2).mean()
    loss += 0.01 * (W2**2).mean()
    
    probs = logits.softmax(dim=1)
    logprobs = probs.log()
    loss2 = -logprobs[range(len(Y_train)), Y_train].mean()
    loss += 0.01 * (W1**2).mean()
    loss += 0.01 * (W2**2).mean()

    print(f'Loss on train data: {loss.item()} {loss2.item()}')
        
    # Backward pass
    W1.grad = None # set to zero the gradient
    W2.grad = None # set to zero the gradient
    loss.backward()
    
    # Update weights
    W1.data += -50 * W1.grad
    W2.data += -50 * W2.grad

# Sample from the 'neural net' model
def sample():
  g = torch.Generator().manual_seed(2147483647)
  
  for i in range(10):
    out = []
    ix1 = 0
    ix2 = 0
    while True:
      x1 = W1[ix1]
      x2 = W2[ix2]
      logits = x1 + x2 # predict log-counts
      counts = logits.exp() # counts, equivalent to N
      p = counts / counts.sum() # probabilities for next character
      
      ix3 = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
      out.append(itos[ix3])
      if ix3 == 0:
        break
      ix1, ix2 = ix2, ix3
    print(''.join(out))

# Evaluate model
def evaluate(X, Y):
    ix1 = X[:, 0]
    ix2 = X[:, 1]
    x1 = W1[ix1]
    x2 = W2[ix2]
    logits = x1 + x2
    loss = F.cross_entropy(logits, Y)  
    return loss.item()

train()
sample()
print(f'Loss on dev data: {evaluate(X_dev, Y_dev)}')
print(f'Loss on test data: {evaluate(X_test, Y_test)}')
