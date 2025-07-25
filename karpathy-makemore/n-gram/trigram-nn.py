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
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = ys.nelement()
print('number of examples: ', num)

# Initialize the 'network'
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27+27, 27), generator=g, requires_grad=True)

  # Encode & cat two chars into a single one-hot input
def encode_input(t):
  ch1 = t[:, 0]
  ch2 = t[:, 1]
  #ch1 = torch.split(t, [1,1],1)[0].squeeze(1)
  #ch2 = torch.split(t, [1,1],1)[1].squeeze(1)
  enc1 = F.one_hot(ch1, num_classes=27).float() # encode 1st char
  enc2 = F.one_hot(ch2, num_classes=27).float() # encode 2st char
  xenc = torch.cat((enc1, enc2), 1)
  return xenc

# Gradient descent (training a net!)
for k in range(100):
  xenc = encode_input(xs)
  logits = xenc @ W # predict log-counts
  counts = logits.exp() # counts, equivalent to N
  probs = counts / counts.sum(1, keepdim=True) # probabilities for next character

  # 0.01*(W**2).mean() is a loss regularization
  # 0.01 is regularization strength, it's like controlling N + how much in array mode
  # It's a gravity force that pushes W to be 0
  loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
  print(loss.item())
  
  # Backward pass
  W.grad = None # set to zero the gradient
  loss.backward()
  
  # Update weights
  W.data += -50 * W.grad

# Sample from the 'neural net' model
g = torch.Generator().manual_seed(2147483647)

for i in range(10):
  out = []
  ix1 = 0
  ix2 = 0
  while True:
    input_vector = torch.cat(([torch.tensor([ix1]), torch.tensor([ix2])]), 0).unsqueeze(0)
    xenc = encode_input(input_vector)
    logits = xenc @ W # predict log-counts
    counts = logits.exp() # counts, equivalent to N
    p = counts / counts.sum(1, keepdims=True) # probabilities for next character
    
    ix3 = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix3])
    if ix3 == 0:
      break
    ix1, ix2 = ix2, ix3
  print(''.join(out))
