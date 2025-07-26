# Sync with python shell
def r(): exec(open('mlp-notebook.py').read(), globals())

#import torch
#import torch.nn.functional as F
#import matplotlib.pyplot as plt # for making figures
#import random

# read in all the words
words = open('names.txt', 'r').read().splitlines()

# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# build the dataset
def build_dataset(words):
  block_size = 3 # context length: how many characters do we take to predict the next one?
  X, Y = [], []
  for w in words:
    #print(w)
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      #print(''.join(itos[i] for i in context), '--->', itos[ix])
      context = context[1:] + [ix] # crop and append
    
  X = torch.tensor(X)
  Y = torch.tensor(Y)
  return X, Y

# Splitting data into train, dev/validation/ test sets
# with the rate of 80%, 10%, 10% where train split is
# used for training model, dev split is used to tune
# hyperparameters to fit to the dev/validation set,
# finally test split is used to evaluate the model
# at the end. Careful! Use test split only a few times!

random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

# Create embedding look-up table C (Bengio paper)
#
# C is a first layer with no non-linearity
# (tanh would add non-linearity)
# C is a weight matrix
# we are encoding integers into one-hot and feeding
# them into a neural net and this first layer C
# embeds them.
#
# 27 chars are embedded into a 2 dimensional space
# in case C shape is (27, 2)
# in other words C embedds int
#
# Indexing C directly is much faster than doing
# one-hot encoding and the result is the same
C = torch.rand((27, 2))

# Tensor may simply be indexed
# print(C[5])

# Tensor may be indexed by array
# this would select several indicies
# print(C[[5,6,7]])

# Duplications are allowed
# print(C[[5,6,7,7,7,7,7]])

# Tensor may be indexed by another tensor
# print(C[torch.tensor([5,6,7])])

# Tensor may be indexed with multi-dimensional tensors
# C[X] - the shape of this is (32,3,2), where (32,3),
# (in this arch 3 is a block size)
# is the shape of X but we also retrieved 2 dimensions
# of C assuming C has shape (27,2) and X[13,2] has
# index of tensor(1) C[X][13,2] == C[1]

# Embedded input layer, essentially weights
# index by input training data, shape (32,3,2)
emb = C[X]

# Hidden layer, we have 2 dimensional embeddings * 3,
# which is 6, 100 is just a number of neurons
W1 = torch.randn((6, 100))

# Number of biases equals to number of neurons
# in hidden layer
b1 = torch.randn(100)

# What we'd like to do is emb @ W1 + b, but we
# cannot due to different shapes
# so we'd like to transform it the following way
# this, however would not generalize if we want to
# change the block size later
torch.cat([emb[:, 0, :], emb[:, 1, :], emb[:, 2, :]], 1)

# The following is equal to above but also allows
# having a variable number of chars within a block.
# Unbind dimension 1, e.g. index 1 of emb
torch.cat(torch.unbind(emb, 1), 1)

# Above is a proper way of doing it, but there is
# a better approach is to use view()
# torch.cat(torch.unbind(emb, 1), 1) == emb.view(32,6)

# We can do like this
# emb.view(emb.shape[0],6) @ W1 + b1
# but a better way is
# emb.view(-1,6) @ W1 + b1

# Hidden layer of activations
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)

# Output layer, hidden layer gives 100 inputs
# and the eventual output is just 27 chars
W2 = torch.randn((100, 27))
b2 = torch.randn(27)

# Output probabilities
logits = h @ W2 + b2
counts = logits.exp()
prob = counts / counts.sum(1, keepdim=True)

# Loss
loss = -prob[torch.arange(Y.shape[0]), Y].log().mean()

# ------------ now made respectable :) ---------------
Xtr.shape, Ytr.shape # dataset
g = torch.Generator().manual_seed(2147483647) # for reproducibility
C = torch.randn((27, 10), generator=g)
W1 = torch.randn((30, 200), generator=g)
b1 = torch.randn(200, generator=g)
W2 = torch.randn((200, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]
print(sum(p.nelement() for p in parameters)) # number of parameters in total

for p in parameters:
  p.requires_grad = True

# Candidate learning rates
lre = torch.linspace(-3, 0, 1000) # learning rate exponent
lrs = 10**lre # list of learning rate candidates
lri = []
lossi = []
stepi = []

for i in range(100000):
  # MINIBATCHES:
  # With minibatches the quality of a gradient
  # is lower, so the direction is not as reliable,
  # it's not the actual gradient direction, but
  # the gradient direction is good wenough even if
  # it's estimating only on 32 examples.
  # IT'S MUCH BETTER TO HAVE AN APPROXIMATE GRADIENT
  # AND JUST MAKE MORE STEPS COMPARED TO HAVING AN
  # EXACT GRADIENT AND TAKE FEWER STEPS!!!
  # Batch size affects noise of the loss
  
  # Minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (32,))
  
  # forward pass
  emb = C[Xtr[ix]] # (32, 3, 10)
  h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 200)
  logits = h @ W2 + b2 # (32, 27)
  loss = F.cross_entropy(logits, Ytr[ix])
  #print(loss.item())
  
  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()
  
  # update
  #lr = lrs[i]
  lr = 0.1 if i < 100000 else 0.01
  for p in parameters:
    p.data += -lr * p.grad

  # track stats
  #lri.append(lre[i])
  stepi.append(i)
  lossi.append(loss.log10().item())

# Loss on the entire dataset
emb = C[Xdev] # (32, 3, 2)
h = torch.tanh(emb.view(-1,30) @W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Ydev)
print(loss.item())

# visualize dimensions 0 and 1 of the embedding matrix C for all characters
plt.figure(figsize=(8,8))
plt.scatter(C[:,0].data, C[:,1].data, s=200)
for i in range(C.shape[0]):
  plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color='white')
plt.grid('minor')
plt.show()

# When train loss is around equal to dev loss
# this is called underfitting, in this case new
# layers may be added to net. When train loss is
# lower than dev loss it means model is overfitting -
# it memorizes training data but fails to generalize
# good enough and essentially fails to generate new data

# Concern: when tanh layer becomes much bigger it
# may happen that the bottleneck of the net is located
# in the embedded 2 dimensional layer - it may be the
# case that too many chars are going through these two
# dimensions and the neural net is not able to use that
# space effectively and that is, again, the bottleneck
# to the model performance

# In production one would have all the hyper parameters
# and run lots of experiments to find the best results
# After best hyperparameters has been found, one should
# once evaluate model on test set and use that data
# within a report in a paper or wherever.

# How to tune hyperparameters?
# 1. Increase the size of the hidden layer (tanh)
# 2. Change dimensionality of embedded layer
# 3. Chnage the number of chars in a block size
# 4. Change details of optimization, e.g.
#    - number of iterations
#    - learning rate and its decay
#    - batch size

# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
  out = []
  context = [0] * block_size # initialize with all ...
  while True:
    emb = C[torch.tensor([context])] # (1,block_size,d)
    h = torch.tanh(emb.view(1, -1) @ W1 + b1)
    logits = h @ W2 + b2
    probs = F.softmax(logits, dim=1)
    ix = torch.multinomial(probs, num_samples=1, generator=g).item()
    context = context[1:] + [ix]
    out.append(ix)
    if ix == 0:
      break
    
  print(''.join(itos[i] for i in out))  
