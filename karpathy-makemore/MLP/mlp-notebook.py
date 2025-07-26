# Sync with python shell
def r(): exec(open('mlp-notebook.py').read(), globals())

#import torch
#import torch.nn.functional as F
#import matplotlib.pyplot as plt # for making figures

# read in all the words
words = open('names.txt', 'r').read().splitlines()

# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?
X, Y = [], []
for w in words[:5]:
  print(w)
  context = [0] * block_size
  for ch in w + '.':
    ix = stoi[ch]
    X.append(context)
    Y.append(ix)
    print(''.join(itos[i] for i in context), '--->', itos[ix])
    context = context[1:] + [ix] # crop and append
  
X = torch.tensor(X)
Y = torch.tensor(Y)

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
print(C[5])

# Tensor may be indexed by array
# this would select several indicies
print(C[[5,6,7]])

# Duplications are allowed
print(C[[5,6,7,7,7,7,7]])

# Tensor may be indexed by another tensor
print(C[torch.tensor([5,6,7])])

# Tensor may be indexed with multi-dimensional tensors
print(C[X])

# The shape of this is (32,3,2), where (32,3) is the
# shape of X but we also retrieved 2 dimensions of C
# assuming C has shape (27,2)

print(C[X].shape)
