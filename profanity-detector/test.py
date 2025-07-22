import torch
from model import LSTM

# Hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_len = 22           # what is the longest word in dataset to encode?
batch_size = 64        # how many independent sequences will we process in parallel?
n_embd = 128           # embedding size per character
n_hidden = 64          # size of LSTM hidden state
n_layer = 1            # number of LSTM layers
max_iters = 500        # total number of batches trained
eval_iters = 200       # number of iterations to evaluate
eval_interval = 50     # validation loss every
learning_rate = 0.001  # by how much weights update each iteration?

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

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM(vocab_size, n_embd, n_hidden, n_layer).to(device)
model.load_state_dict(torch.load('profanity_model.pt', map_location=device))
model.eval()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# Main loop
print("Type a word to check if it's profanity. Type 'exit' to quit.")
while True:
    word = input("Enter word: ").strip()
    if word.lower() == 'exit':
        break
    x = torch.tensor(encode_word(word), dtype=torch.long).unsqueeze(0).to(device) # (1, max_len)
    print(x)
    with torch.no_grad():
        logit, _ = model(x)
        prob = torch.sigmoid(logit)
        print(prob)
        print("Profanity" if prob > 0.5 else "Not profanity")
