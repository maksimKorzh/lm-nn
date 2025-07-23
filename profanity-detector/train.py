import torch
from model import *

# Encode data
X = torch.tensor([encode_word(w) for w in words], dtype=torch.long)
Y = torch.tensor(labels, dtype=torch.float)

# Split data into train/test subsets
split_idx = int(0.9 * len(X))
X_train, Y_train = X[:split_idx], Y[:split_idx]
X_val,   Y_val   = X[split_idx:], Y[split_idx:]

# Data loading
def get_batch(split):
  data_x = X_train if split == 'train' else X_val
  data_y = Y_train if split == 'train' else Y_val
  ix = torch.randint(len(data_x), (batch_size,))
  x = data_x[ix].to(device)
  y = data_y[ix].to(device)
  return x, y

# Evaluate model
@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
      out[split] = losses.mean()
  model.train()
  return out

# Init model
model = LSTM(vocab_size, n_embd, n_hidden, n_layer).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
  # Every once in a while evaluate the loss on train and val sets
  if iter % eval_interval == 0 or iter == max_iters - 1:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

  # Sample a batch of data
  xb, yb = get_batch('train')

  # Evaluate the loss
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

# Save model
torch.save(model.state_dict(), 'profanity_model.pt')
