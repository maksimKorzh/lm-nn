import torch
from model import *

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM(vocab_size, n_embd, n_hidden, n_layer).to(device)
model.load_state_dict(torch.load('profanity_model.pt', map_location=device))
model.eval()
m = model.to(device)
print(sum(p.numel() for p in m.parameters()), 'parameters')

bad_count = 0
good_count = 0
bad_correct = 0
bad_wrong = 0
good_correct = 0
good_wrong = 0
threshold = 0.5

with open('input.txt') as f:
  words = f.read().splitlines()
  for w in words:
    x = torch.tensor(encode_word(w), dtype=torch.long).unsqueeze(0).to(device) # (1, max_len)
    with torch.no_grad():
      logit, _ = model(x)
      prob = torch.sigmoid(logit).item()
      if ',1' in w:
        bad_count += 1
        if prob > threshold: bad_correct += 1
        if prob < threshold:
          print(f'should be bad:\t{prob:.4f}\t{w.split(",")[0]}')
          bad_wrong += 1
      if ',0' in w:
        good_count += 1
        if prob < threshold: good_correct += 1
        if prob > threshold:
          print(f'should be good:\t{prob:.4f}\t{w.split(",")[0]}')
          good_wrong += 1

good_acc = good_correct / good_count * 100
bad_acc = bad_correct / bad_count * 100
overall_acc = (good_correct + bad_correct) / (good_count + bad_count) * 100

print(f"Bad accuracy: {bad_acc:.2f}%")
print(f"Good accuracy: {good_acc:.2f}%")
print(f"Overall accuracy: {overall_acc:.2f}%")


# Main loop
print("Type a word to check if it's profanity. Type 'exit' to quit.")
while True:
    word = input("Enter word: ").strip().lower()
    if word.lower() == 'exit':
        break
    x = torch.tensor(encode_word(word), dtype=torch.long).unsqueeze(0).to(device) # (1, max_len)
    with torch.no_grad():
        logit, _ = model(x)
        prob = torch.sigmoid(logit).item()
        print(prob)
        print("Profanity" if prob > 0.5 else "Not profanity")

