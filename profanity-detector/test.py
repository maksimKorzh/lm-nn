import torch
from model import *

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM(vocab_size, n_embd, n_hidden, n_layer).to(device)
model.load_state_dict(torch.load('profanity_model.pt', map_location=device))
model.eval()
m = model.to(device)
print(sum(p.numel() for p in m.parameters()), 'parameters')

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
        print("Profanity" if prob > 0.5 else "Not profanity")
