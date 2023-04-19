# Author: Jaward Sesay - Jaykef (苏杰)
# 2023-04-15

# Import pytorch libraries
import torch
import torch.nn as nn
import torch.optim as optim

# Define the GPT model
class dummyGPT(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.transformer = nn.Transformer(hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        out = self.transformer(src, tgt)
        out = self.fc(out)
        return out

# Define the dataset
data = ["The quick brown fox jumps over the lazy dog.",
        "The quick brown fox jumps over the lazy dog again.",
        "The quick brown fox jumps over the lazy dog one more time.",
        "The quick brown fox jumps over the lazy dog once more.",]

# Convert text to numerical data
word_to_index = {}
index_to_word = {}
for sentence in data:
    for word in sentence.split():
        if word not in word_to_index:
            index = len(word_to_index)
            word_to_index[word] = index
            index_to_word[index] = word

X = torch.tensor([word_to_index[word] for sentence in data for word in sentence.split()[:-1]])
Y = torch.tensor([word_to_index[word] for sentence in data for word in sentence.split()[1:]])

# Define model parameters
vocab_size = len(word_to_index)
embedding_size = 128
hidden_size = 128
num_layers = 1

# Define the model, loss function, and optimizer
model = dummyGPT(vocab_size, embedding_size, hidden_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X, Y)
    loss = criterion(outputs.view(-1, vocab_size), Y)
    loss.backward()
    optimizer.step()
    print("Epoch {}: loss={}".format(epoch, loss.item()))

# Generate text given a prompt
prompt = "The quick brown"
prompt_tensor = torch.tensor([word_to_index[word] for word in prompt.split()])
output = model(prompt_tensor.unsqueeze(0), prompt_tensor.unsqueeze(0))
next_word_index = torch.argmax(output[-1]).item()
next_word = index_to_word.get(next_word_index, "<unk>")
print(prompt + " " + next_word)
