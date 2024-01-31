import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Define your GPT-like model class
class GPTModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, block_size, dropout=0.1):
        super(GPTModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=4,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
            dropout=dropout
        )
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        output = self.transformer(embedded)
        logits = self.fc(output)
        return logits

# Define a function to train the model
def train_model(model, dataloader, num_epochs, criterion, optimizer, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")
    
    print("Training finished!")

# Define a function to generate text from the trained model
def generate_text(model, start_text, max_length, temperature=1.0):
    model.eval()
    with torch.no_grad():
        input_ids = [char_to_idx[char] for char in start_text]
        for _ in range(max_length):
            input_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(device)
            output_logits = model(input_tensor)
            output_logits = output_logits[:, -1, :] / temperature
            predicted_token = torch.multinomial(torch.softmax(output_logits, dim=-1), 1)
            predicted_char = idx_to_char[predicted_token.item()]
            input_ids.append(predicted_token.item())
            if predicted_char == '\n':
                break
    return ''.join([idx_to_char[idx] for idx in input_ids])

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Define your dataset and dataloader here
# Make sure to have a proper dataset and dataloader setup

# Define hyperparameters
vocab_size = len(chars)
embedding_dim = 256
hidden_dim = 512
num_layers = 4
block_size = 128
dropout = 0.1
num_epochs = 50
learning_rate = 0.001
temperature = 0.7
max_length = 200

# Create and initialize the GPT-like model
model = GPTModel(vocab_size, embedding_dim, hidden_dim, num_layers, block_size, dropout)
model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Train the model
train_model(model, dataloader, num_epochs, criterion, optimizer, device)

# Generate text using the trained model
start_text = "Once upon a time"
generated_text = generate_text(model, start_text, max_length, temperature=temperature)
print("\nGenerated Text:")
print(generated_text)
