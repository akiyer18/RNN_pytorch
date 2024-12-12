import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
import random
import re
from data_rnn import *

# ------------- Data Preparation -------------

# Load the NDFA dataset
x_train, (i2w, w2i) = load_ndfa(n=150_000, seed=0)

# Define Hyperparameters
embedding_dim = 32    # e=32
hidden_size = 16      # h=16
num_layers = 2        # Single layer
batch_size = 64
epochs = 10
learning_rate = 0.001
max_length = 50       # Adjust based on your data
vocab_size = len(i2w) #15 for NDFA

# Convert sequences to PyTorch tensors
x_train_tensors = [torch.tensor(seq, dtype=torch.long) for seq in x_train]

# Define maximum sequence length (based on your data or set a reasonable limit)
max_length = 50  # Adjust as needed

# Pad sequences with the `.pad` token (index 0) at the end
x_train_padded = pad_sequence(
    x_train_tensors,
    batch_first=True,
    padding_value=w2i['.pad']
)

# Truncate or pad to `max_length`
if x_train_padded.size(1) > max_length:
    x_train_padded = x_train_padded[:, :max_length]
else:
    padding = (0, max_length - x_train_padded.size(1))
    x_train_padded = torch.nn.functional.pad(x_train_padded, padding, value=w2i['.pad'])

# Creating Targets for Next-Token Prediction
# Input: all tokens except the last
# Target: all tokens except the first
x_input = x_train_padded[:, :-1]  # Shape: (batch, time)
y_target = x_train_padded[:, 1:]  # Shape: (batch, time)

# Create TensorDataset
dataset = TensorDataset(x_input, y_target)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ------------- Model Definition -------------

class SimpleLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super(SimpleLSTMModel, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=w2i['.pad']
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.LongTensor): Input tensor of shape (batch, time)

        Returns:`
            torch.FloatTensor: Output tensor of shape (batch, time, vocab_size)
        """
        embedded = self.embedding(x)  # Shape: (batch, time, embedding_dim)
        lstm_out, _ = self.lstm(embedded)  # Shape: (batch, time, hidden_size)
        logits = self.fc(lstm_out)  # Shape: (batch, time, vocab_size)
        return logits

# ------------- Device Configuration -------------

# Check for MPS (Apple GPU) support
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using Apple GPU (MPS) for training.")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA GPU for training.")
else:
    device = torch.device('cpu')
    print("Using CPU for training.")

# Initialize the model
model = SimpleLSTMModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_size=hidden_size,
    num_layers=num_layers
)

# Move the model to the selected device
model.to(device)

# ------------- Loss Function and Optimizer -------------

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=w2i['.pad'])  # Ignore padding in loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ------------- Sampling Function -------------

def generate_sequence(model, seed_seq, w2i, i2w, device, max_length=50):
    """
    Generate a sequence by sampling from the model starting with a seed sequence.

    Args:
        model (nn.Module): The trained PyTorch model.
        seed_seq (list of int): The seed sequence as a list of integer indices.
        w2i (dict): Word-to-index mapping.
        i2w (list): Index-to-word mapping.
        device (torch.device): The device to run the model on.
        max_length (int): Maximum length of the generated sequence.

    Returns:
        list of str: The generated sequence as a list of tokens.
    """
    model.eval()
    generated_seq = seed_seq.copy()
    input_tensor = torch.tensor([generated_seq], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_length - len(seed_seq)):
            outputs = model(input_tensor)
            last_logits = outputs[0, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated_seq.append(next_token)
            if next_token == w2i['s']:
                break
            input_tensor = torch.tensor([generated_seq], dtype=torch.long).to(device)

    generated_tokens = [i2w[idx] if idx < len(i2w) else '.unk' for idx in generated_seq]
    return generated_tokens

# ------------- Training Loop -------------

# Define a seed sequence
seed_sequence = [w2i['s']]  # Start with 's'

# Number of samples to generate per epoch
num_samples = 10

# Training Loop
model.train()  # Ensure the model is in training mode

losses = []  # To track training loss

for epoch in range(1, epochs + 1):
    epoch_loss = 0
    for batch_x, batch_y in dataloader:
        # Move data to the appropriate device
        batch_x = batch_x.to(device)  # Shape: (batch, time)
        batch_y = batch_y.to(device)  # Shape: (batch, time)

        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(batch_x)  # Shape: (batch, time, vocab_size)

        # Reshape outputs and targets for loss computation
        outputs = outputs.view(-1, vocab_size)  # Shape: (batch * time, vocab_size)
        batch_y = batch_y.view(-1)  # Shape: (batch * time)

        # Compute loss
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        loss.backward()

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        # Accumulate loss
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)
    print(f'Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}')

    # Generate and print samples after each epoch, similar to the Q7 to check the model performance
    print(f'\n--- Samples after Epoch {epoch} ---')
    for i in range(num_samples):
        generated = generate_sequence(
            model=model,
            seed_seq=seed_sequence,
            w2i=w2i,
            i2w=i2w,
            device=device,
            max_length=max_length
        )
        # Join tokens to form a string
        generated_str = ''.join(generated)
        print(f'Sample {i + 1}: {generated_str}')
    print('-----------------------------------\n')
    # If you don't want to generate samples after per epoch, comment the part above

# ------------- Evaluation -------------

def evaluate(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            outputs = outputs.view(-1, vocab_size)
            batch_y = batch_y.view(-1)

            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Evaluate on the training data (or replace with validation DataLoader)
val_loss = evaluate(model, dataloader, criterion, device)
print(f'Validation Loss: {val_loss:.4f}')

# ------------- Making Final Predictions -------------

# Function to convert indices to tokens
def indices_to_tokens(indices, i2w):
    return [i2w[idx] if idx < len(i2w) else '.unk' for idx in indices]

# Example prediction
sample_sequence = [w2i['s']]  # Starting with 's'
generated_tokens = generate_sequence(
    model=model,
    seed_seq=sample_sequence,
    w2i=w2i,
    i2w=i2w,
    device=device,
    max_length=max_length
)
# Join tokens to form a string
generated_str = ''.join(generated_tokens)
print("\nFinal Sample:", generated_str)