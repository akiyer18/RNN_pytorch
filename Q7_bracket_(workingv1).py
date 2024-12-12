import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import torch.distributions as dist
import numpy as np
import random
from data_rnn import *

# ------------- Data Preparation -------------
# Load data
x_train, (i2w, w2i) = load_brackets(n=150_000)

# Define mappings
i2w = ['.pad', '.start', '.end', '.unk', '(', ')']
w2i = {'.pad': 0, '.start': 1, '.end': 2, '.unk': 3, '(': 4, ')': 5}

# Hyperparameters
embedding_dim = 32  # e=32
hidden_size = 16  # h=16
num_layers = 1  # Single layer
batch_size = 64
epochs = 3
learning_rate = 0.001
max_length = 50  # Adjust based on your data
vocab_size = len(i2w)  # 6

# Convert sequences to PyTorch tensors
x_train_tensors = [torch.tensor(seq, dtype=torch.long) for seq in x_train]

# Pad sequences
x_train_padded = pad_sequence(
    x_train_tensors,
    batch_first=True,
    padding_value=w2i['.pad'])

# Truncate or pad to `max_length`
if x_train_padded.size(1) > max_length:
    x_train_padded = x_train_padded[:, :max_length]
else:
    padding = (0, max_length - x_train_padded.size(1))
    x_train_padded = torch.nn.functional.pad(x_train_padded, padding, value=w2i['.pad'])

# Creating Targets for Next-Token Prediction
x_input = x_train_padded[:, :-1]  # Shape: (batch, time)
y_target = x_train_padded[:, 1:]  # Shape: (batch, time)

# Create TensorDataset
dataset = TensorDataset(x_input, y_target)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ------------ Sampling Function ------------

def sample(lnprobs, temperature=0.3):
    """
    Sample an element from a categorical distribution.

    Args:
        lnprobs (torch.Tensor): Outcome logits (1D tensor).
        temperature (float): Sampling temperature.
                             1.0 follows the given distribution,
                             0.0 returns the maximum probability element.

    Returns:
        int: The index of the sampled element.
    """
    if temperature == 0.0:
        return lnprobs.argmax().item()
    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)
    return cd.sample().item()


# ------------ Model Definition ------------

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

        Returns:
            torch.FloatTensor: Output tensor of shape (batch, time, vocab_size)
        """
        embedded = self.embedding(x)  # Shape: (batch, time, embedding_dim)
        lstm_out, _ = self.lstm(embedded)  # Shape: (batch, time, hidden_size)
        logits = self.fc(lstm_out)  # Shape: (batch, time, vocab_size)
        return logits


# ------------ Device Configuration ------------

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

# ------------ Loss Function and Optimizer ------------

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=w2i['.pad'])  # Ignore padding in loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# ------------ Sequence Generation Function ------------

def generate_sequence(model, seed_seq, w2i, i2w, device, max_length=50, temperature=0.5):
    """
    Generate a sequence of tokens from the model based on a seed sequence.

    Args:
        model (nn.Module): Trained PyTorch model.
        seed_seq (list): List of integer indices representing the seed sequence.
        w2i (dict): Word to index mapping.
        i2w (list): Index to word mapping.
        device (torch.device): Device to run the model on.
        max_length (int): Maximum length of the generated sequence.
        temperature (float): Sampling temperature.

    Returns:
        list: Generated sequence of tokens.
    """
    model.eval()  # Set model to evaluation mode

    # Copy the seed sequence to avoid modifying the original
    generated_seq = seed_seq.copy()

    # Move the seed sequence to a tensor and add batch dimension
    input_tensor = torch.tensor([generated_seq], dtype=torch.long).to(device)  # Shape: (1, time)

    with torch.no_grad():
        for _ in range(max_length - len(seed_seq)):
            # Get model outputs
            outputs = model(input_tensor)  # Shape: (1, time, vocab_size)

            # Get the logits for the last time step
            last_logits = outputs[0, -1, :]  # Shape: (vocab_size,)

            # Sample the next token using the provided sample function
            next_token = sample(last_logits, temperature)

            # Append the sampled token to the sequence
            generated_seq.append(next_token)

            # If the sampled token is '.end', stop generation
            if next_token == w2i['.end']:
                break

            # Update the input tensor for the next prediction
            input_tensor = torch.tensor([generated_seq], dtype=torch.long).to(device)  # Shape: (1, time)

    # Convert indices to tokens
    generated_tokens = [i2w[idx] for idx in generated_seq]
    return generated_tokens


# ------------ Training Loop ------------

model.train()  # Set the model to training mode

# Define a seed sequence for sampling
seed_seq = [w2i['.start'], w2i['('], w2i['('], w2i[')']]

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
    print(f'Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}')

    # ------------ Generating Samples After Each Epoch ------------

    print(f'\n--- Generating Samples After Epoch {epoch} ---')
    for i in range(1, 11):
        generated_tokens = generate_sequence(
            model=model,
            seed_seq=seed_seq,
            w2i=w2i,
            i2w=i2w,
            device=device,
            max_length=50,
            temperature=0.5  # Low temperature for controlled sampling
        )
        print(f'Sample {i}:', ' '.join(generated_tokens))
    print('-' * 50 + '\n')


# ------------ Evaluation ------------

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


# ------------ Making Predictions ------------

# Function to convert indices to tokens
def indices_to_tokens(indices, i2w):
    return [i2w[idx] for idx in indices]


# Example prediction
# Define the seed sequence
seed_seq = [w2i['.start'], w2i['('], w2i['('], w2i[')']]

# Generate a single sequence
generated_tokens = generate_sequence(
    model=model,
    seed_seq=seed_seq,
    w2i=w2i,
    i2w=i2w,
    device=device,
    max_length=50,
    temperature=0.5  # Low temperature for controlled sampling
)

# Print the generated sequence
print("Generated Sequence:", ' '.join(generated_tokens))
