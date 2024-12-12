import torch.nn as nn
import torch 

class SeqMLP(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_classes, dropout = 0):
        """
        Initializes the Sequence-to-Sequence Model.
        
        Args:
            vocab_size (int): Number of unique tokens in the vocabulary.
            embedding_dim (int, optional): Dimension of the embedding vectors. Defaults to 300.
            hidden_size (int, optional): Dimension of the hidden layer. Defaults to 300.
            num_classes (int, optional): Number of output classes. Defaults to 2.
        """
        super(SeqMLP, self).__init__()
        
        # 1) Embedding layer: Converts integer indices to embedding vectors
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=300)
        
        # 2) Linear layer: Maps each embedding vector to a hidden representation
        self.linear = nn.Linear(in_features=300, out_features=hidden_size)
        
        # 3) ReLU activation: Introduces non-linearity
        self.relu = nn.ReLU()
        
        # 5) Final Linear layer: Projects the pooled representation to the number of classes
        self.output_linear = nn.Linear(in_features=hidden_size, out_features=num_classes)
        
    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, time_steps), dtype=torch.long
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes), dtype=torch.float
        """
        # 1) Embedding: (batch, time) -> (batch, time, embedding_dim)
        embeds = self.embedding(x)
        
        # 2) Linear layer: (batch, time, embedding_dim) -> (batch, time, hidden_size)
        linear_out = self.linear(embeds)
        
        # 3) ReLU activation: (batch, time, hidden_size) -> (batch, time, hidden_size)
        relu_out = self.relu(linear_out)
        
        # 4) Global max pool along the time dimension: (batch, time, hidden_size) -> (batch, hidden_size)
        # torch.max returns a tuple (values, indices). We take the first element (values).
        pooled_out, _ = torch.max(relu_out, dim=1)
        
        # 5) Final Linear layer: (batch, hidden_size) -> (batch, num_classes)
        output = self.output_linear(pooled_out)
        
        # 6) Output tensor: (batch, num_classes)
        return output