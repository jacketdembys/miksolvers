import torch
import torch.nn as nn
from utils import *


class GPT2ForRegression(nn.Module):
    def __init__(self, input_dim=22, 
                 output_dim=10, 
                 embed_dim=128, 
                 num_heads=4, 
                 num_layers=3, 
                 ff_dim=256):
        super(GPT2ForRegression, self).__init__()
        
        # Name the model
        self.name = "Transformer [in{}, emb{}, nhead{}, nlayer{}, ffdim{}, out{}]".format(str(input_dim), str(embed_dim), str(num_heads), str(num_layers), str(ff_dim), str(output_dim))
         
        # Embedding for input features
        #self.input_embedding = nn.Linear(1, embed_dim)
        self.input_embedding = nn.Linear(input_dim, embed_dim)
        
        # Positional embeddings
        self.position_embedding = nn.Embedding(input_dim, embed_dim)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        
        # Final output layer for regression
        self.output_layer = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        """
        Forward pass for the model.
        :param x: Input tensor of shape (batch_size, input_dim)
        :return: Output tensor of shape (batch_size, output_dim)
        """
        batch_size, seq_len = x.size()  # seq_len = input_dim
        
        # Project inputs to embedding space
        #print(x.shape)
        #x = x.unsqueeze(-1)
        #print(x.shape)
        x = self.input_embedding(x)  # Shape: (batch_size, input_dim, embed_dim)
        #print(x.shape)

        # Add positional embeddings
        #position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)  # Shape: (1, input_dim)
        #position_embeddings = self.position_embedding(position_ids)  # Shape: (1, input_dim, embed_dim)
              
        
        #print(x.shape)
        #print(position_embeddings.shape)
        #x = x + position_embeddings  # Shape: (batch_size, input_dim, embed_dim)
        

        # Pass through Transformer blocks
        for layer in self.layers:
            x = layer(x)  # Shape: (batch_size, input_dim, embed_dim)
        
        # Take the first token's representation for regression (e.g., x[:, 0, :])
        #x = x.mean(dim=1)  # Aggregate embeddings across features
                
        # Final regression output
        ##print(x.shape)
        output = self.output_layer(x)  # Shape: (batch_size, output_dim)
        return output, output


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        
        # Multi-Head Self-Attention
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Feedforward Neural Network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            #nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        
        # Layer normalization and dropout
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Forward pass for a single Transformer block.
        :param x: Input tensor of shape (batch_size, seq_len, embed_dim)
        :return: Output tensor of the same shape
        """
        # Self-attention with residual connection
        attention_output, _ = self.attention(x, x, x)  # (batch_size, seq_len, embed_dim)
        x = x + self.dropout(attention_output)
        #x = x + attention_output
        x = self.layer_norm_1(x)
        
        # Feedforward with residual connection
        ff_output = self.feed_forward(x)  # (batch_size, seq_len, embed_dim)
        x = x + self.dropout(ff_output)
        #x = x + ff_output
        x = self.layer_norm_2(x)
        
        return x


if __name__ == "__main__":

    print("\n\n")
    print("Testing Transformer Architecture")

    # Define model
    model = GPT2ForRegression(input_dim=19, output_dim=7, embed_dim=192, num_heads=12, num_layers=4, ff_dim=1024)

    print("==> Trainable parameters: {}".format(count_parameters(model)))

    # Create a dummy input: (batch_size=16, input_dim=22)
    inputs = torch.rand(16, 18)

    # Forward pass
    outputs, _ = model(inputs)  # Shape: (16, 10)

    print("Model input shape:", inputs.shape)
    print("Model output shape:", outputs.shape)


    