import torch
import torch.nn as nn
from utils import *

def causal_mask(seq_len, device):
    """
    Creates a lower-triangular causal mask (size [seq_len, seq_len])
    where True indicates "do not attend" (for future tokens).
    GPT-style models use causal masking to enforce autoregressive flow.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.bool()

class GPT3ForRegression(nn.Module):
    def __init__(
        self,
        input_dim=22,
        output_dim=10,
        embed_dim=768,      # Larger default than GPT-2 (e.g., 768 used by GPT-2 small, but GPT-3 uses 12k+ in reality)
        num_heads=12,       # More heads to reflect GPT-3 style scaling
        num_layers=6,       # GPT-3 is typically 96+ layers, but we'll keep it small for illustration
        ff_dim=3072,        # Common GPT-3 ratio ~4x embed_dim
        dropout=0.1,
        max_seq_len=22
    ):
        """
        GPT-3–like Transformer (decoder-only) for a regression problem.
        
        input_dim = 22  -> We treat each input feature as a 'token'.
        output_dim = 10 -> We output 10 continuous values (e.g., joint angles).
        embed_dim, num_heads, num_layers, ff_dim, dropout, max_seq_len -> GPT-3–style hyperparameters.
        """
        super().__init__()
        
        # Model name (optional): helps track hyperparameters
        self.name = (
            f"GPT3ForRegression [seq={input_dim}, emb={embed_dim}, "
            f"heads={num_heads}, layers={num_layers}, ff={ff_dim}, out={output_dim}]"
        )

        # Each of the 22 input features is a scalar to be embedded into an 'embed_dim' vector
        #self.token_embedding = nn.Linear(1, embed_dim)
        self.token_embedding = nn.Linear(input_dim, embed_dim)

        # Positional embeddings (GPT-3 also uses learnable embeddings)
        self.position_embedding = nn.Embedding(input_dim, embed_dim)

        # Transformer blocks
        self.layers = nn.ModuleList([
            GPT3Block(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final linear layer for regression
        self.output_layer = nn.Linear(embed_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

    def forward(self, x):
        """
        Forward pass for GPT-3–style regression.
        :param x: Tensor of shape (batch_size, seq_len=22), each row is a 22-dimensional input.
        :return: Tensor of shape (batch_size, output_dim=10).
        """
        batch_size, seq_len = x.shape
        device = x.device

        # Reshape to [batch_size, seq_len, 1] so we can embed each scalar "token"
        #x = x.unsqueeze(-1)                                     # (batch_size, seq_len, 1)
        #print(x.shape)
        x = self.token_embedding(x)                             # (batch_size, seq_len, embed_dim)
        #print(x.shape)
        # Create position ids [0..seq_len-1]
        #position_ids = torch.arange(seq_len, device=device).unsqueeze(0)  # shape (1, seq_len)
        #pos_emb = self.position_embedding(position_ids)                  # shape (1, seq_len, embed_dim)
        
        # Add position embeddings
        #x = x + pos_emb  # shape (batch_size, seq_len, embed_dim)
        #x = self.dropout(x)
        #print(x.shape)

        # Causal mask: prevents a token from attending to future tokens
        # In many regression tasks you might not need a causal mask, but we include it for GPT-3–style.
        #attn_mask = causal_mask(seq_len, device=device)
        attn_mask = None
        #print(attn_mask.shape)

        # Pass through GPT-3–style decoder blocks
        for layer in self.layers:
            x = layer(x, attn_mask)

        # Aggregate across sequence -> single vector per batch
        # We use mean pooling here; you could also use x[:,0,:] if you want a 'first token' representation
        #x = x.mean(dim=1)  # (batch_size, embed_dim)

        # Final regression output
        out = self.output_layer(x)  # (batch_size, output_dim)
        return out, out

class GPT3Block(nn.Module):
    """
    A single GPT-3–like Transformer decoder block.
    
    Key difference from GPT-2 style: pre-layer normalization
    or a slight tweak in how the feedforward and attention layers are ordered.
    GPT-3 is massive in scale, but the fundamental structure is quite similar.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            #nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """
        :param x: (batch_size, seq_len, embed_dim)
        :param attn_mask: (seq_len, seq_len) True = do not attend
        :return: (batch_size, seq_len, embed_dim)
        """
        # GPT-3 typically uses 'pre-ln': LayerNorm before attention
        # 1) Pre-layer norm
        normed_x = self.ln1(x)
        # 2) Self-attention
        attn_output, _ = self.attn(normed_x, normed_x, normed_x, attn_mask=attn_mask)
        # 3) Residual connection
        #x = x + self.dropout(attn_output)
        x = x + attn_output
        
        # 4) Another layer norm
        normed_x = self.ln2(x)
        # 5) Feedforward
        ff_out = self.mlp(normed_x)
        # 6) Residual connection
        #x = x + self.dropout(ff_out)
        x = x + ff_out

        return x



if __name__ == "__main__":

    print("\n\n")
    print("Testing Transformer Architecture")

    # Define model
    #model = GPT3ForRegression()
    model = GPT3ForRegression(input_dim=19, output_dim=7, embed_dim=192, num_heads=12, num_layers=3, ff_dim=768)

    print("==> Trainable parameters: {}".format(count_parameters(model)))

    # Create a dummy input: (batch_size=16, input_dim=22)
    inputs = torch.rand(16, 22)

    # Forward pass
    outputs, _ = model(inputs)  # Shape: (16, 10)

    print("Model input shape:", inputs.shape)
    print("Model output shape:", outputs.shape)
