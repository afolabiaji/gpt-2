from torch import nn
import torch
import torch.nn.functional as F

def causal_mask(seq_length, device):
    # Create an upper triangular matrix filled with -inf
    mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask  # Shape: (seq_length, seq_length)


class GPT(nn.Module):
    def __init__(self, max_seq_length, embedding_dim, vocab_size, num_blocks, num_heads, ff_dim):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.positional_embedding = nn.Embedding(max_seq_length, embedding_dim)
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = nn.ModuleList(
            [TransformerBlock(embedding_dim, num_heads, ff_dim) for _ in range(num_blocks)]
        )
        self.lm_head = nn.Linear(embedding_dim, vocab_size)  # Output projection layer

    def forward(self, input, targets=None):
        # Token and positional embeddings
        tok_emb = self.token_embedding(input)  # Shape: (B, T, D)
        pos_indices = torch.arange(0, input.size(1), device=input.device)  # Shape: (T,)
        pos_emb = self.positional_embedding(pos_indices)  # Shape: (T, D)

        # Combine token and positional embeddings
        emb = tok_emb + pos_emb  # Shape: (B, T, D)

        # Pass through Transformer blocks
        x = emb
        for block in self.transformer:
            x = block(x)

        # Project to vocabulary size
        logits = self.lm_head(x)  # Shape: (B, T, vocab_size)

        # Compute loss if targets are provided
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

        return logits




class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        # Linear layers for Q, K, V
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        # Output projection
        self.output = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x, mask=None):
        B, T, D = x.size()
        H = self.num_heads
        d_k = self.head_dim

        # Linear projections
        Q = self.query(x)  # Shape: (B, T, D)
        K = self.key(x)    # Shape: (B, T, D)
        V = self.value(x)  # Shape: (B, T, D)

        # Split and reshape into multiple heads
        Q = Q.view(B, T, H, d_k).transpose(1, 2)  # Shape: (B, H, T, d_k)
        K = K.view(B, T, H, d_k).transpose(1, 2)  # Shape: (B, H, T, d_k)
        V = V.view(B, T, H, d_k).transpose(1, 2)  # Shape: (B, H, T, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(d_k)  # Shape: (B, H, T, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)  # Shape: (B, H, T, T)
        attention_output = torch.matmul(attention_weights, V)  # Shape: (B, H, T, d_k)

        # Combine heads back
        attention_output = attention_output.transpose(1, 2).contiguous().view(B, T, D)  # Shape: (B, T, D)

        # Final output projection
        return self.output(attention_output)
    
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, ff_dim, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim)
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
    
    def forward(self, input, mask=None):
        seq_length = input.size(1)
        device = input.device
        # Generate causal mask
        mask = causal_mask(seq_length, device)

        attn_output = input + self.norm1(self.attention(input, mask))
        ffn_output = attn_output + self.norm2(self.ffn(attn_output))
        return ffn_output
