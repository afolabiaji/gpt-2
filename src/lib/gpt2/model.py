from torch import nn
import torch
import torch.nn.functional as F
from dataclasses import dataclass

def causal_mask(seq_length, device):
    # Create an upper triangular matrix filled with -inf
    mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask  # Shape: (seq_length, seq_length)

@dataclass
class GPTConfig():
    max_seq_length:int = 1024
    embedding_dim:int =768
    vocab_size: int = 50257
    num_blocks: int = 12
    num_heads: int = 12


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.max_seq_length = config.max_seq_length
        self.embedding_dim = config.embedding_dim
        self.transformer = nn.ModuleDict(
            dict(
                positional_embedding = nn.Embedding(config.max_seq_length, config.embedding_dim),
                token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim),
                transformer_blocks = nn.ModuleList(
                    [TransformerBlock(config) for _ in range(config.num_blocks)]
                ),
                layer_norm = nn.LayerNorm(config.embedding_dim)
            )
        )
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)  # Output projection layer

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.max_seq_length
        # Token and positional embeddings
        tok_emb = self.transformer.token_embedding(idx)  # Shape: (B, T, D)
        pos_indices = torch.arange(0, T, dtype=torch.long, device=idx.device)  # Shape: (T,)
        pos_emb = self.transformer.positional_embedding(pos_indices)  # Shape: (T, D)

        # Combine token and positional embeddings
        x = tok_emb + pos_emb  # Shape: (B, T, D)

        # Pass through Transformer blocks
        for block in self.transformer.transformer_blocks:
            x = block(x)

        x = self.transformer.layer_norm(x)
        # Project to vocabulary size
        logits = self.lm_head(x)  # Shape: (B, T, vocab_size)

        # Compute loss if targets are provided
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

        return logits




class MultiHeadAttention(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.embedding_dim = config.embedding_dim
        self.max_seq_length = config.max_seq_length
        assert self.embedding_dim % self.num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.attnetion_matrix = nn.Linear(self.embedding_dim, 3 * self.embedding_dim)
        # Output projection
        self.output = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.register_buffer("bias", torch.tril(
            torch.ones(
                self.max_seq_length, self.max_seq_length
            )
            .view(
                1, 1, self.max_seq_length, self.max_seq_length
            )
        ))

    def forward(self, x, mask=None):
        B, T, D = x.size()
        H = self.num_heads
        d_k = self.head_dim

        QKV = self.attnetion_matrix(x)
        Q, K, V = QKV.split(self.embedding_dim, dim=-1)# Shape: (B, T, D)

        # Split and reshape into multiple heads
        Q = Q.view(B, T, H, d_k).transpose(1, 2)  # Shape: (B, H, T, d_k)
        K = K.view(B, T, H, d_k).transpose(1, 2)  # Shape: (B, H, T, d_k)
        V = V.view(B, T, H, d_k).transpose(1, 2)  # Shape: (B, H, T, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float))  # Shape: (B, H, T, T)
        scores = scores.masked_fill(self.bias[:, :, T, T] == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)  # Shape: (B, H, T, T)
        attention_output = torch.matmul(attention_weights, V)  # Shape: (B, H, T, d_k)

        # Combine heads back
        attention_output = attention_output.transpose(1, 2).contiguous().view(B, T, D)  # Shape: (B, T, D)

        # Final output projection
        return self.output(attention_output)
    
class TransformerBlock(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.attention = MultiHeadAttention(config)
        self.norm1 = nn.LayerNorm(self.embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.embedding_dim, 4 * self.embedding_dim),
            nn.GELU(),
            nn.Linear(4 * self.embedding_dim, self.embedding_dim)
        )
        self.norm2 = nn.LayerNorm(self.embedding_dim)
    
    def forward(self, input):
        seq_length = input.size(1)
        device = input.device
        # Generate causal mask
        mask = causal_mask(seq_length, device)

        attn_output = input + self.attention(self.norm1(input))
        ffn_output = attn_output + self.ffn(self.norm2(attn_output))
        return ffn_output
