import torch
import torch.nn as nn
import math


class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()

        # lookup table — shape: (vocab_size, embed_dim)
        # each row = one token's vector
        self.embed = nn.Embedding(vocab_size, embed_dim)

        self.embed_dim = embed_dim

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        # output shape: (batch_size, seq_len, embed_dim)
        return self.embed(x) * math.sqrt(self.embed_dim)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=7, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        # create a matrix of shape (max_seq_len, embed_dim)
        # each row = position encoding for that position
        pe = torch.zeros(max_seq_len, embed_dim)

        # position indices → [0, 1, 2, 3, 4, 5, 6]
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()

        # division term — creates wave patterns of different frequencies
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * (-math.log(10000.0) / embed_dim)
        )

        # fill even indices with sine wave
        pe[:, 0::2] = torch.sin(position * div_term)

        # fill odd indices with cosine wave
        pe[:, 1::2] = torch.cos(position * div_term)

        # add batch dimension → shape: (1, max_seq_len, embed_dim)
        pe = pe.unsqueeze(0)

        # register as buffer — saves with model but not trained
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        # add position encoding to embedding
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        # embed_dim must be divisible by num_heads
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # each head works on a smaller slice of the embedding
        # 128 embed_dim / 8 heads = 16 dimensions per head
        self.head_dim = embed_dim // num_heads

        # Q, K, V projections — one linear layer each
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # final projection after combining all heads
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def split_heads(self, x, batch_size):
        # x shape: (batch_size, seq_len, embed_dim)
        # reshape to: (batch_size, num_heads, seq_len, head_dim)
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        # Step 1 — create Q, K, V from input
        Q = self.W_q(x)  # (batch_size, seq_len, embed_dim)
        K = self.W_k(x)  # (batch_size, seq_len, embed_dim)
        V = self.W_v(x)  # (batch_size, seq_len, embed_dim)

        # Step 2 — split into multiple heads
        Q = self.split_heads(
            Q, batch_size
        )  # (batch, heads, seq_len, head_dim)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Step 3 — calculate attention scores
        # Q × K^T tells us how much each token attends to each other
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # scale down — prevents scores from getting too large
        scores = scores / math.sqrt(self.head_dim)

        # Step 4 — apply mask if provided (optional)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Step 5 — softmax → converts scores to probabilities (0 to 1)
        # each row sums to 1 — these are the attention weights
        attention_weights = torch.softmax(scores, dim=-1)

        # Step 6 — multiply weights by Values
        # high attention weight → that token contributes more
        out = torch.matmul(attention_weights, V)

        # Step 7 — combine all heads back together
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, -1, self.embed_dim)

        # Step 8 — final projection
        out = self.W_o(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        # attention layer
        self.attention = MultiHeadAttention(embed_dim, num_heads)

        # feed forward layers
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),  # expand: 128 → 256
            nn.ReLU(),  # activation
            nn.Linear(ff_dim, embed_dim),  # compress: 256 → 128
        )

        # layer normalisation — one for attention, one for ff
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # ── Attention + Residual connection ──
        attended = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attended))

        # ── Feed Forward + Residual connection ──
        fed = self.ff(x)
        x = self.norm2(x + self.dropout(fed))

        return x


class PatternMind(nn.Module):
    def __init__(
        self,
        vocab_size=1003,
        embed_dim=128,
        num_heads=8,
        num_blocks=4,
        ff_dim=256,
        max_seq_len=7,
        dropout=0.1,
    ):
        super().__init__()

        # Step 1 — embedding layer
        self.embedding = Embedding(vocab_size, embed_dim)

        # Step 2 — positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout)

        # Step 3 — stack of transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
                for _ in range(num_blocks)
            ]
        )

        # Step 4 — output layer
        # predicts one number from vocab_size possible numbers
        self.output_layer = nn.Linear(embed_dim, vocab_size)

        # Step 5 — dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len)

        # Step 1 — embed tokens
        x = self.embedding(x)
        # x shape: (batch_size, seq_len, embed_dim)

        # Step 2 — add position information
        x = self.pos_encoding(x)

        # Step 3 — pass through all transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Step 4 — take only the LAST token's output
        # the last token has attended to all previous tokens
        # so it has the most complete understanding of the sequence
        x = x[:, -1, :]
        # x shape: (batch_size, embed_dim)

        # Step 5 — project to vocab size
        x = self.output_layer(x)
        # x shape: (batch_size, vocab_size)

        return x


if __name__ == "__main__":
    # create model
    model = PatternMind()

    # count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print("PatternMind ready")
    print(f"Total parameters: {total_params:,}")

    # test forward pass with fake input
    fake_input = torch.tensor(
        [[1, 5, 7, 11, 19, 2]]
    )  # [SOS, 2, 4, 8, 16, EOS]
    output = model(fake_input)

    print(f"Input shape  : {fake_input.shape}")
    print(f"Output shape : {output.shape}")
    print(
        f"Output is a probability distribution over {output.shape[1]} tokens"
    )
