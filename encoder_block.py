# encoder_block.py

import numpy as np
from attention import MultiHeadAttention
from feedforward import FeedForwardNetwork

class EncoderBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)

    def layer_norm(self, x, eps=1e-6):
        """
        Layer normalization over last dimension.
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return (x - mean) / (std + eps)

    def __call__(self, x, mask=None):
        """
        Forward pass for encoder block.
        Args:
            x: (batch_size, seq_len, d_model)
            mask: optional mask
        Returns:
            Output tensor of same shape
        """
        # Multi-Head Attention sublayer
        attn_output = self.mha(x, x, x, mask)  # Self-attention
        x = self.layer_norm(x + attn_output)   # Add & Norm

        # Feed Forward sublayer
        ffn_output = self.ffn(x)
        x = self.layer_norm(x + ffn_output)     # Add & Norm

        return x


#testing encoder
if __name__ == "__main__":
    np.random.seed(0)
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2
    d_ff = 32

    from positional_encoding import positional_encoding

    # Create dummy input
    x = np.random.randn(batch_size, seq_len, d_model)

    # Add positional encoding
    pos_enc = positional_encoding(seq_len, d_model)
    x = x + pos_enc

    encoder_block = EncoderBlock(d_model, num_heads, d_ff)
    output = encoder_block(x)

    print("Encoder Block Output shape:", output.shape)  # Should be (2, 4, 8)
