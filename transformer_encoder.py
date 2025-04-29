# transformer_encoder.py

import numpy as np
from encoder_block import EncoderBlock
from positional_encoding import positional_encoding

class TransformerEncoder:
    def __init__(self, num_layers, d_model, num_heads, d_ff, max_seq_len):
        """
        Args:
            num_layers: number of encoder blocks to stack
            d_model: input/output dimension
            num_heads: number of attention heads
            d_ff: feedforward hidden dimension
            max_seq_len: maximum length of input sequences
        """
        self.num_layers = num_layers
        self.d_model = d_model
        self.pos_encoding = positional_encoding(max_seq_len, d_model)  # (max_seq_len, d_model)

        self.encoder_blocks = [EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]

    def __call__(self, x, mask=None):
        """
        Forward pass through Transformer Encoder.
        Args:
            x: (batch_size, seq_len, d_model)
            mask: optional mask
        Returns:
            Encoded output
        """
        batch_size, seq_len, _ = x.shape

        # Add positional encoding
        x += self.pos_encoding[:seq_len]

        # Pass through stacked encoder blocks
        for block in self.encoder_blocks:
            x = block(x, mask)

        return x


#testing trnasformer encoder
if __name__ == "__main__":
    np.random.seed(0)
    batch_size = 2
    seq_len = 5
    d_model = 8
    num_heads = 2
    d_ff = 32
    num_layers = 2
    max_seq_len = 10

    x = np.random.randn(batch_size, seq_len, d_model)

    transformer_encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, max_seq_len)
    output = transformer_encoder(x)

    print("Transformer Encoder Output shape:", output.shape)  # Should be (2, 5, 8)
