# feedforward.py

import numpy as np

class FeedForwardNetwork:
    def __init__(self, d_model, d_ff):
        """
        Two fully connected layers with ReLU in between.
        d_model: input and output dimension
        d_ff: hidden dimension (larger)
        """
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2. / d_model)
        self.b1 = np.zeros((d_ff,))
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2. / d_ff)
        self.b2 = np.zeros((d_model,))

    def __call__(self, x):
        """
        Forward pass.
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Linear layer 1 + ReLU
        x = np.matmul(x, self.W1) + self.b1
        x = np.maximum(0, x)  # ReLU activation

        # Linear layer 2
        x = np.matmul(x, self.W2) + self.b2
        return x

#testing feedforward
if __name__ == "__main__":
    np.random.seed(0)
    batch_size = 2
    seq_len = 4
    d_model = 8
    d_ff = 32

    x = np.random.randn(batch_size, seq_len, d_model)
    ffn = FeedForwardNetwork(d_model, d_ff)
    output = ffn(x)

    print("FeedForward Output shape:", output.shape)  # Should be (2, 4, 8)
