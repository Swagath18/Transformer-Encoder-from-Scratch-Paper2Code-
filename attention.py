# attention.py

import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute the scaled dot product attention.

    Args:
        Q: Queries (batch_size, seq_len_q, d_k)
        K: Keys (batch_size, seq_len_k, d_k)
        V: Values (batch_size, seq_len_v, d_v)
        mask: (optional) Masking matrix (broadcastable to scores)

    Returns:
        output: Attention weighted values
        attention_weights: Softmax scores
    """

    d_k = Q.shape[-1]  # key dimensionality

    # Step 1: Compute scores (QK^T)
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)  # (batch_size, seq_len_q, seq_len_k)

    # Step 2: (optional) Masking
    if mask is not None:
        scores = scores + (mask * -1e9)  # mask out irrelevant tokens by setting large negative

    # Step 3: Softmax to get attention weights
    attention_weights = softmax(scores)

    # Step 4: Multiply by V
    output = np.matmul(attention_weights, V)

    return output, attention_weights

def softmax(x):
    """
    Stable softmax function (row-wise).
    """
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


#testing
# if __name__ == "__main__":
#     np.random.seed(0)
#     batch_size = 2
#     seq_len = 4
#     d_k = 8

#     Q = np.random.randn(batch_size, seq_len, d_k)
#     K = np.random.randn(batch_size, seq_len, d_k)
#     V = np.random.randn(batch_size, seq_len, d_k)

#     output, attention_weights = scaled_dot_product_attention(Q, K, V)

#     print("Output shape:", output.shape)              # Should be (batch_size, seq_len, d_k)
#     print("Attention weights shape:", attention_weights.shape)  # Should be (batch_size, seq_len, seq_len)


#Multiheadattenion 
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Initialize weight matrices for Q, K, V and final output projection
        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)
        self.W_o = np.random.randn(d_model, d_model)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, d_k).
        Transpose to shape (batch_size, num_heads, seq_len, d_k)
        """
        x = x.reshape(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def combine_heads(self, x, batch_size):
        """
        Combine heads into original shape.
        (batch_size, seq_len, num_heads, d_k) -> (batch_size, seq_len, d_model)
        """
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, -1, self.d_model)

    def __call__(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]

        # Linear projections
        Q = np.matmul(Q, self.W_q)
        K = np.matmul(K, self.W_k)
        V = np.matmul(V, self.W_v)

        # Split into multiple heads
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Scaled Dot-Product Attention on each head
        scaled_attention, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads
        concat_attention = self.combine_heads(scaled_attention, batch_size)

        # Final linear layer
        output = np.matmul(concat_attention, self.W_o)

        return output


#multihead test
if __name__ == "__main__":
    np.random.seed(0)
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2

    Q = np.random.randn(batch_size, seq_len, d_model)
    K = np.random.randn(batch_size, seq_len, d_model)
    V = np.random.randn(batch_size, seq_len, d_model)

    mha = MultiHeadAttention(d_model, num_heads)
    output = mha(Q, K, V)

    print("Multi-Head Attention Output shape:", output.shape)  # (2, 4, 8)
