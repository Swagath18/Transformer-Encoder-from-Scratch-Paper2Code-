# positional_encoding.py

import numpy as np

def positional_encoding(max_seq_len, d_model):
    """
    Compute sinusoidal positional encodings.

    Args:
        max_seq_len: maximum length of sequence
        d_model: embedding dimension

    Returns:
        pos_encoding: (max_seq_len, d_model) positional encoding matrix
    """
    PE = np.zeros((max_seq_len, d_model))
    position = np.arange(0, max_seq_len)[:, np.newaxis]  # (max_seq_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))  # (d_model/2,)

    PE[:, 0::2] = np.sin(position * div_term)  # even indices
    PE[:, 1::2] = np.cos(position * div_term)  # odd indices

    return PE

#testing PE
if __name__ == "__main__":
    max_seq_len = 10
    d_model = 16

    pe = positional_encoding(max_seq_len, d_model)
    print("Positional Encoding shape:", pe.shape)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,8))
    plt.plot(np.arange(max_seq_len), pe[:, 0], label='dim 0 (sin)')
    plt.plot(np.arange(max_seq_len), pe[:, 1], label='dim 1 (cos)')
    plt.legend()
    plt.title("Positional Encoding Patterns (dimensions 0 and 1)")
    plt.show()
