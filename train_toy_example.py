# train_toy_example.py

import numpy as np
from transformer_encoder import TransformerEncoder

np.random.seed(42)

# 1. Generate toy dataset
def generate_data(num_samples, seq_len, d_model, threshold=0.0):
    X = np.random.randn(num_samples, seq_len, d_model)
    y = (np.sum(X, axis=(1,2)) > threshold).astype(int)  # sum all elements
    return X, y

# 2. Simple classifier head
class SimpleClassifier:
    def __init__(self, d_model):
        self.W = np.random.randn(d_model) * 0.01
        self.b = 0.0

    def __call__(self, x_encoded):
        """
        Args:
            x_encoded: (batch_size, seq_len, d_model)
        Returns:
            logits: (batch_size,)
        """
        x_pooled = np.mean(x_encoded, axis=1)  # Average pool over sequence length
        logits = np.dot(x_pooled, self.W) + self.b
        return logits

    def parameters(self):
        return [self.W, self.b]

# 3. Binary Cross Entropy Loss
def binary_cross_entropy_loss(logits, targets):
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    loss = -np.mean(targets * np.log(probs + 1e-10) + (1 - targets) * np.log(1 - probs + 1e-10))
    return loss, probs

# 4. Train loop
def train_model(X_train, y_train, encoder, classifier, num_epochs=100, lr=1e-2):
    for epoch in range(num_epochs):
        # Forward
        encoder_output = encoder(X_train)
        logits = classifier(encoder_output)
        loss, probs = binary_cross_entropy_loss(logits, y_train)

        # Backward manually (gradient descent)
        grad_logits = probs - y_train  # derivative of BCE wrt logits
        grad_logits = grad_logits[:, np.newaxis]  # (batch_size, 1)

        x_pooled = np.mean(encoder_output, axis=1)  # (batch_size, d_model)
        
        grad_W = np.mean(grad_logits * x_pooled, axis=0)
        grad_b = np.mean(grad_logits)

        # Update classifier parameters
        classifier.W -= lr * grad_W
        classifier.b -= lr * grad_b

        if epoch % 10 == 0:
            acc = np.mean((probs > 0.5) == y_train)
            print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {acc*100:.2f}%")

# 5. Main
if __name__ == "__main__":
    num_samples = 500
    seq_len = 5
    d_model = 8
    num_heads = 2
    d_ff = 32
    num_layers = 2
    max_seq_len = 10

    X_train, y_train = generate_data(num_samples, seq_len, d_model)

    encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, max_seq_len)
    classifier = SimpleClassifier(d_model)

    train_model(X_train, y_train, encoder, classifier)
