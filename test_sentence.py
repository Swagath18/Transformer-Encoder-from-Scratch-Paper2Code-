# test_sentence.py

import numpy as np
from transformer_encoder import TransformerEncoder
from glove_loader import load_glove_embeddings
from train_toy_example import SimpleClassifier  # reuse classifier class

def sentence_to_embedding(sentence, embeddings_dict, embedding_dim, max_seq_len):
    words = sentence.lower().split()
    embeddings = []
    for word in words:
        if word in embeddings_dict:
            embeddings.append(embeddings_dict[word])
        else:
            embeddings.append(np.zeros(embedding_dim))  # unknown word = zero vector

    # Pad or trim to max_seq_len
    if len(embeddings) < max_seq_len:
        pad_len = max_seq_len - len(embeddings)
        embeddings += [np.zeros(embedding_dim)] * pad_len
    else:
        embeddings = embeddings[:max_seq_len]

    return np.array(embeddings)

if __name__ == "__main__":
    # 1. Load GloVe
    embeddings_dict, embedding_dim = load_glove_embeddings("glove.6B.50d.txt")

    # 2. Load model (fresh random Transformer + Classifier for now)
    d_model = embedding_dim  # 50
    num_heads = 2
    d_ff = 128
    num_layers = 2
    max_seq_len = 10

    encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, max_seq_len)
    classifier = SimpleClassifier(d_model)

    # 3. Prepare input sentence
    sentence = "The movie was fantastic and exciting"
    sentence_embedding = sentence_to_embedding(sentence, embeddings_dict, embedding_dim, max_seq_len)
    sentence_embedding = np.expand_dims(sentence_embedding, axis=0)  # (1, seq_len, d_model)

    # 4. Forward pass
    encoded = encoder(sentence_embedding)
    logits = classifier(encoded)
    prob = 1 / (1 + np.exp(-logits))

    print(f"Sentence: {sentence}")
    print(f"Predicted probability of class 1: {prob[0]:.4f}")
