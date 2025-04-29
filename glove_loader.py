# glove_loader.py

def load_glove_embeddings(glove_file_path):
    """
    Load GloVe word embeddings into a dictionary.

    Args:
        glove_file_path: path to GloVe txt file.

    Returns:
        embeddings_dict: {word: vector}
        embedding_dim: dimension of embeddings
    """
    embeddings_dict = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = list(map(float, values[1:]))
            embeddings_dict[word] = vector
    embedding_dim = len(vector)
    return embeddings_dict, embedding_dim


#testing glove
if __name__ == "__main__":
    embeddings_dict, embedding_dim = load_glove_embeddings("glove.6B.50d.txt")
    print(f"Loaded {len(embeddings_dict)} words with dimension {embedding_dim}")
    print("Example vector for 'movie':", embeddings_dict.get('movie'))
