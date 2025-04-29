# Transformer Encoder from Scratch (Paper2Code)

This project implements the **Transformer Encoder** from the paper ["Attention is All You Need" (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762), **from scratch** using **NumPy only** — no deep learning frameworks.

It closely follows the paper structure:
- Scaled Dot-Product Attention
- Multi-Head Attention
- Positional Encoding
- FeedForward Networks
- Layer Normalization + Residual Connections
- Stacked Encoder Blocks

---

## Project Structure

- `attention.py` : Scaled Dot-Product Attention and Multi-Head Attention
- `positional_encoding.py` : Sinusoidal positional encoding
- `feedforward.py` : Two-layer Feed Forward network
- `encoder_block.py` : One Transformer Encoder Block
- `transformer_encoder.py` : Full Transformer Encoder (stack of blocks)
- `glove_loader.py` : Load pre-trained GloVe embeddings
- `train_toy_example.py` : Train Transformer + simple classifier on toy synthetic task
- `test_sentence.py` : Pass real-world sentences through the Transformer Encoder for prediction

---

## How to Run

### Install Requirements
```bash
pip install numpy matplotlib
```
## (Optional) Download GloVe Embeddings
- Download GloVe 6B embeddings
- Extract and place glove.6B.50d.txt in your project directory.

## Run Examples
1. Train Transformer on Toy Task
```bash
python train_toy_example.py
```
This trains the Transformer + Classifier on a synthetic dataset.

2. Test on Real Sentence
```bash
python test_sentence.py
This processes a real English sentence and predicts a class.
```
## What's Implemented
- Scaled Dot-Product Attention
- Multi-Head Attention
- Sinusoidal Positional Encoding
- FeedForward Networks
- Residual + Layer Normalization
- Stacking of Encoder Blocks
- Manual Training Loop (NumPy)
- Integration of GloVe real embeddings
- Real sentence inference testing

## What's Next
- Migrate project to PyTorch
- Fine-tune full Transformer Encoder on real datasets (e.g., IMDB)
- Visualize attention maps
- Build full Transformer (add Decoder block)

Built to learn deeply from research papers.

---

## References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).  
  ["Attention is All You Need"](https://arxiv.org/abs/1706.03762). *Advances in Neural Information Processing Systems (NeurIPS)*.

This project implements the Transformer Encoder architecture described in the paper above for educational and research purposes.
