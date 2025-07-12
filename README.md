# 🧠 MicroGPT — Tiny Decoder-Only Transformer from Scratch

> A minimal, readable implementation of GPT-style language models in PyTorch.

![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)

## 🚀 Overview

**MicroGPT** is a decoder-only Transformer built from scratch using PyTorch, with:

- ✅ Scaled dot-product attention
- ✅ Multi-head self-attention
- ✅ Positional encoding (sinusoidal)
- ✅ LayerNorm, residual connections
- ✅ Mini GPT architecture (decoder-only)
- ✅ BLEU evaluation and ONNX export support

This is not a full GPT-2 clone, but a **minimal educational version** that trains on small datasets like WikiText2 and can generate coherent text after a few epochs.

---

## 📦 Model Architecture

```text
Input → Token Embedding + Positional Encoding
      → N x [Masked Multi-Head Attention → FeedForward → LayerNorm + Residual]
      → Linear → Softmax (vocab logits)
````

* Total Parameters: \~8.1M
* Blocks: 2 decoder layers
* Embedding Dim: 128
* Heads: 4
* FFN Dim: 512

---

## 🛠️ Usage

### 1. Install dependencies

```bash
pip install torch nltk tqdm datasets tokenizers
```

### 2. Train the model

```python
python gpt.py
```

Modify hyperparameters in `gpt.py`:

```python
BLOCK_SIZE = 64
EMBED_DIM = 128
NUM_LAYERS = 2
```

### 3. Generate Text

```python
prompt = "The future of AI"
print(generate_text(model, tokenizer, prompt))
```

### 4. Evaluate BLEU Score

```python
evaluate_bleu(model, clean_texts(train_texts), tokenizer)
```

### 5. Export to ONNX

```python
torch.onnx.export(model, ...)
```

---

## 📈 Example Output

```text
Prompt: Once upon a time
Output: Once upon a time , the of the , and the of the . The
```

For 1–2 epochs, this is expected. For performance improvement training with larger data and larger epoch is necessary(large systems are also required atleast 12$ plan by colab).

---

## 🧪 Goals

This repo is **educational-first**, aimed at helping you:

* Understand GPT internals
* Train a small LM from scratch
* Export and deploy with ONNX
* Evaluate using BLEU

---

## ✍️ Author

Built with ❤️ by [@Naren](https://github.com/your-username) part of a self-built NLP curriculum.
