# Project Guidelines & Standing Rules

## 1. Project Goal
Build and train a modern ~50M parameter Decoder-Only Transformer LLM (`MiniLlama-50M`) from scratch in pure PyTorch, capable of running local CPU smoke tests on this laptop and full training on a Google Colab T4 GPU (16GB VRAM).

## 2. Architecture Standards
Use modern Llama-3/Qwen-3 architecture blocks:
- **Normalization**: RMSNorm with Pre-Norm formulation.
- **Positional Encoding**: Rotary Positional Embeddings (RoPE).
- **Attention**: Grouped-Query Attention (GQA with 8 query heads and 2 KV heads) using `torch.nn.functional.scaled_dot_product_attention`.
- **Feed-Forward Network**: SwiGLU Feed-Forward Networks.
- **Embeddings**: Tied token embeddings (weight sharing between input token embedding and output language model head).

## 3. Coding Standards
- **Clarity & Modularity**: Write clean, modular, self-contained Python files with type hints.
- **No Wrapper Frameworks**: Never use heavy wrapper frameworks like `transformers.Trainer` for the core model.
- **Pure PyTorch**: Write explicit PyTorch `nn.Module` classes and native training loops so every tensor transformation is transparent and readable.
