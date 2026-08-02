# From-Kernel-to-Silicon---Transformers

<div align="center">

![Transformers Cover](https://miro.medium.com/v2/resize:fit:1200/1*-BVYEaV8IlxCWGk7pgxCZg.png)

### **From-Kernel-to-Silicon: Building a Decoder-Only Transformer from Scratch**

*A step-by-step engineering journey constructing an autoregressive Generative Pre-trained Transformer (GPT) in PyTorch, tracing every architectural evolution from raw matrix lookups to modern deep multi-head attention.*

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

</div>

---

##  Project Overview

This repository documents the ground-up development of a **Decoder-Only GPT architecture** trained on the TinyShakespeare dataset (~1.11M characters). 

Rather than relying on high-level abstractions like `torch.nn.TransformerDecoder`, this project builds every component—positional embeddings, causal scaled dot-product attention, multi-head parallelization, residual paths, and normalization layers—from first principles.

The final model scales to **~10.8M parameters**, capable of learning spatial character dependencies, grammatical structures, and poetic cadence.

---

##  Why a Decoder-Only Architecture? 

Unlike the original 2017 Transformer (*Vaswani et al.*), which used an **Encoder-Decoder** setup for sequence-to-sequence translation (e.g., English to French), this project uses a **Decoder-Only** architecture (the standard for GPT-3, LLaMA, and Mistral).

* **Autoregressive Task Alignment**: The core objective is **causal language modeling**—predicting the next character based strictly on past tokens ($P(x_t \mid x_{<t})$). An encoder processes tokens bidirectionally (looking both left and right), which violates the strict causal constraint required for autoregressive generation.
* **Redundancy of Cross-Attention**: In encoder-decoder models, the decoder uses **Cross-Attention** layers to attend to the encoder's hidden states. For unconstrained or prompt-conditioned generation, the prompt and generated text exist in the same continuous token stream—making cross-attention redundant.
* **Unified Sequence Processing**: With causal masking, a single stack of decoder layers processes both the context (the prompt) and the generation (the target) within the exact same self-attention mechanism.
* **Parameter & Computational Efficiency**: Stripping away the encoder and cross-attention sub-layers removes unnecessary parameters and memory overhead, allowing all capacity to be focused on unified deep sequence modeling.

---

##  Hyperparameters at a Glance

| Parameter | Value | Description |
| :--- | :---: | :--- |
| **Context Window ($T$)** | `256` | Maximum sequential history per forward pass |
| **Batch Size ($B$)** | `64` | Parallel independent sequence streams |
| **Embedding Dimension ($C$)** | `384` | Channel vector dimension ($n_{\text{embed}}$) |
| **Attention Heads ($n_{\text{head}}$)** | `6` | Head dimension $d_k = 384 / 6 = 64$ |
| **Transformer Blocks ($n_{\text{layer}}$)** | `6` | Sequential stacked residual layers |
| **Total Parameters** | **~10.8M** | Fully trainable parameters |
| **Vocabulary Size ($V$)** | `65` | Unique character-level tokens |

---

##  The Incremental Evolution Journey

The codebase is organized into sequential stages, illustrating how resolving specific architectural bottlenecks drives performance gains:

### Evolution Breakdown

1. **V01 - Naive Bigram Lookup**: Base statistical frequency table; zero spatial horizon ($T=1$).
2. **V02 - Context & Positional Encodings**: Extends sequence window ($T=256$) and adds positional embeddings.
3. **V03 - Causal Scaled Dot-Product Attention**: Introduces Key, Query, Value projections and lower-triangular causal masking.
4. **V04 - Multi-Head Attention**: Parallelizes context extraction across multiple representation subspaces.
5. **V05 - Position-Wise FFN**: Adds a $4\times$ channel expansion feed-forward step for intra-token non-linear processing.
6. **V06 - Residual Connections**: Resolves vanishing gradients across deep layer stacks using identity skip paths ($x + F(x)$).
7. **V07 - Pre-Layer Normalization**: Stabilizes training variance by placing LayerNorm before attention and FFN blocks.
8. **V08 - Regularization & Final System**: Adds dropout across projections, residual paths, and attention weights for generalized sampling.

*Details of evolution at stages and Walkthrough given in status.md*

