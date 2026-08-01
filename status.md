# Decoder-Only Transformer (GPT) Architectural Evolution & Status Report

## 1. Executive Summary & Project Overview

This document provides a comprehensive, chronological status report on the architectural evolution of a **Decoder-Only Generative Pre-trained Transformer (GPT)** built from scratch in PyTorch. Trained on the **TinyShakespeare** dataset, this model demonstrates the step-by-step transition from a simple, zero-context Bigram lookup table to a multi-layer, multi-head self-attention Transformer language model with ~10.8 million parameters.

### Current Hyperparameter Configuration
* **Batch Size ($B$)**: 64 (independent sequences processed in parallel)
* **Block Size / Context Length ($T$)**: 256 tokens
* **Embedding Dimension ($C$ / $n_{	ext{embed}}$)**: 384
* **Attention Heads ($n_{	ext{head}}$)**: 6 (Head size $d_k = 384 / 6 = 64$)
* **Transformer Layers ($n_{	ext{layer}}$)**: 6 blocks
* **Dropout Rate**: 0.2
* **Learning Rate**: $3 	imes 10^{-4}$ (AdamW Optimizer)
* **Total Parameters**: ~10.8M
* **Dataset**: TinyShakespeare (`input.txt`), ~1.11M characters, Vocabulary size $V = 65$ unique characters.

---

### Evolutionary Summary Matrix

| Version | Architectural Upgrade | Key Mechanism / Innovation | Train Loss | Val Loss | Primary Limitation Solved | Remaining Bottleneck |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **V01** | Naive Bigram Model | Single lookup table ($V 	o V$) | ~2.50 | ~2.52 | Baseline token mapping | No context horizon ($T=1$) |
| **V02** | Context Window & Positional Embedding | $T=256$, Batching, $E_{	ext{tok}} + E_{	ext{pos}}$ | ~2.42 | ~2.45 | Extended sequence length | Tokens cannot communicate |
| **V03** | Single-Head Causal Self-Attention | $Q, K, V$ projections + Scaled Dot-Product | ~2.20 | ~2.24 | Enables historical communication | Single feature relation channel; no computation depth |
| **V04** | Multi-Head Self-Attention | 6 parallel heads concatenated + projection | ~2.12 | ~2.18 | Captures diverse subspace relations | Lack of position-wise non-linear feature transformation |
| **V05** | Position-Wise FeedForward Network | Linear($C 	o 4C$) $	o$ ReLU $	o$ Linear($4C 	o C$) | ~1.91 | ~2.05 | Per-token computation & capacity | Vanishing gradients in deeper network stacks |
| **V06** | Transformer Blocks & Residual Connections | 6 Stacked Blocks with $x + F(x)$ skip paths | ~1.90 | ~2.03 | Gradient flow through deep network | Internal Covariate Shift & unstable activation distributions |
| **V07** | Pre-Layer Normalization | Pre-LN before Attention & FFN | ~1.93 | ~2.04 | Stable variance/mean per token vector | High training variance & overfitting on small dataset |
| **V08** | Regularization & Full Architecture | Dropout (0.2) in MHA, FFN, and Residuals | **~2.05** | **~2.11** | Overfitting mitigation; generalized sampling | Character-level tokenization limits semantic density |

---

## 2. Theoretical Foundations & Key Concepts

### 2.1 Autoregressive Language Modeling
The goal of an autoregressive generative model is to estimate the joint probability distribution of a sequence of tokens $X = (x_1, x_2, \dots, x_T)$. Using the chain rule of probability, this is factorized into a product of conditional probabilities:

$$P(X) = \prod_{t=1}^{T} P(x_t \mid x_1, x_2, \dots, x_{t-1})$$

The model processes context tokens $x_{<t}$ and outputs a probability vector over the vocabulary $V$ for the next token $x_t$.

### 2.2 Character Tokenization & Integer Mapping
The TinyShakespeare corpus is mapped character-by-character:
* Vocabulary size $V = 65$ (letters, digits, punctuation, whitespace).
* Encoder mapping: $	ext{stoi}: 	ext{char} 	o \mathbb{Z} \cap [0, V-1]$.
* Decoder mapping: $	ext{itos}: \mathbb{Z} \cap [0, V-1] 	o 	ext{char}$.

### 2.3 Cross-Entropy Loss & Optimization
Given predicted logits $z_t \in \mathbb{R}^V$ and target character class $y_t \in \{0, \dots, V-1\}$:

$$P(x_t = y_t) = rac{\exp(z_{t, y_t})}{\sum_{j=1}^{V} \exp(z_{t, j})}$$

$$\mathcal{L}_{	ext{CE}} = -rac{1}{B \cdot T} \sum_{i=1}^{B} \sum_{t=1}^{T} \ln P(x_{i,t} = y_{i,t})$$

Optimization is performed using **AdamW** with $ eta_1 = 0.9,  eta_2 = 0.999$, weight decay, and a learning rate of $3 	imes 10^{-4}$.

---

## 3. Detailed Architectural Breakdown & Timeline

### Stage 1: Version 01 — Naive Bigram Language Model

#### Overview & Purpose
The baseline implementation predicts the next token solely based on the identity of the current token, ignoring all previous history.

```
Input Index x_t (B, 1) ---> Lookup Table (V x V) ---> Logits (B, 1, V) ---> Softmax ---> Next Token Prediction
```

#### Core Mathematics
$$z_t = W_{	ext{embed}}[x_t], \quad W_{	ext{embed}} \in \mathbb{R}^{V 	imes V}$$
$$P(x_{t+1} \mid x_t) = 	ext{Softmax}(z_t)$$

#### Code Implementation
```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # (B, T, V)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
```

#### Tensor Dimensions
* Input `idx`: $(B, T)$ where $T=1$
* Token Embeddings: $(B, 1, 65)$
* Logits reshaped: $(B \cdot 1, 65)$

#### Why It Failed / Bottleneck
* Zero temporal horizon ($T=1$). A bigram model cannot learn multi-character words, sentence structure, or character role dependencies (e.g., character names in Shakespeare).
* Cross-entropy loss remains high (~2.50), yielding gibberish output.

---

### Stage 2: Version 02 — Positional Embeddings & Context Windows

#### Overview & Purpose
To allow predicting based on context, we expand the sequence window to $T = 	ext{block\_size} = 256$ and introduce **Positional Embeddings** to inform the network of token locations.

```
Token Indices (B, T) ---> Token Embedding (B, T, C) --+
                                                       |---> Sum (B, T, C) ---> Linear Head ---> Logits (B, T, V)
Position Indices (T) ---> Pos Embedding   (T, C)    --+
```

#### Core Mathematics
$$x_t = E_{	ext{tok}}[i_t] + E_{	ext{pos}}[t]$$
where $E_{	ext{tok}} \in \mathbb{R}^{V 	imes C}$ and $E_{	ext{pos}} \in \mathbb{R}^{T 	imes C}$.

#### Code Implementation
```python
self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
self.positional_embedding_table = nn.Embedding(block_size, n_embed)

# Forward pass
tok_embd = self.token_embedding_table(idx) # (B, T, C)
pos_embd = self.positional_embedding_table(torch.arange(T, device=device)) # (T, C)
x = tok_embd + pos_embd # (B, T, C) via broadcasting
```

#### Tensor Dimensions
* Input `idx`: $(64, 256)$
* Token Embedding `tok_embd`: $(64, 256, 384)$
* Positional Embedding `pos_embd`: $(256, 384)$
* Combined Representation `x`: $(64, 256, 384)$

#### Bottleneck
Adding positional embeddings to a flat vector representation without cross-token interaction does not give the network a mechanism to transfer information between timesteps. Tokens remain isolated islands.

---

### Stage 3: Version 03 — Single-Head Causal Self-Attention

#### Overview & Purpose
This version introduces **Scaled Dot-Product Self-Attention** with a causal mask, enabling tokens to aggregate context from past tokens while prohibiting future peeking.

```
                     +---> Query (Q) ---> Q @ K^T / sqrt(d_k) ---> Causal Mask ---> Softmax ---> Weights
Input Vector (B,T,C) |---> Key   (K) -----------^                                                   |
                     +---> Value (V) -------------------------------------------------> Weights @ V ---> Output (B,T,d_k)
```

#### Core Mathematics
For input sequence $X \in \mathbb{R}^{B 	imes T 	imes C}$:

1. **Linear Projections**:
   $$Q = X W_Q, \quad K = X W_K, \quad V = X W_V \quad (W_Q, W_K, W_V \in \mathbb{R}^{C 	imes d_k})$$

2. **Attention Score Matrix**:
   $$A = rac{Q K^T}{\sqrt{d_k}} \in \mathbb{R}^{B 	imes T 	imes T}$$

3. **Causal Masking & Softmax**:
   $$	ilde{A}_{i,j} =  egin{cases} A_{i,j} & 	ext{if } i \ge j \ -\infty & 	ext{if } i < j \end{cases}$$
   $$W = 	ext{Softmax}(	ilde{A}) \in \mathbb{R}^{B 	imes T 	imes T}$$

4. **Weighted Value Aggregation**:
   $$	ext{Out} = W V \in \mathbb{R}^{B 	imes T 	imes d_k}$$

#### Code Implementation
```python
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        # Scaled Dot-Product Attention
        weights = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5) # (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.drop(weights)

        out = weights @ v # (B, T, head_size)
        return out
```

*Note on Implementation Detail*: In the raw code snippet, `C**(-0.5)` used `C` from `x.shape` ($n_{	ext{embed}} = 384$) rather than $d_k = 64$. Standard Transformer practice scales by $d_k^{-0.5} = 64^{-0.5} = 0.125$ to keep variance at 1. Correcting scaling to `k.shape[-1]**(-0.5)` ensures optimal gradient scaling regardless of embedding size.

#### Loss Milestone
* Loss dropped from **~2.40** down to **~2.20**.

#### Bottleneck
A single attention head can only focus on one type of relation (e.g., matching consonants to vowels). It cannot simultaneously track syntax, rhyming, and structural boundaries.

---

### Stage 4: Version 04 — Multi-Head Self-Attention (Parallel Channels)

#### Overview & Purpose
Multi-Head Attention runs multiple self-attention heads in parallel ($h=6$, $d_k=64$). Each head projects inputs into different representation subspaces, allowing the model to process diverse relationships concurrently.

```
                 +---> Head 1 (64-dim) --->|
                 |---> Head 2 (64-dim) --->|
Input (B, T, 384)|---> Head 3 (64-dim) --->|---> Concat (B,T,384) ---> Linear Projection ---> Drop ---> Output (B,T,384)
                 |---> Head 4 (64-dim) --->|
                 |---> Head 5 (64-dim) --->|
                 +---> Head 6 (64-dim) --->|
```

#### Core Mathematics
$$	ext{head}_i = 	ext{Attention}(X W_Q^i, X W_K^i, X W_V^i)$$
$$	ext{MultiHead}(X) = 	ext{Concat}(	ext{head}_1, 	ext{head}_2, \dots, 	ext{head}_h) W_O$$
where $W_O \in \mathbb{R}^{C 	imes C}$.

#### Code Implementation
```python
class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, h * head_size)
        out = self.proj(out) # (B, T, n_embed)
        return self.drop(out)
```

#### Tensor Dimensions
* Input `x`: $(64, 256, 384)$
* Each Head Output: $(64, 256, 64)$
* Concatenated Output: $(64, 256, 384)$
* Projected Output: $(64, 256, 384)$

#### Bottleneck
Attention aggregates and blends information across tokens, but does not provide individual, token-wise non-linear feature transformation. Tokens communicate, but lack time to "digest" or "compute" on gathered context.

---

### Stage 5: Version 05 — Position-Wise FeedForward Network

#### Overview & Purpose
A FeedForward Network (FFN) is appended after attention. While attention acts as a communication operator across tokens, the FFN acts as a computation operator applied independently to each token vector.

```
Token Vector (384-dim) ---> Linear Projection (1536-dim) ---> ReLU ---> Linear Projection (384-dim) ---> Output
```

#### Core Mathematics
$$	ext{FFN}(x) = 	ext{ReLU}(x W_1 + b_1) W_2 + b_2$$
where $W_1 \in \mathbb{R}^{C 	imes 4C}$, $b_1 \in \mathbb{R}^{4C}$, $W_2 \in \mathbb{R}^{4C 	imes C}$, and $b_2 \in \mathbb{R}^C$.

#### Why the $4 	imes n_{	ext{embed}}$ Expansion?
Expanding the channel dimension by 4x ($384 	o 1536$) creates a higher-dimensional space where non-linear combinations of features gathered during attention can be disentangled and stored.

#### Code Implementation
```python
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.ffnet = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffnet(x)
```

#### Loss Milestone
* Loss dropped significantly from **~2.28** to **~1.91 (train)** and **~2.05 (val)**.

#### Bottleneck
Stacking multiple layers of Attention + FFN causes vanishing/exploding gradients during backpropagation, making deep architectures (>2 blocks) unstable or un-trainable.

---

### Stage 6: Version 06 — Transformer Blocks & Residual Connections

#### Overview & Purpose
To scale the model depth to 6 layers, we introduce **Residual (Skip) Connections** proposed by He et al. (2015). Skip connections allow gradients to flow unimpeded directly back to earlier layers.

```
          +------------------------+ (Skip Connection)
          |                        v
x ---> [LayerNorm] ---> [MultiHead] ---> (+) ---> [LayerNorm] ---> [FeedForward] ---> (+) ---> Output
  |                                        ^   |                                        ^
  +----------------------------------------+   +----------------------------------------+
```

#### Core Mathematics & Gradient Flow
$$x_{l+1} = x_l + F(x_l)$$

When computing the derivative of loss $\mathcal{L}$ with respect to activation $x_l$:
$$rac{\partial \mathcal{L}}{\partial x_l} = rac{\partial \mathcal{L}}{\partial x_{l+1}} rac{\partial x_{l+1}}{\partial x_l} = rac{\partial \mathcal{L}}{\partial x_{l+1}} \left( I + rac{\partial F(x_l)}{\partial x_l} 
ight)$$

The identity operator $I$ guarantees that error gradients can flow backward to early layers even if intermediate weight derivatives $rac{\partial F(x_l)}{\partial x_l}$ approach zero.

#### Code Implementation
```python
class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.attention = MultiHead(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.attention(self.ln1(x)) # Residual connection 1
        x = x + self.ffwd(self.ln2(x))       # Residual connection 2
        return x
```

---

### Stage 7: Version 07 — Layer Normalization & Pre-LN Architecture

#### Overview & Purpose
Internal Covariate Shift causes layer input distributions to wander during optimization. We apply **Layer Normalization** to keep activations standardized with zero mean and unit variance.

#### Layer Normalization vs. Batch Normalization
* **Batch Normalization (BatchNorm)**: Computes mean and variance across the *batch dimension* ($N$). Sensitive to batch size and dynamic sequence lengths.
* **Layer Normalization (LayerNorm)**: Computes mean and variance across the *channel dimension* ($C$) independently for each token vector.

```
Batch Normalization:                  Layer Normalization:
  (Mean/Var calculated vertically)      (Mean/Var calculated horizontally)
       [Batch 1] [Batch 2] ...               [Token 1] -> Mean/Var over C
Dim 1  x11       x21                   Dim 1  x11
Dim 2  x12       x22                   Dim 2  x12
...                                    ...
Dim C  x1C       x2C                   Dim C  x1C
```

#### Mathematical Formulation
For a token vector $x \in \mathbb{R}^C$:
$$\mu = rac{1}{C} \sum_{i=1}^C x_i, \quad \sigma^2 = rac{1}{C} \sum_{i=1}^C (x_i - \mu)^2$$
$$\hat{x}_i = rac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma_i +  eta_i$$
where $\gamma,  eta \in \mathbb{R}^C$ are learnable scale and shift parameters.

#### Pre-LN vs. Post-LN
* **Post-LN (Original Transformer 2017)**: $x_{l+1} = 	ext{LayerNorm}(x_l + F(x_l))$. Requires learning rate warmup to prevent gradient exploding early in training.
* **Pre-LN (Modern GPT Standard)**: $x_{l+1} = x_l + F(	ext{LayerNorm}(x_l))$. Keeps the main residual path completely clean, enabling immediate stable training without warmups.

#### Performance Status
* Step 4800: **Train Loss: 1.9299**, **Val Loss: 2.0375**.

---

### Stage 8: Version 08 — Regularization (Dropout) & Final GPT System

#### Overview & Purpose
To prevent overfitting on the 1.1M character TinyShakespeare dataset, **Dropout (0.2)** is injected into:
1. Attention softmax weights (dropping attention edges).
2. Multi-Head projection layer outputs.
3. FeedForward Network layer outputs.

#### Complete System Architecture Summary

```
====================================================================================================
Layer / Submodule               Input Shape             Output Shape            Parameters
====================================================================================================
Token Embedding Table           (B, T)                  (B, T, 384)             65 x 384 = 24,960
Positional Embedding Table      (T,)                    (T, 384)                256 x 384 = 98,304
----------------------------------------------------------------------------------------------------
Block 1..6 (x6 Stacked):
  ├── LayerNorm 1               (B, T, 384)             (B, T, 384)             2 x 384 = 768
  ├── Multi-Head Attention:
  │   ├── Key Linear            (B, T, 384)             (B, T, 384)             384 x 384 = 147,456
  │   ├── Query Linear          (B, T, 384)             (B, T, 384)             384 x 384 = 147,456
  │   ├── Value Linear          (B, T, 384)             (B, T, 384)             384 x 384 = 147,456
  │   └── Out Projection        (B, T, 384)             (B, T, 384)             384 x 384 + 384 = 147,840
  ├── LayerNorm 2               (B, T, 384)             (B, T, 384)             2 x 384 = 768
  └── FeedForward Net:
      ├── FC 1 (Expand)         (B, T, 384)             (B, T, 1536)            384 x 1536 + 1536 = 591,360
      └── FC 2 (Contract)       (B, T, 1536)            (B, T, 384)             1536 x 384 + 384 = 590,208
----------------------------------------------------------------------------------------------------
Final LayerNorm                 (B, T, 384)             (B, T, 384)             2 x 384 = 768
Language Model Head             (B, T, 384)             (B, T, 65)              384 x 65 + 65 = 25,025
====================================================================================================
Total Parameters: ~10,788,929 (~10.8 Million)
====================================================================================================
```

#### Final Training Results
* Step 4800: **Train Loss: 2.0543**, **Val Loss: 2.1065**.
* Note: The loss values with dropout are slightly higher than without dropout (1.92 vs 2.05), but the train-val gap is tighter, indicating superior generalization during generation.

---

## 4. Parameter Count & Memory Analysis

### Exact Parameter Breakdown Table

| Layer Component | Mathematical Dimension | Formula | Exact Parameter Count |
| :--- | :--- | :--- | :---: |
| **Token Embedding** | $V 	imes C$ | $65 	imes 384$ | 24,960 |
| **Position Embedding** | $T 	imes C$ | $256 	imes 384$ | 98,304 |
| **6 x Transformer Blocks** | | | |
| - *LN1 & LN2 ($\gamma,  eta$)* | $2 	imes (2 	imes C)$ | $6 	imes 2 	imes (2 	imes 384)$ | 9,216 |
| - *Attention ($W_Q, W_K, W_V$)* | $3 	imes (C 	imes C)$ | $6 	imes 3 	imes (384 	imes 384)$ | 2,654,208 |
| - *Attention Proj ($W_O, b_O$)* | $C 	imes C + C$ | $6 	imes (384 	imes 384 + 384)$ | 887,040 |
| - *FFN FC1 ($W_1, b_1$)* | $C 	imes 4C + 4C$ | $6 	imes (384 	imes 1536 + 1536)$ | 3,548,160 |
| - *FFN FC2 ($W_2, b_2$)* | $4C 	imes C + C$ | $6 	imes (1536 	imes 384 + 384)$ | 3,541,248 |
| **Final LayerNorm** | $2 	imes C$ | $2 	imes 384$ | 768 |
| **Language Model Head** | $C 	imes V + V$ | $384 	imes 65 + 65$ | 25,025 |
| **TOTAL** | | | **10,788,929** (~10.8M) |

---

## 5. Next Evolution Steps & Roadmap (Version 09+)

To upgrade this decoder-only Transformer into modern state-of-the-art architectures (GPT-2 / LLaMA / Mistral standard), the following steps are recommended:

### 1. Vectorized Multi-Head Attention
* **Current**: Uses `nn.ModuleList([Head(...) for _ in range(num_heads)])` which launches individual PyTorch kernels in a Python loop.
* **Proposed Upgrade**: Combine Query, Key, and Value projections into a single fused projection `nn.Linear(n_embed, 3 * n_embed)`. Reshape tensors to $(B, n_{	ext{head}}, T, d_k)$ for matrix operations.

### 2. PyTorch Native Scaled Dot-Product Attention (FlashAttention)
* **Proposed Upgrade**: Replace manual score calculation and triangular masking with `torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)`. This unlocks fused CUDA kernels (FlashAttention), reducing memory complexity from $O(T^2)$ to $O(T)$ and speeding up training significantly.

### 3. GELU Activation Function
* **Proposed Upgrade**: Replace `nn.ReLU()` in FeedForward with `nn.GELU(approximate='tanh')`. GELU (Gaussian Error Linear Unit) provides smooth gradients around zero, preventing dead neurons.

### 4. Weight Tying (Embedding & Final LM Head)
* **Proposed Upgrade**: Tie the weights of `token_embedding_table.weight` with `lm_head.weight`. Since both represent linear mappings between discrete tokens and dense embeddings, weight sharing reduces parameter count by 25k and improves generalization.

### 5. Learning Rate Decay Schedule (Cosine Annealing with Warmup)
* **Proposed Upgrade**: Implement a linear warmup for the first 100-500 steps followed by a cosine decay schedule down to $0.1 	imes 	ext{lr}_{	ext{max}}$.

### 6. Sub-Word Tokenization (Tiktoken / BPE)
* **Proposed Upgrade**: Shift from character-level tokenization ($V=65$) to Byte-Pair Encoding (BPE, $V=50,257$ via OpenAI `tiktoken`). This increases text density per context window by ~3x, allowing $T=256$ to cover long-range paragraph structures.

---
## Final Status: 
```
step 0: train loss 4.3585, val loss 4.3557
step 500: train loss 2.0170, val loss 2.0917
step 1000: train loss 1.6125, val loss 1.7842
step 1500: train loss 1.4475, val loss 1.6449
step 2000: train loss 1.3575, val loss 1.5784
step 2500: train loss 1.2872, val loss 1.5347
step 3000: train loss 1.2367, val loss 1.5039
step 3500: train loss 1.1909, val loss 1.4929
step 4000: train loss 1.1529, val loss 1.4847
step 4500: train loss 1.1201, val loss 1.4798

Provost:
No, no, to dry sore stamping to him,--took to bring a
perishment chaves: what's an at an upround; if
preservement drowry you and a descent but ere his face?

GLOUCESTER:
Provide him, and hath sent and judgment realtice
A given and times most himself. Hard you.

LARTIUS:
Are you come to something further? why
high cot was i' the gift?

LAMILLIUS:
As come cornation.

LAUCIO:
I tell Faith her,
He wards us slaved his course.

POMPEY:
Ah, my master Potmass: dam daughters, how he
Though the v
`
*Report generated automatically from code evaluation.*
```
---
*Report generated automatically from code evaluation.*

## Resources:
* Andrej Karpathy Let's Build GPT from scratch
* PyTorch Documentation
* Google AI
* Attention is All you Need - 2017
* CampusX - Transformers
* Krish Naik - Complete Transformers for NLP
* Umar Jamil - Attention is all you need (Transformer) - Model explanation (including math), Inference and Training
* 3Bluuebrown - transformers, the tech behind LLMs

---