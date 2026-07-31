# Understanding the Construction of a GPT (Generative Pretrained Transformer)

To understand the construction of a GPT (**Generative Pretrained Transformer**), we must track how data flows through tensors. In this lecture, Andrej Karpathy builds a **decoder-only Transformer** for character-level language modeling.

---

# 1. Fundamental Data & Hyperparameters

> **Time:** 0:09:28 – 0:21:20

These are the four fundamental dimensions that define almost every tensor inside the model.

| Symbol | Parameter | Meaning | Example |
|---------|-----------|---------|---------|
| **B** | `batch_size` | Number of independent sequences processed simultaneously. This keeps the GPU fully utilized. | 4, 32, 64 |
| **T** | `block_size` | Maximum context length (number of previous tokens the model is allowed to look at). | 8, 128, 256 |
| **C** | `vocab_size` | Total number of unique tokens (characters for a character-level GPT). Determines embedding lookup size and final output dimension. | 65 (Tiny Shakespeare) |
| — | `n_embed` | Size of the internal vector space where every token is represented. | 32, 128, 384, 768 |

---

# 2. The Tensor Lifecycle (How Data Flows)

---

## Step 1 — Raw Input

The model first receives integer token IDs.

```text
idx
Shape = (B, T)
```

Example:

```python
tensor([
 [3,6,5,0,14,7,10,10],
 [5,11,12,0,3,4,8,5],
 [12,0,2,13,0,1,9,9],
 ...
])
```

Each integer represents one token.

---

## Step 2 — Token Embedding

Every integer index is converted into a learned vector.

Embedding table:

```text
(vocab_size, n_embed)
```

Example:

```text
(65,384)
```

The lookup operation transforms

```text
(B,T)
```

into

```text
(B,T,n_embed)
```

---

## Step 3 — Positional Embedding

Transformers have **no inherent concept of order**.

Therefore another embedding table is learned.

Its shape is

```text
(block_size,n_embed)
```

Example

```text
(256,384)
```

Each position

```
0
1
2
...
255
```

has its own learnable vector.

---

## Step 4 — Combine Both Embeddings

The two embeddings are simply added.

```python
x = token_embedding(idx) + position_embedding
```

Shape remains

```text
(B,T,n_embed)
```

Now each token knows

- **what it is**
- **where it occurs**

---

# 3. Self-Attention Block

The combined tensor enters the Transformer block.

---

## Query

```text
Q = Linear(x)
```

Shape

```text
(B,T,head_size)
```

---

## Key

```text
K = Linear(x)
```

Shape

```text
(B,T,head_size)
```

---

## Value

```text
V = Linear(x)
```

Shape

```text
(B,T,head_size)
```

Each token now owns three learned vectors.

---

### Intuition

**Query**

> "What information am I looking for?"

**Key**

> "What information do I contain?"

**Value**

> "What information should I send if selected?"

---

# 4. Computing Attention Scores

Attention scores are computed using

```python
wei = Q @ K.transpose(-2,-1)
```

Tensor shape

```text
(B,T,T)
```

This creates a communication matrix.

Every token compares itself against every previous token.

---

# 5. Causal Mask (Lower Triangular Matrix)

GPT must **never look into the future**.

Therefore we apply

```python
tril = torch.tril(torch.ones(T,T))
```

Future positions become

```text
-∞
```

before Softmax.

Only previous tokens remain visible.

This is what makes GPT an **autoregressive model**.

---

# 6. Softmax

The attention matrix is normalized.

```python
wei = softmax(wei)
```

Every row becomes a probability distribution.

The probabilities indicate

> **How much should this token pay attention to every previous token?**

---

# 7. Weighted Sum of Values

The communication happens here.

```python
out = wei @ V
```

Shape

```text
(B,T,head_size)
```

Each token now contains information aggregated from every relevant earlier token.

---

# 8. Feed Forward Network

Attention allows tokens to communicate.

The Feed Forward Network allows every token to **think independently**.

Typical structure

```python
Linear
↓

ReLU / GELU

↓

Linear
```

Applied independently at every position.

Input

```text
(B,T,n_embed)
```

Output

```text
(B,T,n_embed)
```

---

# 9. LM Head (Language Modeling Head)

Finally,

```python
Linear(n_embed,vocab_size)
```

projects

```text
(B,T,n_embed)
```

to

```text
(B,T,vocab_size)
```

These are the raw logits for predicting the next token.

---

# 10. Decoder-Only vs Encoder-Decoder

> **Time:** 1:42:39 – 1:46:22

The original **Attention Is All You Need (2017)** architecture was created for **Machine Translation**.

Example

```text
French
↓

Encoder

↓

Decoder

↓

English
```

---

## Encoder

Receives the entire input sentence.

There is **no causal mask**.

Every word can attend to every other word.

Output

A contextual representation of the full sentence.

---

## Decoder

Contains two attention mechanisms.

### Masked Self-Attention

Looks only at previously generated words.

---

### Cross Attention

Allows the decoder to query the encoder output.

This is how the decoder reads the translated input sentence.

---

## Decoder-Only GPT

Karpathy's implementation removes

- Encoder
- Cross Attention

because the task is

> Predict the next token given previous tokens.

Therefore only **Masked Self-Attention** is required.

---

# Walkthrough Using the Sentence

## "The Winner Takes It All"

---

# Phase 1 — Vocabulary Construction

Sentence

```text
The Winner Takes It All
```

---

## Step 1 — Extract Unique Characters

Sorted vocabulary

```python
vocabulary = [
' ',
'A',
'I',
'T',
'a',
'e',
'h',
'i',
'k',
'l',
'n',
'r',
's',
't',
'w'
]
```

```python
vocab_size = 15
```

---

## Step 2 — Encoder / Decoder Mapping

Character → Integer

```text
' ' → 0
'A' → 1
'I' → 2
'T' → 3
'a' → 4
...
'w' → 14
```

Sentence becomes

```text
"T  h  e     W  i   n   n   e  r   s     T  a  k  e  s     I  t     A  l  l"
```

Encoded tensor

```python
[
3,6,5,0,
14,7,10,10,5,11,12,
0,
3,4,8,5,12,
0,
2,13,
0,
1,9,9
]
```
---

# Phase 2 — The Old Neural Bigram Model

The original model used

```python
nn.Embedding(vocab_size,vocab_size)
```

For this example

```python
nn.Embedding(15,15)
```

---

## The Embedding Table

Rows

Current character

Columns

Votes for next character

Example

| Current Character | Vote for `' '` | Vote for `'A'` | ... | Vote for `'h'` | ... | Vote for `'l'` |
|-------------------|---------------|---------------|-----|---------------|-----|---------------|
| `'T'` | 0.12 | -1.45 | ... | **3.84** | ... | -0.92 |
| `'l'` | -0.55 | 0.22 | ... | -2.10 | ... | **2.95** |

Initially all numbers are random.

---

## Forward Pass

Suppose input is

```text
'T'
```

which has integer ID

```text
3
```

The model simply reads

```text
Row 3
```

Highest value

```text
3.84
```

corresponds to

```text
'h'
```

Therefore

```text
Prediction

'T' → 'h'
```

In our sentence this happens to be correct.

Pure luck.

---

# The Fatal Limitation

Consider the character

```text
'l'
```
(B,T,C)


```text
All
```

The first

```text
l
```

should predict

```text
l
```

The second

```text
l
```

should predict

```text
End of sentence
```

But the Bigram model only sees

```text
'l'
```

It cannot distinguish

```text
A l
```

from

```text
A l l
```

because it has **no memory**.

One row can only contain one prediction distribution.

---

# Phase 3 — The Transformer

We upgrade the model.

Instead of

```python
Embedding(15,15)
```

we use

```python
Embedding(15,32)
```

Now the columns no longer represent vocabulary.

Instead they represent

```text
32 learned semantic coordinates
```

---

# Training Evolution

Initially

Every coordinate is random.

---

## Iteration 1

Input

```text
A l
```

Prediction

Garbage.

Loss

Large.

Cross Entropy penalizes incorrect predictions.

---

## Iteration 500

Backpropagation adjusts the coordinates.

Optimizer gradually discovers

```text
When

A

is followed by

l

the next token often becomes

l
```

---

## Iteration 10,000

Characters that behave similarly become close together inside the 32-dimensional embedding space.

The embeddings now contain meaningful structure rather than random numbers.

---

# How Self-Attention Solves "All"

Suppose the trained model processes

```text
All
```

Every token produces

- Query (Q)
- Key (K)
- Value (V)

---

## Scenario A — First 'l'

Sequence

```text
A l
```

The current

```text
l
```

creates a Query

> "Who should I look back at?"

The previous

```text
A
```

creates a Key

> "I am a capital A."

The model computes

```text
Q_current_l · K_previous_A
```

Since this pattern frequently appeared during training,

the dot product becomes **large**.

Therefore attention strongly focuses on

```text
A
```

The Value vector shifts the representation into

```text
"I am the first l"
```

The LM Head predicts

```text
Next token = l
```

---

## Scenario B — Second 'l'

Sequence

```text
A l l
```

The second

```text
l
```

again creates a Query.

But now it attends to

```
A

and

l
```

The attention matrix recognizes

```text
A → l → l
```

The representation changes into

```text
"I am the final l"
```

Therefore the LM Head predicts

```text
Space

or

End-of-sequence
```

instead of another

```text
l
```

---

# Unified Summary

## 1. Text

```text
"The Winner Takes It All"
```

↓

Encoded into integers.

---

## 2. Embedding Table

Converts every integer into a

```text
32-dimensional learned vector
```

capturing semantic and structural information.

---

## 3. Batch Processing

If

```text
batch_size = 4
```

then four independent fragments of text are processed simultaneously.

All batches share the same embedding tables and parameters.

---

## 4. Self-Attention

Computes

```text
QKᵀ
```

which produces a

```text
(T,T)
```

communication matrix.

Each token dynamically decides

> Which previous tokens matter?

---

## 5. Contextual Understanding

Unlike the Bigram model,

the Transformer can distinguish

```text
first 'l'
```

from

```text
second 'l'
```

because the representation is conditioned on the entire previous context.

---

## 6. Final Prediction

The context-aware representation is passed through the LM Head,

which predicts the probability distribution over every token in the vocabulary,

allowing the model to generate the next character with contextual awareness.


To build a clear mind-map of our journey, let's break down the evolution of the **GPT (Generative Pretrained Transformer)** from scratch. Our ultimate goal was to move from simple character prediction to a model that understands context and generates coherent text by following the architecture described in the *Attention is All You Need* paper.

### 1. The Foundation: Tokenization & Encoder-Decoder (0:09:28)
*   **The Goal:** Translate raw text into numbers a computer can process.
*   **The Process:** We identified every unique character in our *Tiny Shakespeare* dataset (65 total). We created a dictionary (mapping) to turn characters to integers (`encode`) and integers to characters (`decode`).
*   **Question - Why (B, T) instead of 1D?** We need to train in **batches** ($B$) to be efficient on GPUs and we need to feed chunks of text of length ($T$) to give the model "context." If we used 1D, we could only process one character at a time. The 2D structure $(B, T)$ allows the model to see a sequence of $T$ tokens across $B$ independent examples simultaneously.

### 2. The Bigram Language Model (0:22:11)
*   **The Embedding Table (VocabSize, VocabSize):** Initially, this acted as a direct lookup table. Each row is a vector of size 65. If the input `idx` is 24, the embedding table returns the 24th row—a vector of 65 numbers representing the "score" (logit) for every possible next character.
*   **What was stored?** It essentially stored the probability distribution of what character comes next given the current character. It was very naive because it ignored all previous context except the current character.

### 3. The Upgrade to `n_embed = 32` (0:58:26)
*   **Why is this an upgrade?** The initial (65, 65) table was "hard-coded" to output logits directly. By switching to `n_embed = 32`, we decoupled the **representation** of the character from the **prediction** of the character. 
*   The embedding is now an *internal, learned concept* space (latent space) of 32 dimensions. We then use a separate linear layer (`LM Head`) to project those 32 dimensions back to the 65 possible output logits. This allows the model to learn complex relationships between characters that aren't just one-to-one mappings.

### 4. The Transformer Evolution (The Mind-Map)
*   **Stage 1: Self-Attention (0:42:13):** We moved from simple averaging (looking at everything) to **weighted aggregation**. Tokens started "querying" the past to see which ones are interesting (using Dot Product).
*   **Stage 2: Multi-Head Attention (1:21:59):** We ran multiple attention mechanisms in parallel. Each head learns a different relationship (e.g., one head watches for vowels, another for punctuation).
*   **Stage 3: Feed-Forward Networks (1:24:25):** We gave the tokens a place to "think" about the information they gathered during communication.
*   **Stage 4: Residuals & LayerNorm (1:26:48 - 1:32:51):** We added "skip-connections" to help the gradient flow through deep networks and normalized the inputs to stabilize training.

### 5. Clearing the Confusion: The Encoder-Decoder
*   **Important Distinction:** The initial encoder-decoder (for `str` to `int` mapping) is **NOT** the same as the "Encoder-Decoder" mentioned in Transformer papers.
*   **Initial mapping:** Just a basic utility to convert text to numbers.
*   **Transformer Encoder-Decoder:** A high-level architecture. The "Encoder" reads an input sequence (like French), and the "Decoder" generates an output sequence (like English). 
*   **What we built:** We built a **Decoder-only** Transformer. We don't have an encoder because we aren't translating; we are just generating. We keep the "Auto-regressive" property by using a **Triangular Mask** (0:57:00), which ensures the model can't "cheat" by looking at future tokens.

### Summary of Transitions
| Stage | Key Component | Purpose |
| :--- | :--- | :--- |
| **1** | Tokenizer | Mapping text to integers. |
| **2** | Bigram Embedding | Simple direct lookup for next-char prediction. |
| **3** | `n_embed` Logic | Learning latent features in 32D vector space. |
| **4** | Causal Self-Attention | Enabling data-dependent communication between tokens. |
| **5** | Residuals/Norm | Making the deep model stable and trainable. |