import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 12 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed=32
# ------------

torch.manual_seed(2506)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open(r'data\input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # tells pyTorch not to do backpropagation
def estimate_loss(): # average outs the loss for eval_iterations
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    '''This class is responsible for creating the Attention or Communication Mechanism - that lets the tokens communicate with past timesteps'''
    def __init__(self, head_size):
        super().__init__()
        self.key=nn.Linear(n_embed, head_size, bias=False) # The attentione heads are like channels of dimension say (32,16) i.e 32 other tokens define the word and 16 other timesteps contributes to its semantic meaning
        self.query=nn.Linear(n_embed, head_size, bias =False)
        self.value=nn.Linear(n_embed, head_size, bias =False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # buffer is a variable that is needed for use, but is not a model param. 

    def forward(self, x):

        # Extracting tensor shape params
        B,T,C=x.shape
        # mapping key, query and value to the input
        k=self.key(x)   
        q=self.query(x)
        v=self.value(x)

        weights= q @ k.transpose(-2,-1) * C**(-0.5) # Scaled Dot product
        weights=weights.masked_fill(self.tril[:T,:T]==0, float('-inf')) # type: ignore
        weights=F.softmax(weights, dim=-1)

        # Perform weighted aggregation of values
        out=weights @ v
        return out

# Now that we are done with Single-Head Attention - lets implement Multi_head attention - replicas of how many heads we want.
# Each head is responsible for some definite and specific purpose - say head 1 captures relationship of how are vowels related, while say head 2 maps how the punctuation contributes to the next token 
class MultiHead(nn.Module):
    '''Multiple Heads of Self Attention in parallel'''
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # using the function from base class we can create a list of submodules within a list

    def forward(self,x):
        # here we are just concatenating all the separate heads over the last dimention(channel dim) into a single unit
        return torch.cat([h(x) for h in self.heads], dim=-1) # take every single heads from the list of multiheads and for each of them compute k,q,v using x as argument and then concat all of them back


# Now by far whatever improvements we did, still we compute logits directly after the attention level. Though the loss decreased from 2.4 in ur first version to 2.2 in our latest- but still the logits were calculated directly, therefore the architecture don't get enough time to think upon what they found from other tokens
# Thus we need a linear single layer of network with relu non-linearity for propagation of the logits - with feedforward the loss further decreased from 2.28 to 2.23 
class FeedForward(nn.Module):
    '''a simnple linear layer followed by non-linearity'''

    def __init__(self, n_embed):
        super().__init__()
        self.ffnet=nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.ffnet(x) # executing each tokens on the feed forward neural net
        


# Now by far whatever we did in the Decoder architecture (Pos encoding, sh, mh, scaled dot prod, feedforwd) can be trated as block of calculation & computation. Now to improve performance we can repeat these block k times 
class Block(nn.Module):
    '''Transformer Block: Communication (Attention) followed by Computation'''

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size=n_embed//n_head
        self.attention=MultiHead(n_head, head_size)
        self.ffwd=FeedForward(n_embed)

    def forward(self, x):
        x=self.attention(x)
        x=self.ffwd(x)
        return x
# Block of Transformer creates Abstraction by judt defining the predefined methods and objects for Communication and Computation repeatedly


# Gradually improved architeture of a naive Bigram model implemnted with M-H attention
class BigramLanguageModel(nn.Module):

    # Configurations 
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # We besides creating the embedding table to store the embedding vectors for every token also create a positional_embeddding table for positional encoding
        self.positional_embedding_table=nn.Embedding(block_size,n_embed)
        # self.sa = Head(n_embed) # creating the self attention head - commented out since we have improved to Multi_head attention
        # Now that we have evolved to Multiple heads, lets comment out Self-head and implement Multi-head attention
        # self.mha = MultiHead(num_heads=4, head_size=n_embed//4) # we are dividing the entire num of embedd. into 4 sets for 4 heads of uniform size. (4 * 8Dim = 32Dim embedd vector)
        # self.feedfwd = FeedForward(n_embed)

        # We now comment out all the isolated unit level operations, since we have created a Block containing the same operations and we can just Sequentially repeat them
        self.blocks = nn.Sequential(
            Block(n_embed=32, n_head=4),
            Block(n_embed=32, n_head=4),
            Block(n_embed=32, n_head=4),
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)
                

    def forward(self, idx, targets=None):
        B,T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_embd = self.token_embedding_table(idx) # (B,T,C)
        pos_embd = self.positional_embedding_table(torch.arange(T, device=device)) # creating pos embd vectors from 0 to range T-1 of shape (T,C)
        x=tok_embd + pos_embd

        # x = self.mha(x)
        # s=self.feedfwd(x) # (B,T,C)

        x=self.blocks(x)
        logits=self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):

            # idx can go out of range that's not present in the scope of Embeddings table, thus we need to filter it
            idx_cond=idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond) # forward(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

    
model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))