import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
#-------------------

# set seed
torch.manual_seed(0)

# load data
path = 'code/pytorch/birds/'
input = ['shakespeare.txt', 'DonQuixote.txt', 'ExemplaryNovels.txt']
with open(path + input[0], 'r', encoding='utf-8') as f:
    text = f.read()

# unique characters
vocab = sorted(list(set(text)))
vocab_length = len(vocab)

# character to index and index to character
ctoi = {c:i for i,c in enumerate(vocab)}
itoc = {i:c for i,c in enumerate(vocab)}
encode = lambda s: [ctoi[c] for c in s]
decode = lambda l: ''.join([itoc[i] for i in l])

# split data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(data.size(0) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# loss
@torch.no_grad()
def estimate_loss(model):
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[i] = loss
        out[split] = losses.mean().item()
    return out

class Head(nn.Module):
    """Self-attention head."""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        B, T, C = x.size()
        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)
        attn = (Q @ K.transpose(-2, -1)) / C**0.5
        attn = attn.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        x = attn @ V
        return x
    
class MultiHeadAttention(nn.Module):
    """Multi-head attention."""
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        # self.linear = nn.Linear(n_heads*head_size, n_embd)
        
    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        # x = self.linear(x)
        return x

class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_length, n_embd)
        self.positional_embedding = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHeadAttention(4, n_embd//4)
        self.lm_head = nn.Linear(n_embd, vocab_length)
        
    def forward(self, x, targets=None):
        B, T = x.size()
        tok_embd = self.token_embedding(x)
        pos_embd = self.positional_embedding(torch.arange(T, device=x.device))
        x = tok_embd + pos_embd
        x = self.sa_heads(x)
        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_length), targets.view(-1))
            return logits, loss
        return logits, None
    
    def generate(self, idx, max_new_tokens):
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _ = self(idx[:, -block_size:])
                logits = logits[:,-1,:]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, idx_next], dim=1)
        return idx

#-------------------

# model
model = LanguageModel()
m = model.to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# training
for i in range(max_iters):
    # sample batch
    x, y = get_batch('train')
    # forward pass
    logits, loss = model(x, y)
    optimizer.zero_grad()
    # backward pass
    loss.backward()
    # update weights
    optimizer.step()
    # print loss
    if i % eval_interval == 0:
        losses = estimate_loss(model)
        print(f'iteration {i}, train loss: {losses["train"]:.2f}, val loss: {losses["val"]:.2f}')

# generate text
context = torch.tensor([[ctoi['\n']]], dtype=torch.long, device=device)
print(decode(m.generate(context, 500)[0].tolist()))