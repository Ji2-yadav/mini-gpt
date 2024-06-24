
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
batch_size = 32
block_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
max_iters = 5000
eval_interval = 100
eval_iters = 200
n_embd = 64
dropout = 0.1
n_head = 4
n_layer = 4
learning_rate = 1e-3
input_data_path = 'data/input.txt'
weights_path = 'weights/wt.pt'

#  Download Tiny Shakespeare Dataset
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open(input_data_path, 'r') as f:
    text = f.read()
data = sorted(list(set(text)))

vocab_size = len(data)      # all characters in data

# tokenize and dekonize functions
stoi = { d:i for i,d in enumerate(data)}
itos = {i:d for i,d in enumerate(data)}

def tokenize(token):
    res = [stoi[t] for t in token]
    return res

def detokenize(token):
    ll = [itos[t] for t in token]
    res = ''.join(ll)
    return res

data = torch.tensor(tokenize(text))

bp = int(0.9*len(data))

# split the data
train_data = data[:bp]
val_data  = data[bp:]

def get_batch(split):
    split_type = train_data if split=='train' else val_data
    idx = torch.randint(len(split_type) - block_size, (batch_size,))
    x = torch.stack([split_type[i:i+block_size] for i in idx])
    y = torch.stack([split_type[i+1:i+block_size+1] for i in idx])
    x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def get_losses():
    global model
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            model = model.to(device)
            X = X.to(device)
            Y = Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # self attention
        B,T,C = x.shape
        k = self.key(x)   # B, T, C
        q = self.query(x) # B, T, C
        v = self.value(x) # B, T, C

        # wei = torch.zeros(T,T)
        wei = q @ k.transpose(-2,-1) * C**-0.5 # B, T, T
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v # B, T, C
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4* n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.attention = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.ln1(self.attention(x))
        x = x + self.ln2(self.ffwd(x))
        return x


class miniLLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_emb(idx) # B, T, C
        pos_emb = self.pos_emb(torch.arange(T, device=device)) #T, C
        x = token_emb + pos_emb # B, T, C
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # B, T, vocab_size
        if(targets is None):
            loss=None

        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_nums):
        for _ in range(max_nums):
            idx_cond = idx[:, -block_size:]
            logits, loss = self.forward(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            pred = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, pred), dim=1)
        return idx

m = miniLLM(vocab_size)
model = m.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

def training(save=False):
    # training
    for iter in range(max_iters):
        # Print loss after each eval_interval
        if iter%eval_interval==0 or iter==max_iters-1:
            curr_loss = get_losses()
            print(f"Iter - {iter} : Train loss - {curr_loss['train']}, Val loss - {curr_loss['val']}")

        # Get batch
        xb, yb = get_batch('train')

        # Evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if save:
            torch.save(model.state_dict(), weights_path)


def inference():
    model.eval()
    context = torch.zeros((1,1), dtype = torch.long, device=device)
    output = model.generate(context, max_nums=1000)
    op = detokenize(output[0].tolist())
    print(op)

inference()
# training()


