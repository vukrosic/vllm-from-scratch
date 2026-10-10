"""A small GPT that writes one character at a time, and the notes (the KV cache) it keeps while it writes.

A token here is one character. Every step file imports this.
"""
import os, math, json, time
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.path.join(HERE, 'files')

# ---------------------------------------------------------------- the text
TEXT = open(os.path.join(HERE, 'shakespeare.txt')).read()
CHARS = sorted(set(TEXT))                       # every different character, 65 of them
V = len(CHARS)
STOI = {c: i for i, c in enumerate(CHARS)}
DATA = torch.tensor([STOI[c] for c in TEXT])
CUT = int(len(DATA) * 0.9)
TRAIN, HELD = DATA[:CUT], DATA[CUT:]            # the model never trains on HELD


def decode(ids):
    return ''.join(CHARS[int(i)] for i in ids)


# ---------------------------------------------------------------- the model
MAXLEN = 512                                    # the longest text the model can hold: prompt and answer together
E, HEADS, LAYERS = 128, 4, 4
D = E // HEADS
NOTE_BYTES = LAYERS * 2 * E * 8                 # one token's notes: a key and a value in every layer, 8 bytes a number


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(E), nn.LayerNorm(E)
        self.query = nn.Linear(E, E, bias=False)    # what am I looking for?
        self.key = nn.Linear(E, E, bias=False)      # what do I hold?
        self.value = nn.Linear(E, E, bias=False)    # what do I hand over?
        self.out = nn.Linear(E, E)
        self.think = nn.Sequential(nn.Linear(E, 4 * E), nn.GELU(), nn.Linear(4 * E, E))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.letter = nn.Embedding(V, E)            # a list of numbers for each character
        self.place = nn.Embedding(MAXLEN, E)        # a list of numbers for each position
        self.layers = nn.ModuleList(Layer() for _ in range(LAYERS))
        self.norm = nn.LayerNorm(E)
        self.head = nn.Linear(E, V)                 # back to 65 scores

    def forward(self, idx, keep=None):
        """The whole text in one pass. Every position looks at itself and the earlier ones.

        keep: an empty list. It gets every layer's keys and values, shaped [B, T, 2, HEADS, D].
        """
        B, T = idx.shape
        x = self.letter(idx) + self.place(torch.arange(T))
        future = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        for lay in self.layers:
            h = lay.norm1(x)
            q = lay.query(h).view(B, T, HEADS, D)
            k = lay.key(h).view(B, T, HEADS, D)
            v = lay.value(h).view(B, T, HEADS, D)
            if keep is not None:
                keep.append(torch.stack([k, v], 2))
            match = q.transpose(1, 2) @ k.permute(0, 2, 3, 1) / math.sqrt(D)       # B, HEADS, T, T
            share = match.masked_fill(future, float('-inf')).softmax(-1)
            x = x + lay.out((share @ v.transpose(1, 2)).transpose(1, 2).reshape(B, T, E))
            x = x + lay.think(lay.norm2(x))
        return self.head(self.norm(x))


# ---------------------------------------------------------------- the notes
def new_memory(slots):
    """The memory for notes: `slots` places, each holds one token's key and value for every layer."""
    return [torch.zeros(slots, 2, HEADS, D, dtype=torch.float64) for _ in range(LAYERS)]


@torch.no_grad()
def read_prompt(model, mem, tokens, where):
    """Read a whole prompt in one pass and write its notes into the places `where`. Returns the scores for the next token."""
    keep = []
    scores = model(tokens.view(1, -1), keep)
    for l in range(LAYERS):
        mem[l][where] = keep[l][0]
    return scores[0, -1]


@torch.no_grad()
def step(model, mem, tok, pos, new, flat, owner):
    """One new token for every request in the batch.

    tok, pos, new: for each of the B requests its newest token, that token's position and the place for its notes.
    flat:  the places of all notes of all requests, one after another (the new places included).
    owner: for each entry of flat, the request it belongs to.
    A request reads only its own notes, wherever they lie in the memory.
    """
    B = len(tok)
    x = model.letter(tok) + model.place(pos)
    own = owner.view(-1, 1).expand(-1, HEADS)
    for l, lay in enumerate(model.layers):
        h = lay.norm1(x)
        q = lay.query(h).view(B, HEADS, D)
        mem[l][new, 0] = lay.key(h).view(B, HEADS, D)          # write the new token's notes
        mem[l][new, 1] = lay.value(h).view(B, HEADS, D)
        kv = mem[l][flat]                                       # collect every request's notes: N, 2, HEADS, D
        match = (kv[:, 0] * q[owner]).sum(-1) / math.sqrt(D)    # N, HEADS
        top = q.new_zeros(B, HEADS).scatter_reduce(0, own, match, 'amax', include_self=False)
        e = (match - top[owner]).exp()
        share = e / q.new_zeros(B, HEADS).index_add_(0, owner, e)[owner]     # adds up to 1 inside each request
        got = q.new_zeros(B, HEADS, D).index_add_(0, owner, share.unsqueeze(-1) * kv[:, 1])
        x = x + lay.out(got.view(B, E))
        x = x + lay.think(lay.norm2(x))
    return model.head(model.norm(x))


HEAT = 0.8


def pick(scores, u):
    """Roll a loaded die. u is a number between 0 and 1 that the request brought along, so every engine rolls the same."""
    p = (scores / HEAT).softmax(-1).cumsum(-1)
    return (p < u.unsqueeze(-1) * p[..., -1:]).sum(-1).clamp(max=V - 1)


# ---------------------------------------------------------------- training
STEPS = 4000


def get_model(steps=STEPS, size=12, seed=0):
    """Train once (about ten minutes on a laptop), save in files/, reuse on later runs.

    The model trains with 32-bit numbers and serves with 64-bit ones. With 32 bits, the rounding depends on how many
    requests share a step, and about one answer in 200 changes at one character. With 64 bits every engine here
    writes the same text, so the engines can be compared answer by answer.
    """
    os.makedirs(FILES, exist_ok=True)
    path = os.path.join(FILES, f'gpt-{steps}.pt')
    torch.manual_seed(seed)
    model = GPT()
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        return model.double().eval()
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    t0 = time.time()
    for s in range(steps):
        at = torch.randint(len(TRAIN) - MAXLEN - 1, (size,), generator=gen)
        x = torch.stack([TRAIN[i:i + MAXLEN] for i in at])
        y = torch.stack([TRAIN[i + 1:i + MAXLEN + 1] for i in at])
        loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % 250 == 0:
            print(f'  training step {s:4d} of {steps}   loss {loss.item():.3f}   {time.time() - t0:5.0f} s', flush=True)
    torch.save(model.state_dict(), path)
    with torch.no_grad():
        at = torch.linspace(0, len(HELD) - MAXLEN - 2, 40).long()
        x = torch.stack([HELD[i:i + MAXLEN] for i in at])
        y = torch.stack([HELD[i + 1:i + MAXLEN + 1] for i in at])
        held = F.cross_entropy(model(x).view(-1, V), y.view(-1)).item()
    json.dump(dict(steps=steps, held_loss=held, seconds=time.time() - t0), open(path[:-3] + '.json', 'w'))
    return model.double().eval()


if __name__ == '__main__':
    m = get_model()
    print(sum(p.numel() for p in m.parameters()), 'numbers in the model')
    print(json.load(open(os.path.join(FILES, f'gpt-{STEPS}.json'))))
