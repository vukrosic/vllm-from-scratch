"""Step 1. The notes. Without them the model rereads the whole text for every new token."""
import time, torch
from model import *
from workload import LINES

model = get_model()
start = LINES[100]
prompt, WANT = HELD[start:start + 40], 400
dice = torch.rand(WANT, generator=torch.Generator().manual_seed(1))
print('prompt:', repr(decode(prompt)))

# a) no notes: the whole text goes through the model again for every new token
t = time.perf_counter()
text, pushed = prompt.clone(), 0
with torch.no_grad():
    for i in range(WANT):
        scores = model(text.view(1, -1))[0, -1]
        pushed += len(text)
        text = torch.cat([text, pick(scores, dice[i]).view(1)])
slow = time.perf_counter() - t
plain = decode(text[40:])

# b) notes: read the prompt once, then only the newest token goes through the model
t = time.perf_counter()
mem = new_memory(MAXLEN)
out = [int(pick(read_prompt(model, mem, prompt, torch.arange(40)), dice[0]))]
n = 40
while len(out) < WANT:
    scores = step(model, mem, torch.tensor([out[-1]]), torch.tensor([n]), torch.tensor([n]),
                  torch.arange(n + 1), torch.zeros(n + 1, dtype=torch.long))
    n += 1
    out.append(int(pick(scores[0], dice[len(out)])))
fast = time.perf_counter() - t

print(f'no notes: {pushed:,} tokens pushed through the model for {WANT} new ones, {slow:.2f} s')
print(f'notes:    {40 + WANT - 1:,} tokens pushed through the model for {WANT} new ones, {fast:.2f} s')
print(f'same text: {plain == decode(out)}    {slow / fast:.1f} times faster')
print('answer starts:', repr(decode(out[:120])))
print()
print(f'one token of notes: {LAYERS} layers x 2 lists (key, value) x {E} numbers x 8 bytes = {NOTE_BYTES:,} bytes')
print(f'the longest text, {MAXLEN} tokens: {MAXLEN * NOTE_BYTES / 1e6:.1f} MB of notes')
print(f'the model itself: {sum(p.numel() for p in model.parameters()):,} numbers, {sum(p.numel() for p in model.parameters()) * 8 / 1e6:.1f} MB')
print("the paper's 13-billion model: 40 layers x 2 x 5,120 numbers x 2 bytes = 819,200 bytes a token, 1.6 GB for 2,048 tokens")
