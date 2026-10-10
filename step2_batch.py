"""Step 2. Many requests in one step of the model. One step for 16 costs far less than 16 steps for one."""
import time, torch
from model import *

model = get_model()
T = 80                                          # every request has 80 tokens of notes
mem = new_memory(64 * T)
print(f'{"at once":>8} {"one step":>10} {"tokens a second":>16} {"against one":>12}')
base = None
for B in [1, 2, 4, 8, 16, 32, 64]:
    flat = torch.arange(B * T)
    owner = torch.repeat_interleave(torch.arange(B), T)
    tok, pos, new = torch.zeros(B, dtype=torch.long), torch.full((B,), T - 1), torch.arange(B) * T + T - 1
    best = 1e9
    for _ in range(7):                          # the fastest of 7 tries of 30 steps
        t = time.perf_counter()
        for _ in range(30):
            step(model, mem, tok, pos, new, flat, owner)
        best = min(best, (time.perf_counter() - t) / 30)
    base = base or B / best
    print(f'{B:8d} {best * 1000:7.2f} ms {B / best:16,.0f} {B / best / base:11.1f}x')
