"""Step 6. One prompt, four answers. The prompt's notes are kept once, and a shared block is copied only when written."""
from engine import run, line, SLOTS
from memory import Paged
from workload import make

reqs = make(250, answers=4)                     # 250 prompts, each asked 4 times with other dice
apart = run('four answers apart', reqs, alloc=Paged(SLOTS, 16))
share = run('four answers shared', reqs, alloc=Paged(SLOTS, 16), share=True)
print(line('every answer its own notes', apart))
print(line('prompt notes shared', share))
print(f'prompts read: {apart["reads"]} -> {share["reads"]}     blocks copied before a write: {share["copies"]}')
print(f'blocks saved by sharing, average over the run: {share["shared"]:.1f} %')
print(f'inside at once: {apart["inside"]:.1f} -> {share["inside"]:.1f}     steps: {apart["steps"]:,} -> {share["steps"]:,}')
print(f'same answers: {sum(x == y for x, y in zip(apart["answers"], share["answers"]))} of {len(reqs)}')
print(f'four different answers to prompt 0: {len(set(share["answers"][:4]))} different')
print("the paper (Fig. 15): 6.1 % to 9.8 % saved with short prompts, 16.2 % to 30.5 % with long ones, more with beam search")
