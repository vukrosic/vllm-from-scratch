"""Step 5. Pages. The memory is cut into blocks of 16 places, and a request gets one block at a time, from anywhere."""
import torch
from engine import run, line, SLOTS
from memory import Paged
from workload import make
from model import new_memory

# a block table, by hand
alloc, mem = Paged(SLOTS, 16), new_memory(16)
a, b = make(2)
alloc.admit(a, len(a.prompt)); a.n = len(a.prompt)
alloc.admit(b, len(b.prompt)); b.n = len(b.prompt)
print(f'{SLOTS // 16} blocks of 16 places')
print(f'request A, prompt of {a.n} tokens: blocks {a.table}')
print(f'request B, prompt of {b.n} tokens: blocks {b.table}')
while a.n % 16:
    a.n += 1                                    # A writes until its last block is full
alloc.place(a, mem)
print(f'A fills its last block and needs one more: blocks {a.table}')
j = a.n
print(f"A's token {j} lies in its block number {j // 16}, that is block {a.table[j // 16]}, place {j % 16}: memory place {int(a.addr[j])}")
print(f'at most 15 places of a request are empty. With a row of 512, request A would leave {512 - len(a.prompt) - a.want + 1} empty.')
print()

old, new = run('rows of 512'), run('pages')
print(line('rows of 512', old))
print(line('pages of 16', new))
print(f'same answers: {sum(x == y for x, y in zip(old["answers"], new["answers"]))} of {len(old["answers"])}')
print(f'inside at once: {old["inside"]:.1f} -> {new["inside"]:.1f} (most: {new["most_inside"]})     steps: {old["steps"]:,} -> {new["steps"]:,}')
print(f'one step: {old["step_ms"]:.1f} ms -> {new["step_ms"]:.1f} ms, because more notes are read     seconds: {old["secs"]:.1f} -> {new["secs"]:.1f}')
print(f'memory that holds notes: {old["holding"]:.1f} % -> {new["holding"]:.1f} %    (empty ends of last blocks {new["later"] + new["never"]:.1f} %, free blocks {new["free"]:.1f} %)')
