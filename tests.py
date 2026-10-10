"""Tests. Each one checks a claim the film makes. Run: python tests.py"""
import torch
from model import *
from memory import Reserved, Paged
from workload import make
from engine import serve

model = get_model()
FAILED = []


def check(name, ok, note=''):
    print('ok    ' if ok else 'FAILED', name, note)
    if not ok:
        FAILED.append(name)


# 1. notes in scattered blocks give the same scores as reading the whole text again
text = HELD[5000:5100]
whole = model(text.view(1, -1))[0]
alloc, mem = Paged(64 * 16, 16), new_memory(64 * 16)
r = make(1)[0]
r.prompt = text[:40]
alloc.admit(r, 40)
worst = (read_prompt(model, mem, text[:40], r.addr[:40]) - whole[39]).abs().max().item()
for n in range(40, 100):
    r.n = n
    where = alloc.place(r, mem)
    scores = step(model, mem, text[n:n + 1], torch.tensor([n]), torch.tensor([where]),
                  r.addr[:n + 1], torch.zeros(n + 1, dtype=torch.long))
    worst = max(worst, (scores[0] - whole[n]).abs().max().item())
check('notes in scattered blocks = the whole pass', worst < 1e-9, f'(largest difference of a score: {worst:.1e})')
check('a block table gives the address', all(int(r.addr[j]) == r.table[j // 16] * 16 + j % 16 for j in range(100)))

# 2. a shared block is copied before a write, and the first request keeps its notes
alloc, mem = Paged(64 * 16, 16), new_memory(64 * 16)
a, b = make(2)
a.prompt = b.prompt = text[:40]
alloc.admit(a, 40)
read_prompt(model, mem, text[:40], a.addr[:40])
a.n = 40
before = [m[a.addr[:40]].clone() for m in mem]
alloc.fork(a, b)
check('a fork copies nothing', a.table == b.table and alloc.used() == 3 * 16 and alloc.refs[a.table[0]] == 2)
where = alloc.place(b, mem)
for m in mem:
    m[where] = 7.0
check('a write into a shared block goes to a copy', b.table[2] != a.table[2] and b.table[:2] == a.table[:2] and alloc.copies == 1)
check('the first request keeps its notes', all(torch.equal(m[a.addr[:40]], x) for m, x in zip(mem, before)))
check('the copy holds the old notes', all(torch.equal(m[b.addr[:40]], x) for m, x in zip(mem, before)))
alloc.release(a), alloc.release(b)
check('all blocks come back', len(alloc.free) == alloc.n_blocks and not any(alloc.refs))

# 3. every engine writes the same answers
reqs = make(150)
rows = Reserved(4 * MAXLEN)
base = serve(model, reqs, rows)
check('rows: the holes join back to one', rows.holes == [(0, 4 * MAXLEN)] and not rows.rows)
for name, how in [('one at a time', dict(alloc=Reserved(4 * MAXLEN), seats=1)),
                  ('fixed groups', dict(alloc=Reserved(4 * MAXLEN), wait_for_all=True)),
                  ('rows, exact length', dict(alloc=Reserved(4 * MAXLEN, 'exact'))),
                  ('pages of 16', dict(alloc=Paged(4 * MAXLEN, 16))),
                  ('pages of 4', dict(alloc=Paged(4 * MAXLEN, 4)))]:
    s = serve(model, reqs, **how)
    same = sum(x == y for x, y in zip(base['answers'], s['answers']))
    check(f'same answers: {name}', same == len(reqs), f'({same} of {len(reqs)})')

# 4. a request that is sent back writes the same answer
pages = Paged(4 * MAXLEN, 16)
s = serve(model, reqs, pages, spare=0)
same = sum(x == y for x, y in zip(base['answers'], s['answers']))
check('sent back and read again: same answers', s['kicks'] > 0 and same == len(reqs), f'({s["kicks"]} sent back, {same} of {len(reqs)})')
check('pages: all blocks come back', len(pages.free) == pages.n_blocks and not any(pages.refs))

# 5. shared prompts
reqs = make(40, answers=4)
apart = serve(model, reqs, Paged(4 * MAXLEN, 16))
pages = Paged(4 * MAXLEN, 16)
share = serve(model, reqs, pages, share=True)
same = sum(x == y for x, y in zip(apart['answers'], share['answers']))
check('shared prompt notes: same answers', same == len(reqs) and share['reads'] < apart['reads'], f'({same} of {len(reqs)}, prompts read {apart["reads"]} -> {share["reads"]})')
check('shared: all blocks come back', len(pages.free) == pages.n_blocks and not any(pages.refs))

print()
print('ALL TESTS PASS' if not FAILED else f'{len(FAILED)} FAILED: {FAILED}')
raise SystemExit(len(FAILED))
