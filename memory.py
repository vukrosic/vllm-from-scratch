"""Two ways to hand out the memory for notes: one reserved row per request, or small blocks on demand."""
import torch
from model import MAXLEN


class Reserved:
    """One unbroken row per request, reserved when the request arrives.

    rule 'max':   always the longest text the model can hold. The engine does not know how long the answer will be.
    rule 'pow2':  the prompt, plus the answer's length rounded up to the next power of two. Needs a good guess.
    rule 'exact': the prompt plus the true length of the answer. No real engine can know that.
    """

    def __init__(self, slots, rule='max'):
        self.slots, self.rule = slots, rule
        self.holes = [(0, slots)]               # the free stretches: (start, length), from left to right
        self.rows = {}                          # request -> (start, length)
        self.torn = 0                           # times the free memory was enough, but not in one piece

    def row(self, r):
        if self.rule == 'max':
            return MAXLEN
        if self.rule == 'pow2':
            n = 1
            while n < r.want:
                n *= 2
            return min(len(r.prompt) + n, MAXLEN)
        return len(r.prompt) + r.want

    def admit(self, r, need, spare=0):
        n = self.row(r)
        for i, (start, length) in enumerate(self.holes):        # the first hole that is long enough
            if length >= n:
                self.holes[i:i + 1] = [(start + n, length - n)] if length > n else []
                self.rows[r.id] = (start, n)
                r.addr = torch.arange(start, start + n)
                return True
        self.torn += sum(length for _, length in self.holes) >= n
        return False

    def place(self, r, mem):
        return int(r.addr[r.n])                 # always there: it was reserved

    def release(self, r):
        self.holes = sorted(self.holes + [self.rows.pop(r.id)])
        joined = [self.holes[0]]
        for start, length in self.holes[1:]:    # two holes side by side become one
            if joined[-1][0] + joined[-1][1] == start:
                joined[-1] = (joined[-1][0], joined[-1][1] + length)
            else:
                joined.append((start, length))
        self.holes = joined

    def used(self):
        return sum(n for _, n in self.rows.values())

    def room(self, r):
        return self.rows[r.id][1]


class Paged:
    """The memory cut into blocks of `block` places. A request gets one block at a time, from anywhere.

    r.table is the request's block table: its 1st, 2nd, 3rd ... block of notes lies in block number table[0], table[1] ...
    refs counts how many requests use a block. A block with more than one user is copied before anyone writes into it.
    """

    def __init__(self, slots, block=16, seed=0):
        self.slots, self.block = slots, block
        self.n_blocks = slots // block
        order = torch.randperm(self.n_blocks, generator=torch.Generator().manual_seed(seed)).tolist()
        self.free = order                       # shuffled on purpose: nothing below needs neighbours
        self.refs = [0] * self.n_blocks
        self.copies = 0
        self.steps = torch.arange(block)

    def take(self):
        b = self.free.pop()
        self.refs[b] = 1
        return b

    def spread(self, r):
        """The place of every token of the request, read off its block table."""
        r.addr = (torch.tensor(r.table).view(-1, 1) * self.block + self.steps).flatten()

    def admit(self, r, need, spare=0):
        want = -(-need // self.block)           # blocks for what must be stored now, rounded up
        if len(self.free) < want + spare:
            return False
        r.table = [self.take() for _ in range(want)]
        self.spread(r)
        return True

    def place(self, r, mem):
        """The place for the next token's notes. None when no block is free."""
        i = r.n // self.block
        if i == len(r.table):                   # the last block is full: one more block
            if not self.free:
                return None
            r.table.append(self.take())
            self.spread(r)
        elif self.refs[r.table[i]] > 1:         # a shared block: copy it first, then write into the copy
            if not self.free:
                return None
            old, new = r.table[i], self.take()
            self.refs[old] -= 1
            for m in mem:
                m[new * self.block:(new + 1) * self.block] = m[old * self.block:(old + 1) * self.block]
            r.table[i] = new
            self.spread(r)
            self.copies += 1
        return int(r.addr[r.n])

    def fork(self, r, child):
        """A second answer to the same prompt: the same blocks, counted twice. Nothing is copied yet."""
        child.table = list(r.table)
        for b in child.table:
            self.refs[b] += 1
        child.n = r.n
        self.spread(child)

    def release(self, r):
        for b in r.table:
            self.refs[b] -= 1
            if self.refs[b] == 0:
                self.free.append(b)
        r.table = []

    def used(self):
        return (self.n_blocks - len(self.free)) * self.block

    def room(self, r):
        return len(r.table) * self.block
