"""The serving loop: who gets in, who gets memory, one step of the model for everyone inside."""
import os, sys, time
from collections import deque
import torch
from model import new_memory, read_prompt, step, pick, get_model, MAXLEN, FILES
from memory import Reserved, Paged
from workload import make

SLOTS = 16 * MAXLEN         # the memory for notes: 8,192 places, room for 16 of the longest texts


def serve(model, reqs, alloc, seats=None, wait_for_all=False, spare=1, share=False, watch=None):
    """Serve every request, first come first served. Returns the numbers of the run.

    alloc:        who hands out the memory (memory.Reserved or memory.Paged)
    seats:        the most requests inside at once. None: as many as the memory takes.
    wait_for_all: True is the old way. Nobody new gets in until everyone inside has finished.
    spare:        free blocks to keep for every request that can still grow, before a new one gets in (only for Paged).
    share:        requests with the same prompt (r.twin) use the same blocks for it (only for Paged).
    watch:        a function that is called after every step, to look at the memory.
    """
    mem = new_memory(alloc.slots)
    block = getattr(alloc, 'block', MAXLEN)
    waiting, inside = deque(r.reset() for r in reqs), []
    log, kicks, reread, steps, reads = [], 0, 0, 0, 0
    t_read = t_step = 0.0
    t0 = time.perf_counter()
    while waiting or inside:
        # 1. let waiting requests in
        if not (wait_for_all and inside):
            while waiting and (seats is None or len(inside) < seats):
                r = waiting[0]
                tokens = torch.cat([r.prompt, torch.tensor(r.out[:-1], dtype=torch.long)])   # after a kick: the answer so far too
                growing = sum(alloc.room(x) < MAXLEN for x in inside) + (len(tokens) + block <= MAXLEN)
                keep = int(spare * growing)     # the admission rule: this many blocks must stay free
                twin = r.twin if share and not r.out else None
                if twin is not None and twin in inside and twin.n == len(twin.prompt):
                    if len(alloc.free) < keep:  # its twin has just read the same prompt: use the same blocks
                        break
                    alloc.fork(twin, r)
                    r.scores = twin.scores
                else:
                    if not alloc.admit(r, len(tokens), keep):
                        break
                    t = time.perf_counter()
                    r.scores = read_prompt(model, mem, tokens, r.addr[:len(tokens)])
                    t_read += time.perf_counter() - t
                    r.n = len(tokens)
                    reads += 1
                waiting.popleft()
                if r.out:
                    reread += len(tokens)
                else:
                    r.out.append(int(pick(r.scores, r.u[0])))
                inside.append(r)
        assert inside, 'the memory is too small for the next request'

        # 2. a place for everyone's next notes. If the memory is full, the latest arrival goes back to the queue.
        i = 0
        while i < len(inside):
            r = inside[i]
            r.where = alloc.place(r, mem)
            if r.where is None:
                out = inside.pop()              # inside is in order of arrival, so the last one came latest
                alloc.release(out)
                out.n, out.kicked = 0, out.kicked + 1
                waiting.appendleft(out)
                kicks += 1
                continue
            i += 1
        assert inside, 'the memory is too small for one request'

        # 3. one step of the model: one new token for everyone inside
        B = len(inside)
        tok = torch.tensor([r.out[-1] for r in inside])
        pos = torch.tensor([r.n for r in inside])
        new = torch.tensor([r.where for r in inside])
        for r in inside:
            r.n += 1
        flat = torch.cat([r.addr[:r.n] for r in inside])
        owner = torch.repeat_interleave(torch.arange(B), pos + 1)
        t = time.perf_counter()
        scores = step(model, mem, tok, pos, new, flat, owner)
        nxt = pick(scores, torch.stack([r.u[len(r.out)] for r in inside]))
        t_step += time.perf_counter() - t
        for r, x in zip(inside, nxt.tolist()):
            r.out.append(x)
        steps += 1

        # what the memory looks like right now
        holding = sum(r.n for r in inside)
        later = sum(min(len(r.prompt) + r.want - 1, alloc.room(r)) - r.n for r in inside)
        log.append((B, holding, later, alloc.used(), len(waiting), sum(alloc.room(r) for r in inside)))
        if watch:
            watch(steps, inside, waiting, alloc)

        # 4. whoever has finished leaves and gives the memory back
        still = []
        for r in inside:
            if len(r.out) == r.want:
                r.done, r.done_at = steps, time.perf_counter() - t0
                alloc.release(r)
            else:
                still.append(r)
        inside = still

    secs = time.perf_counter() - t0
    tokens = sum(r.want for r in reqs)
    full = [x for x in log if x[4] > 0] or log          # the steps with people still waiting outside
    S = alloc.slots
    avg = lambda k: sum(x[k] for x in full) / len(full)
    return dict(
        steps=steps, secs=secs, tokens=tokens, per_sec=tokens / secs, per_step=tokens / steps,
        inside=avg(0), most_inside=max(x[0] for x in log),
        holding=100 * avg(1) / S,                       # % of the memory that holds notes
        later=100 * avg(2) / S,                         # % reserved and used later
        never=100 * (avg(3) - avg(1) - avg(2)) / S,     # % reserved and never used
        free=100 * (S - avg(3)) / S,                    # % not handed out
        wait=sum(r.done_at for r in reqs) / len(reqs), wait_steps=sum(r.done for r in reqs) / len(reqs),
        kicks=kicks, reread=reread, reads=reads, t_read=t_read, t_step=t_step, step_ms=1000 * t_step / steps,
        shared=100 * (1 - sum(x[3] for x in log) / sum(x[5] for x in log)),    # % of blocks saved by sharing
        log=log, answers=[r.text() for r in reqs],
    )


def line(name, s):
    return (f'{name:<26} {s["steps"]:6d} steps  {s["secs"]:6.1f} s  {s["per_sec"]:7.0f} tokens/s  '
            f'inside {s["inside"]:5.1f}  memory with notes {s["holding"]:5.1f} %  wait {s["wait"]:5.1f} s')


WAYS = {
    'one at a time':   lambda: dict(alloc=Reserved(SLOTS), seats=1),
    'fixed groups':    lambda: dict(alloc=Reserved(SLOTS), wait_for_all=True),
    'rows of 512':     lambda: dict(alloc=Reserved(SLOTS)),
    'rows, guessed':   lambda: dict(alloc=Reserved(SLOTS, 'pow2')),
    'rows, exact':     lambda: dict(alloc=Reserved(SLOTS, 'exact')),
    'pages':           lambda: dict(alloc=Paged(SLOTS, 16)),
    'pages, no spare': lambda: dict(alloc=Paged(SLOTS, 16), spare=0),
}


def run(name, reqs=None, tag='', **how):
    """Serve the 1,000 requests one way and return the numbers. Saved in files/, so every step shows the same run.

    Delete files/ or pass --fresh to measure again.
    """
    path = os.path.join(FILES, 'run-' + (name + tag).replace(' ', '-').replace(',', '') + '.pt')
    if os.path.exists(path) and '--fresh' not in sys.argv:
        return torch.load(path)
    how = how or WAYS[name]()
    s = serve(get_model(), reqs or make(), **how)
    s['torn'] = getattr(how['alloc'], 'torn', 0)
    s['copies'] = getattr(how['alloc'], 'copies', 0)
    torch.save(s, path)
    return s
