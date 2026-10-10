"""The people who ask: 1,000 requests. Each has a prompt from the plays and a length for its answer."""
import torch
from model import HELD, MAXLEN, decode

LINES = [i + 1 for i, c in enumerate(HELD.tolist()[:-MAXLEN - 2]) if c == 0]      # where a line of the play starts


class Request:
    def __init__(self, id, prompt, want, u):
        self.id = id                # the order of arrival
        self.prompt = prompt        # tokens
        self.want = want            # how many tokens the answer will have. The engine must not look at this.
        self.u = u                  # the dice for the answer, so every engine writes the same answer
        self.twin = None            # an earlier request with the same prompt
        self.reset()

    def reset(self):
        self.out = []               # the answer so far
        self.n = 0                  # how many tokens have notes in the memory
        self.addr = None            # where those notes lie
        self.table = []
        self.done = self.done_at = None
        self.kicked = 0
        return self

    def text(self):
        return decode(self.out)


def skewed(n, mean, sigma, gen):
    """Many short ones and a few long ones."""
    z = torch.randn(n, generator=gen)
    return (mean * torch.exp(sigma * z - sigma * sigma / 2)).round().long()


def make(n=1000, seed=0, prompt=40, answer=80, answers=1):
    """n requests. answers=4: every prompt is asked four times in a row, each time with other dice."""
    gen = torch.Generator().manual_seed(seed)
    p = skewed(n, prompt, 0.7, gen).clamp(8, 200)
    a = torch.minimum(skewed(n, answer, 0.9, gen).clamp(8, MAXLEN), MAXLEN - p)
    at = torch.randint(len(LINES), (n,), generator=gen)
    reqs = []
    for i in range(n):
        s = LINES[int(at[i])]
        for k in range(answers):
            r = Request(len(reqs), HELD[s:s + int(p[i])].clone(), int(a[i]), torch.rand(int(a[i]), generator=gen))
            r.twin = reqs[-k] if k else None
            reqs.append(r)
    return reqs


if __name__ == '__main__':
    R = make()
    P = torch.tensor([len(r.prompt) for r in R]).float()
    A = torch.tensor([r.want for r in R]).float()
    print(f'{len(R)} requests')
    print(f'prompt: mean {P.mean():.1f} tokens, shortest {int(P.min())}, longest {int(P.max())}')
    print(f'answer: mean {A.mean():.1f} tokens, shortest {int(A.min())}, longest {int(A.max())}')
    print(f'prompt and answer together: mean {(P + A).mean():.1f}, longest {int((P + A).max())}, limit {MAXLEN}')
    print(f'answers under 100 tokens: {int((A < 100).sum())} of {len(R)}')
    print('first prompt:', repr(decode(R[0].prompt)))
