"""Exercise. When may a new request come in? Find a rule that beats one spare block per request."""
from engine import run, SLOTS
from memory import Paged

print(f'{"spare blocks per request":>25} {"inside":>7} {"notes":>7} {"sent back":>10} {"read again":>11} {"steps":>6} {"seconds":>8}')
for spare in [0, 0.25, 0.5, 1, 2, 4]:
    name = {0: 'pages, no spare', 1: 'pages'}.get(spare)
    s = run(name) if name else run(f'spare {spare}', alloc=Paged(SLOTS, 16), spare=spare)
    print(f'{spare:25} {s["inside"]:7.1f} {s["holding"]:6.1f}% {s["kicks"]:10d} {s["reread"]:11,d} {s["steps"]:6d} {s["secs"]:8.1f}')
print('Your turn: change the line `keep = ...` in engine.py. The rule may look at the requests inside, but not at r.want.')
