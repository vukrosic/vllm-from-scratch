"""Step 3. 1,000 people ask at once. Who gets in, and when? Three ways, the same memory for all three."""
from engine import run, line, SLOTS
from model import MAXLEN

print(f'memory for notes: {SLOTS:,} places. A row of {MAXLEN} for everyone inside: {SLOTS // MAXLEN} seats.')
a = run('one at a time')
b = run('fixed groups')
c = run('rows of 512')
print(line('one at a time', a))
print(line('groups of 16, wait for all', b))
print(line('a free seat is filled', c))
print()
print(f'groups of 16: on average {b["inside"]:.1f} of the 16 seats were writing. The rest had finished and waited.')
print(f'a free seat is filled: {c["inside"]:.1f} of 16.   steps: {a["steps"]:,} -> {b["steps"]:,} -> {c["steps"]:,}')
print(f'average wait for the full answer: {a["wait"]:.1f} s -> {b["wait"]:.1f} s -> {c["wait"]:.1f} s')
print(f'same answers in all three: {a["answers"] == b["answers"] == c["answers"]}')
