"""Step 4. Where the memory goes when every request gets one unbroken row."""
from engine import run, SLOTS

print(f'{"row length":<34} {"inside":>6} {"notes":>7} {"for later":>10} {"never used":>11} {"free":>6} {"steps":>6} {"seconds":>8}')
for name, label in [('rows of 512', 'always 512 (length unknown)'),
                    ('rows, guessed', 'prompt + answer rounded up to 2^n'),
                    ('rows, exact', 'prompt + true answer (cheating)')]:
    s = run(name)
    print(f'{label:<34} {s["inside"]:6.1f} {s["holding"]:6.1f}% {s["later"]:9.1f}% {s["never"]:10.1f}% {s["free"]:5.1f}% {s["steps"]:6d} {s["secs"]:8.1f}')
    if name != 'rows of 512':
        print(f'{"":<34} {s["torn"]} times the next request did not fit although enough places were free: no hole was long enough')
print()
print('notes:      places that hold the notes of a token')
print('for later:  reserved, and a token will land there before the request ends')
print('never used: reserved, and the request ends before it gets there')
print('free:       not handed out, most of it in holes between rows')
print("the paper (Fig. 2), share of the memory that holds notes: 20.4 %, 26.8 %, 38.2 % for the same three, 96.3 % for vLLM")
