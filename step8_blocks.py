"""Step 8. How big is a block? One place, or the whole row of 512?"""
from engine import run, SLOTS
from memory import Paged

print(f'{"block":>6} {"blocks":>7} {"inside":>7} {"notes":>7} {"sent back":>10} {"steps":>6} {"one step":>9} {"seconds":>8}')
for size in [1, 4, 16, 64, 256, 512]:
    s = run('pages') if size == 16 else run(f'pages of {size}', alloc=Paged(SLOTS, size))
    print(f'{size:6d} {SLOTS // size:7d} {s["inside"]:7.1f} {s["holding"]:6.1f}% {s["kicks"]:10d} {s["steps"]:6d} {s["step_ms"]:6.1f} ms {s["secs"]:8.1f}')
print('a block of 512 is the reserved row again. vLLM uses 16.')
