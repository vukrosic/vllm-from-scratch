"""Step 7. Pages are handed out late, so the memory can run out in the middle of an answer."""
from engine import run, line

greedy, careful = run('pages, no spare'), run('pages')
print(line('let in whoever fits now', greedy))
print(line('keep 1 spare block each', careful))
print()
print(f'let in whoever fits now: the memory was full {greedy["kicks"]:,} times. Each time the latest arrival was sent back to the queue.')
print(f'  its blocks were given away, and later {greedy["reread"]:,} tokens were read a second time to write the notes again')
print(f'keep 1 spare block each: sent back {careful["kicks"]} times, {careful["reread"]:,} tokens read again')
print(f'same answers: {sum(x == y for x, y in zip(greedy["answers"], careful["answers"]))} of {len(greedy["answers"])}')
print(f'memory that holds notes: {greedy["holding"]:.1f} % against {careful["holding"]:.1f} %, and still {greedy["secs"]:.1f} s against {careful["secs"]:.1f} s')
