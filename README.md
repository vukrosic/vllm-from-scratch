# vLLM and PagedAttention from scratch

Code for the video **How One Graphics Card Answers Many People**.
A language model writes one token at a time, and for every token it has written it keeps notes (the KV cache) so it never reads that token again. When many people ask at once, the notes of all of them must fit in one fixed memory. The old way reserves one long unbroken row per request, and most of that row stays empty.
PagedAttention (the idea inside vLLM) cuts the memory into small blocks and gives a request one block at a time, from anywhere. These eight small programs build that from nothing: the notes, one step for many requests, the waiting line, rows, blocks, shared blocks, a full memory, and the block size. About 780 lines of Python 3 with torch.

```bash
pip install torch
sh run_all.sh         # the model, eight steps, the exercise and the tests; expected output is in demo-output.txt
```

The first run trains the small model: about 15 minutes on a 16 GB laptop with no graphics card. After that the model is read from `files/`.
Each serving run is saved in `files/run-*.pt`. Delete those files to measure again.
Counts (steps, requests inside, percentages, send-backs) do not change from run to run. Seconds do: they depend on what else the laptop is doing, and a few rows in `demo-output.txt` were measured under load.

| File | Idea | What the run shows |
| --- | --- | --- |
| `model.py` | A small model that keeps notes | A GPT with 4 layers and 874,049 numbers, trained 4,000 steps on Shakespeare. It writes one letter at a time, so one token is one letter. Loss on held-out text: 1.61. |
| `workload.py` | The people who ask | 1,000 requests. Prompts are 8 to 200 tokens (mean 41), answers 8 to 499 (mean 79). 747 answers are under 100 tokens. |
| `memory.py` | Two ways to hand out memory | `Rows`: one unbroken row per request. `Paged`: blocks, a block table per request, and a count of users per block. |
| `engine.py` | The serving loop | Who gets in, who gets memory, one step of the model for everyone inside. The same loop runs every step below. |
| `step1_notes.py` | The notes | 400 new tokens: 95,800 tokens through the model without notes, 439 with them. One token of notes is 8,192 bytes here and 819,200 bytes for the paper's model. |
| `step2_batch.py` | One step for many | A step for 1 request takes 0.79 ms, a step for 16 takes 2.65 ms: 4.8 times more tokens a second. |
| `step3_queue.py` | The waiting line | The same memory, three ways. One at a time: 77,952 steps. Groups of 16: 17,060. A free seat filled right away: 4,938, with 16 requests inside. Notes fill 22.5 % of the memory. |
| `step4_memory.py` | Where the memory goes | Rows of 512: 22.5 % notes, 14.8 % filled later, 62.7 % never used. Rows of the exact length, which nobody can know in advance: 50.6 % notes, and 2,016 times a request did not fit although enough places were free. |
| `step5_pages.py` | Blocks | Blocks of 16 tokens: 62.1 requests inside on average, notes fill 84.0 %, 1,520 steps instead of 4,938. All 1,000 answers are the same as with rows. |
| `step6_share.py` | Shared blocks | 250 prompts, four answers each. Prompts read: 1,083 down to 463. 668 blocks copied before a write. 14.3 % of the blocks saved. |
| `step7_full.py` | A full memory | With no spare block: 1,471 send-backs and 66,544 tokens read again. With one spare block per growing request: 1 send-back, 56 tokens. |
| `step8_blocks.py` | The block size | Blocks of 1, 4, 16, 64, 256, 512 tokens: notes fill 99.6, 96.6, 84.0, 54.7, 22.5, 22.5 %. Send-backs: 676, 121, 1, 0, 0, 0. |
| `exercise.py` | Who gets in | 0, 0.25, 0.5, 1, 2, 4 spare blocks per growing request: 1,471, 122, 22, 1, 0, 0 send-backs and 1,397, 1,421, 1,451, 1,520, 1,691, 2,001 steps. |
| `tests.py` | Checks of the claims | 17 tests. |

## Reading the numbers

- **Fewer steps is not always faster.** With no spare block the run needs 1,397 steps, fewer than the 1,520 of our rule. It also reads 66,544 tokens a second time, and it took 66 s against 39 s.
- **One step gets slower.** With blocks, one step writes for 62 requests instead of 16, and it takes 22 ms instead of 14 ms. The whole run still drops from 78 s to 39 s.
- **Small blocks cost a longer table.** Blocks of 1 token fill 99.6 % of the memory, with a table 16 times longer than for blocks of 16 and 676 send-backs.
- **Sharing is partial.** In step 6 a request shares a prompt only if its twin is still inside and has not written yet. That is why 463 prompts are read, not 250.
- **Two kinds of numbers.** The model is trained with 32-bit numbers and serves with 64-bit ones. That is how the test "same answers with rows and with blocks" can ask for the same letters, 1,000 of 1,000.

## What this is not

This is the idea of the paper "Efficient Memory Management for Large Language Model Serving with PagedAttention" (Kwon and eight co-authors, 2023, arXiv 2309.06180), not its system.

- **The machine.** Everything here runs on a laptop processor. The seconds say little about a graphics card. The paper reports 2 to 4 times more requests served at the same wait; that number is the paper's, not ours.
- **The reading step.** vLLM has its own graphics card code that reads blocks in place. This code gathers the notes of a batch through a list of addresses. The paper measures that its reading step alone is 20 to 26 % slower than reading one row.
- **The rows we compare with.** Our row systems give a request the first free gap that is long enough. The paper compares with its own rebuild of a system named Orca, which is not public.
- **The rule for who gets in.** "Keep one free block for every request that can still grow" is our rule. The paper does not state the rule vLLM uses.
- **The requests.** A token is one letter, the lengths are made up, and all 1,000 requests arrive at the start. Real traffic comes in waves.
- **What is left out.** No parking of blocks in main memory (the paper calls it swapping), no beam search, no serving on several cards.
- **The paper's percentages.** The paper's own figure shows 20.4, 26.8 and 38.2 % of the memory holding notes for three row systems and 96.3 % for vLLM. Ours come from another model and other lengths, so they stand next to those numbers and do not reproduce them.

## Try it

The exercise is one line in `engine.py`:

```python
keep = int(spare * growing)     # the admission rule: this many blocks must stay free
```

Our rule (`spare = 1`) sends a request back 1 time and needs 1,520 steps. Find a rule with fewer steps and few send-backs. The rule may look at the requests that are inside. It may not look at `r.want`, the length of an answer, because a real server does not know it.

`shakespeare.txt` is the public-domain text of Shakespeare's plays.

MIT licence.
