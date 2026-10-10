#!/bin/sh
# Run every step in order. The first run trains the small model (about ten minutes on a laptop).
# The serving runs are saved in files/. Delete files/run-*.pt to measure them again.
set -e
cd "$(dirname "$0")"
PY=${PY:-python3}
for f in model.py workload.py step1_notes.py step2_batch.py step3_queue.py step4_memory.py step5_pages.py step6_share.py step7_full.py step8_blocks.py exercise.py tests.py; do
  echo
  echo "=== $f ==="
  $PY $f
done
