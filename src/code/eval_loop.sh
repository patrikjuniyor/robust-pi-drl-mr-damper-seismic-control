#!/bin/bash
# evaluate every finished policy that has not been evaluated yet; repeat until all queued runs are done
cd "$(dirname "$0")"
total=$(grep -c . queue.txt)
while true; do
  n=$(ls ../runs/*.done 2>/dev/null | wc -l)
  python3 evaluate.py --only V_ >> ../results/eval_extra_seeds.log 2>&1
  if [ "$n" -ge "$total" ]; then break; fi
  sleep 240
done
echo EVAL_LOOP_DONE
