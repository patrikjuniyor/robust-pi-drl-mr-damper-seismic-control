#!/bin/bash
# usage: bash run_queue.sh [iterations]   (default 400; finished runs are skipped, interrupted runs resume)
cd "$(dirname "$0")"
export ITERS=${1:-400}
mkdir -p ../runs/logs
cat queue.txt | xargs -P 2 -L 1 bash -c 'n=$(echo "$@" | sed "s/.*--name \\([^ ]*\\).*/\\1/"); if [ -f ../runs/$n.done ]; then echo skip $n; else python3 ppo.py "$@" --iters $ITERS >> ../runs/logs/$n.log 2>&1; echo finished $n; fi' _
echo ALL_DONE
