#!/bin/bash

WORKDIR="${1:-}"
if [ -z "$WORKDIR" ] || [ ! -d "$WORKDIR" ]; then
  echo "Usage: $0 <working_directory>"
  exit 1
fi

cd "$WORKDIR"

# Source environment
source "${HOME}/myenv.sh"
source selectGPU.sh 2>/dev/null

TRACKING_FILE=".ssptycho_run_${USER}_$(date +%s)"
sh -c "echo \$\$ > '$TRACKING_FILE'; chmod 600 '$TRACKING_FILE'; exec stdbuf -oL -eL ssptycho_run ssptycho.cfg 2>&1" | \
  while IFS= read -r line || [ -n "$line" ]; do
    echo "[$(date '+%H:%M:%S')] $line"
  done | tee -a ssptycho.log &

WRAPPER_PID=$!
echo "Started reconstruction (tracking: $TRACKING_FILE, background PID: $WRAPPER_PID)"
exit 0
