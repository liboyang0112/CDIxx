#!/bin/bash

WORKDIR="${1:-}"
if [ -z "$WORKDIR" ] || [ ! -d "$WORKDIR" ]; then
  echo "Usage: $0 <working_directory>"
  exit 1
fi

cd "$WORKDIR"

# Source environment
source "${HOME}/myenv.sh" 

# CRITICAL: Follow exact pattern from your example
TRACKING_FILE=".ptycho_run_${USER}_$(date +%s)"
sh -c "echo \$\$ > '$TRACKING_FILE'; chmod 600 '$TRACKING_FILE'; exec stdbuf -oL -eL ptycho_run ptycho.cfg 2>&1" | \
  while IFS= read -r line || [ -n "$line" ]; do
    echo "[$(date '+%H:%M:%S')] $line"
  done | tee -a ptycho.log &

WRAPPER_PID=$!
echo "Started reconstruction (tracking: $TRACKING_FILE, background PID: $WRAPPER_PID)"
exit 0
