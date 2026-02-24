#!/bin/bash

WORKDIR="${1:-}"
if [ -z "$WORKDIR" ] || [ ! -d "$WORKDIR" ]; then
  echo "ERROR: Usage: $0 <working_directory>"
  exit 1
fi

cd "$WORKDIR"

# Find latest tracking file
TRACKING_FILE=$(ls -t .ptycho_run_${USER}_* 2>/dev/null | head -n1)
[ -z "$TRACKING_FILE" ] && { echo "No active reconstruction"; exit 1; }

PID=$(cat "$TRACKING_FILE" 2>/dev/null || echo "")
[[ ! "$PID" =~ ^[0-9]+$ ]] && { rm -f "$TRACKING_FILE"; echo "Invalid PID"; exit 1; }

# Verify it's actually ptycho_run (not a reused PID)
if ! ps -p "$PID" -o comm= 2>/dev/null | grep -q "^ptycho_run$"; then
  rm -f "$TRACKING_FILE"
  echo "PID $PID not a ptycho_run process (may have exited)"
  exit 1
fi

echo "[$(date '+%H:%M:%S')] TERMINATION REQUESTED" >> ptycho.log 2>/dev/null || true
echo "Terminating ptycho_run (PID: $PID)..."

# Graceful kill
cleanup() {
  rm -f "$TRACKING_FILE"
  local d="recon_object"; [ -d "$d" ] && {
    rm -f "$d"/stream.m3u8 "$d"/stream*.ts
    echo "#EXTM3U" > "$d/stream.m3u8" 2>/dev/null || true
  }
}
kill -TERM "$PID" 2>/dev/null || true
for i in {1..10}; do
  ps -p "$PID" > /dev/null 2>&1 || { cleanup; echo "SUCCESS"; exit 0; }
  sleep 0.5
done

# Force kill
kill -9 "$PID" 2>/dev/null || true
sleep 1
ps -p "$PID" > /dev/null 2>&1 && { echo "FAILED to kill $PID"; exit 1; }

cleanup
echo "SUCCESS (forced)"

