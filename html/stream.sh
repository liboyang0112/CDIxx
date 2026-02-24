#!/bin/bash
SOCKET="/tmp/CDIxx_recon_object.sock"
STREAM_DIR="/home/boyang/mainpage/CDIxx/stream"
FPS=20

mkdir -p "$STREAM_DIR"

ffmpeg -f mjpeg -framerate "$FPS" -i "unix://$SOCKET" \
  -c:v hevc_nvenc -b:v 8M -preset p7 -tune ll \
  -g 30 -keyint_min 30 -sc_threshold 0 \
  -hls_time 1 -hls_list_size 5 -hls_flags delete_segments+omit_endlist \
  -hls_segment_filename "$STREAM_DIR/seg_%03d.ts" \
  "$STREAM_DIR/stream.m3u8"
