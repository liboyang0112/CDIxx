#include <assert.h>
#include <fmt/core.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fmt/core.h"
#include "memManager.hpp"
#include "streamer.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>  // For correct RGB→YUV conversion
}

static const AVCodec *codec = nullptr;

struct video {
  AVCodecContext *c;
  AVFormatContext *format;
  AVFrame *frame;
  AVPacket *pkt;
  AVStream *stream;
  struct SwsContext *sws_ctx;  // RGB24 → YUV420P converter
  void *av1_streamer;
  int width;   // columns
  int height;  // rows
};

static void* encode(void* arg) {
  struct video* v = (struct video*)arg;

  int ret = avcodec_send_frame(v->c, v->frame);
  if (ret < 0) {
    fmt::println(stderr, "Error sending frame: {}", av_err2str(ret));
    return nullptr;
  }

  while (1) {
    ret = avcodec_receive_packet(v->c, v->pkt);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      break;
    }
    if (ret < 0) {
      fmt::println(stderr, "Encoding error: {}", av_err2str(ret));
      break;
    }

    av_packet_rescale_ts(v->pkt, v->c->time_base, v->stream->time_base);
    v->pkt->stream_index = v->stream->index;

    ret = av_write_frame(v->format, v->pkt);
    if (ret < 0) {
      fmt::println(stderr, "Write error: {}", av_err2str(ret));
    }

    if (v->av1_streamer) {
      streamer_send_packet(v->av1_streamer, v->pkt);
    }
    av_packet_unref(v->pkt);
  }

  return nullptr;
}

void* createVideo(const char* filename, int row, int col, int fps, const char* stream_sock) {
  struct video* thisvid = (struct video*)ccmemMngr.borrowCache(sizeof(struct video));
  memset(thisvid, 0, sizeof(struct video));
  if (col <= 0 || row <= 0) {
    fmt::println(stderr, "Invalid dimensions: {}x{}", col, row);
    exit(1);
  }
  if (col % 2 != 0 || row % 2 != 0) {
    fmt::println(stderr, "NVENC requires even dimensions for YUV420P: {}x{}", col, row);
    exit(1);
  }
  thisvid->height = row;  // rows = height
  thisvid->width = col;   // cols = width

  if (stream_sock) {
    thisvid->av1_streamer = streamer_create(stream_sock, col, row, fps);
  }

  thisvid->pkt = av_packet_alloc();
  int ret = avformat_alloc_output_context2(&thisvid->format, nullptr, "mp4", filename);
  if (!thisvid->format) {
    fmt::println(stderr, "Could not create output context");
    exit(1);
  }

  // Use av1_nvenc (supports yuv420p directly)
  if (!codec) {
    if(stream_sock)
      codec = avcodec_find_encoder_by_name("av1_nvenc");
    else {
      codec = avcodec_find_encoder_by_name("hevc_nvenc");
    }
  }
  if (!codec) {
    fmt::println(stderr, "Encoder not found. Ensure FFmpeg is built with NVENC support.");
    exit(1);
  }

  thisvid->stream = avformat_new_stream(thisvid->format, codec);
  if (!thisvid->stream) {
    fmt::println(stderr, "Failed to create stream");
    exit(1);
  }

  AVCodecContext *c = thisvid->c = avcodec_alloc_context3(codec);
  c->bit_rate = 8000000;
  c->width = col;    // width = columns
  c->height = row;   // height = rows
  c->framerate = (AVRational){fps, 1};  // Set framerate FIRST
  c->time_base = av_inv_q(c->framerate); // Derive time_base from framerate
  c->pix_fmt = AV_PIX_FMT_YUV420P;  // av1_nvenc supports this natively

  // NVENC tuning options
  av_opt_set(c->priv_data, "preset", "p7", 0);   // p7 = fastest/lowest latency
  av_opt_set(c->priv_data, "tune", "ll", 0);     // low-latency mode
  av_opt_set(c->priv_data, "profile", "main", 0);
  c->codec_tag = 0;  // Let muxer set proper FourCC for MP4

  ret = avcodec_open2(c, codec, nullptr);
  if (ret < 0) {
    fmt::println(stderr, "Could not open av1_nvenc codec: {}", av_err2str(ret));
    exit(1);
  }

  // MUST copy params AFTER avcodec_open2()
  ret = avcodec_parameters_from_context(thisvid->stream->codecpar, c);
  if (ret < 0) {
    fmt::println(stderr, "Failed to copy codec params: {}", av_err2str(ret));
    exit(1);
  }

  if (thisvid->av1_streamer) {
    streamer_set_encoder_context(thisvid->av1_streamer, c);
  }

  if (!(thisvid->format->oformat->flags & AVFMT_NOFILE)) {
    ret = avio_open(&thisvid->format->pb, filename, AVIO_FLAG_WRITE);
    if (ret < 0) {
      fmt::println(stderr, "Could not open output file '{}': {}", filename, av_err2str(ret));
      exit(1);
    }
  }

  ret = avformat_write_header(thisvid->format, nullptr);
  if (ret < 0) {
    fmt::println(stderr, "Failed to write header: {}", av_err2str(ret));
    exit(1);
  }

  // Allocate encoder frame (YUV420P)
  AVFrame *frame = thisvid->frame = av_frame_alloc();
  frame->format = c->pix_fmt;
  frame->width = c->width;
  frame->height = c->height;
  frame->pts = 0;

  ret = av_frame_get_buffer(frame, 0);
  if (ret < 0) {
    fmt::println(stderr, "Could not allocate frame buffer: {}", av_err2str(ret));
    exit(1);
  }

  // Setup swscale for correct RGB24 → YUV420P conversion
  thisvid->sws_ctx = sws_getContext(
    col, row, AV_PIX_FMT_RGB24,
    col, row, AV_PIX_FMT_YUV420P,
    SWS_BILINEAR, nullptr, nullptr, nullptr
  );
  if (!thisvid->sws_ctx) {
    fmt::println(stderr, "Could not initialize swscale context");
    exit(1);
  }

  return thisvid;
}

void flushVideo(void* ptr, void* buffer) {
  struct video* thisvid = (struct video*)ptr;
  // Convert RGB → YUV420P with correct 4:2:0 subsampling
  const uint8_t* buf[1];
  buf[0] = (const uint8_t*)buffer;
  int s[1];
  s[0] = thisvid->width * 3;
  sws_scale(
    thisvid->sws_ctx,
    buf, s,
    0, thisvid->height,
    thisvid->frame->data, thisvid->frame->linesize
  );

  // Encode frame
  thisvid->frame->pts = thisvid->frame->pts >= 0 ? thisvid->frame->pts : 0;
  encode(thisvid);
  thisvid->frame->pts++;
}

void flushVideo_float(void* ptr, void* buffer) {
  struct video* thisvid = (struct video*)ptr;
  float* float_data = (float*)buffer;

  // Convert float [0.0,1.0] → grayscale RGB24
  uint8_t* rgb_buf = (uint8_t*)av_malloc(thisvid->width * thisvid->height * 3);
  for (int i = 0; i < thisvid->width * thisvid->height; i++) {
    float val = float_data[i];
    if (val < 0.0f) val = 0.0f;
    else if (val > 1.0f) val = 1.0f;
    uint8_t byte = (uint8_t)(val * 255.0f);
    rgb_buf[i*3 + 0] = byte;  // R
    rgb_buf[i*3 + 1] = byte;  // G
    rgb_buf[i*3 + 2] = byte;  // B
  }

  flushVideo(ptr, rgb_buf);
  av_free(rgb_buf);
}

void saveVideo(void* ptr) {
  struct video* v = (struct video*)ptr;

  // Flush encoder
  avcodec_send_frame(v->c, nullptr);
  int ret;
  while (1) {
    ret = avcodec_receive_packet(v->c, v->pkt);
    if (ret == AVERROR_EOF || ret == AVERROR(EAGAIN)) break;
    if (ret < 0) {
      fmt::println(stderr, "Flush error: {}", av_err2str(ret));
      break;
    }

    av_packet_rescale_ts(v->pkt, v->c->time_base, v->stream->time_base);
    v->pkt->stream_index = v->stream->index;
    av_write_frame(v->format, v->pkt);
    if (v->av1_streamer) {
      streamer_send_packet(v->av1_streamer, v->pkt);
    }
    av_packet_unref(v->pkt);
  }

  av_write_trailer(v->format);

  // Cleanup
  if (v->av1_streamer) {
    streamer_destroy(v->av1_streamer);
  }
  if (!(v->format->oformat->flags & AVFMT_NOFILE)) {
    avio_closep(&v->format->pb);
  }
  avformat_free_context(v->format);
  avcodec_free_context(&v->c);
  av_frame_free(&v->frame);
  av_packet_free(&v->pkt);
  sws_freeContext(v->sws_ctx);
  ccmemMngr.returnCache(v);
}
