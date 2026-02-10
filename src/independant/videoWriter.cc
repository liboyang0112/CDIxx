#include <assert.h>
#include <fmt/core.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "fmt/core.h"
#include "memManager.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

static const AVCodec *codec = nullptr;

struct video {
  AVCodecContext *c;
  AVFormatContext *format;
  AVFrame *frame;
  AVPacket *pkt;
  AVStream *stream;
  struct SwsContext *sws_ctx;
  int width;
  int height;
  bool is_hls_mode;
  int64_t next_pts;
};

static void* encode(void* arg) {
  struct video* v = (struct video*)arg;

  v->frame->duration = 1;
  
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

    if (v->pkt->duration == 0) {
        v->pkt->duration = 1;
    }

    av_packet_rescale_ts(v->pkt, v->c->time_base, v->stream->time_base);
    v->pkt->stream_index = v->stream->index;

    ret = av_write_frame(v->format, v->pkt);
    if (ret < 0) {
      fmt::println(stderr, "Write error: {}", av_err2str(ret));
    }
    av_packet_unref(v->pkt);
  }

  return nullptr;
}

// Helper: create directory recursively
static int mkdir_p(const char* path) {
    char tmp[1024];
    char *p = NULL;
    size_t len;

    snprintf(tmp, sizeof(tmp), "%s", path);
    len = strlen(tmp);
    if (len > 0 && tmp[len - 1] == '/') tmp[len - 1] = 0;

    for (p = tmp; *p == '/'; p++);
    
    for (; *p; p++) {
        if (*p == '/') {
            *p = 0;
            if (mkdir(tmp, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0 && errno != EEXIST) {
                return -1;
            }
            *p = '/';
        }
    }
    if (mkdir(tmp, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0 && errno != EEXIST) {
        return -1;
    }
    return 0;
}

void* createVideo(const char* output_path, int row, int col, int fps, const char* unused) {
  (void)unused;

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
  thisvid->height = row;
  thisvid->width = col;
  thisvid->next_pts = 0;

  thisvid->sws_ctx = sws_getContext(
      col, row, AV_PIX_FMT_RGB24,
      col, row, AV_PIX_FMT_YUV420P,
      SWS_BILINEAR, nullptr, nullptr, nullptr
  );
  if (!thisvid->sws_ctx) {
    fmt::println(stderr, "Could not initialize swscale context");
    exit(1);
  }
  av_log_set_level(AV_LOG_WARNING);

  thisvid->pkt = av_packet_alloc();

  // Detect HLS mode: ONLY directory mode (trailing slash)
  thisvid->is_hls_mode = (output_path && output_path[strlen(output_path)-1] == '/');

  if (thisvid->is_hls_mode) {
    // === HLS OUTPUT MODE (DIRECTORY ONLY) ===
    char playlist_path[1024] = {0};
    char segment_template[1024] = {0};
    char hls_dir[1024] = {0};
    
    // SIMPLE PATH HANDLING (as requested):
    snprintf(playlist_path, sizeof(playlist_path), "%sstream.m3u8", output_path);
    snprintf(segment_template, sizeof(segment_template), "%sseg_%%03d.ts", output_path);
    snprintf(hls_dir, sizeof(hls_dir), "%s", output_path);

    // Ensure directory exists BEFORE avformat_write_header
    if (mkdir_p(hls_dir) != 0) {
        fmt::println(stderr, "Failed to create HLS directory '{}': {}", hls_dir, strerror(errno));
        exit(1);
    }

    // CRITICAL FIX: DO NOT manually open playlist file!
    // Instead: pass playlist_path to avformat_alloc_output_context2()
    int ret = avformat_alloc_output_context2(&thisvid->format, nullptr, "hls", playlist_path);
    if (!thisvid->format) {
      fmt::println(stderr, "Could not create HLS output context");
      exit(1);
    }

    // HLS options
    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "hls_time", "1", 0);
    av_dict_set(&opts, "hls_list_size", "0", 0);
    //av_dict_set(&opts, "hls_flags", "delete_segments+omit_endlist", 0);
    av_dict_set(&opts, "hls_segment_filename", segment_template, 0);
    
    // Create stream
    thisvid->stream = avformat_new_stream(thisvid->format, nullptr);
    if (!thisvid->stream) {
      fmt::println(stderr, "Failed to create HLS stream");
      exit(1);
    }
    thisvid->stream->time_base = (AVRational){1, fps};

    // Setup HEVC NVENC encoder WITH GOP settings
    if (!codec) codec = avcodec_find_encoder_by_name("hevc_nvenc");
    if (!codec) {
      fmt::println(stderr, "HEVC NVENC encoder not found");
      exit(1);
    }

    AVCodecContext *c = thisvid->c = avcodec_alloc_context3(codec);
    c->bit_rate = 10000000;
    c->width = col;
    c->height = row;
    c->framerate = (AVRational){fps, 1};
    c->time_base = av_inv_q(c->framerate);
    c->pix_fmt = AV_PIX_FMT_YUV420P;
    c->codec_tag = MKTAG('h','v','c','1');
    c->gop_size = fps;      // Keyframe every 1 second
    c->keyint_min = fps;

    av_opt_set(c->priv_data, "preset", "p7", 0);
    av_opt_set(c->priv_data, "tune", "ll", 0);
    av_opt_set(c->priv_data, "profile", "main", 0);

    ret = avcodec_open2(c, codec, nullptr);
    if (ret < 0) {
      fmt::println(stderr, "Could not open HEVC codec: {}", av_err2str(ret));
      exit(1);
    }

    ret = avcodec_parameters_from_context(thisvid->stream->codecpar, c);
    if (ret < 0) {
      fmt::println(stderr, "Failed to copy codec params: {}", av_err2str(ret));
      exit(1);
    }
    thisvid->stream->codecpar->codec_tag = MKTAG('h','v','c','1');

    // CRITICAL: DO NOT call avio_open() here!
    // avformat_write_header() will open the playlist file internally using thisvid->format->url
    
    ret = avformat_write_header(thisvid->format, &opts);
    av_dict_free(&opts);
    if (ret < 0) {
      fmt::println(stderr, "Failed to write HLS header: {}", av_err2str(ret));
      exit(1);
    }

    // Allocate encoder frame
    AVFrame *frame = thisvid->frame = av_frame_alloc();
    frame->format = c->pix_fmt;
    frame->width = c->width;
    frame->height = c->height;
    frame->pts = 0;
    frame->time_base = c->time_base;

    ret = av_frame_get_buffer(frame, 0);
    if (ret < 0) {
      fmt::println(stderr, "Could not allocate frame buffer: {}", av_err2str(ret));
      exit(1);
    }

    fmt::println("HLS streaming started: {}", playlist_path);
    fmt::println("  Segments: {}", segment_template);
    return thisvid;
  } else {
    // === REGULAR MP4 FILE MODE ===
    int ret = avformat_alloc_output_context2(&thisvid->format, nullptr, "mp4", output_path);
    if (!thisvid->format) {
      fmt::println(stderr, "Could not create MP4 output context");
      exit(1);
    }

    if (!codec) codec = avcodec_find_encoder_by_name("hevc_nvenc");
    if (!codec) {
      fmt::println(stderr, "HEVC NVENC encoder not found");
      exit(1);
    }

    thisvid->stream = avformat_new_stream(thisvid->format, codec);
    if (!thisvid->stream) {
      fmt::println(stderr, "Failed to create stream");
      exit(1);
    }

    AVCodecContext *c = thisvid->c = avcodec_alloc_context3(codec);
    c->bit_rate = 8000000;
    c->width = col;
    c->height = row;
    c->framerate = (AVRational){fps, 1};
    c->time_base = av_inv_q(c->framerate);
    c->pix_fmt = AV_PIX_FMT_YUV420P;
    c->codec_tag = MKTAG('h','v','c','1');
    c->gop_size = fps;
    c->keyint_min = fps;

    av_opt_set(c->priv_data, "preset", "p7", 0);
    av_opt_set(c->priv_data, "tune", "ll", 0);
    av_opt_set(c->priv_data, "profile", "main", 0);

    ret = avcodec_open2(c, codec, nullptr);
    if (ret < 0) {
      fmt::println(stderr, "Could not open codec: {}", av_err2str(ret));
      exit(1);
    }

    ret = avcodec_parameters_from_context(thisvid->stream->codecpar, c);
    if (ret < 0) {
      fmt::println(stderr, "Failed to copy codec params: {}", av_err2str(ret));
      exit(1);
    }
    thisvid->stream->codecpar->codec_tag = MKTAG('h','v','c','1');

    if (!(thisvid->format->oformat->flags & AVFMT_NOFILE)) {
      ret = avio_open(&thisvid->format->pb, output_path, AVIO_FLAG_WRITE);
      if (ret < 0) {
        fmt::println(stderr, "Could not open output file '{}': {}", output_path, av_err2str(ret));
        exit(1);
      }
    }

    ret = avformat_write_header(thisvid->format, nullptr);
    if (ret < 0) {
      fmt::println(stderr, "Failed to write header: {}", av_err2str(ret));
      exit(1);
    }

    AVFrame *frame = thisvid->frame = av_frame_alloc();
    frame->format = c->pix_fmt;
    frame->width = c->width;
    frame->height = c->height;
    frame->pts = 0;
    frame->time_base = c->time_base;

    ret = av_frame_get_buffer(frame, 0);
    if (ret < 0) {
      fmt::println(stderr, "Could not allocate frame buffer: {}", av_err2str(ret));
      exit(1);
    }

    return thisvid;
  }
}

void flushVideo(void* ptr, void* buffer) {
  struct video* thisvid = (struct video*)ptr;
  
  const uint8_t* src_data[1] = {(const uint8_t*)buffer};
  int src_stride[1] = {thisvid->width * 3};
  
  sws_scale(
      thisvid->sws_ctx,
      src_data, src_stride,
      0, thisvid->height,
      thisvid->frame->data, thisvid->frame->linesize
  );

  thisvid->frame->pts = thisvid->next_pts++;
  thisvid->frame->duration = 1;
  
  encode(thisvid);
}

void flushVideo_float(void* ptr, void* buffer) {
  struct video* thisvid = (struct video*)ptr;
  float* float_data = (float*)buffer;

  uint8_t* rgb_buf = (uint8_t*)av_malloc(thisvid->width * thisvid->height * 3);
  if (!rgb_buf) {
    fmt::println(stderr, "Failed to allocate RGB buffer for float conversion");
    return;
  }
  
  for (int i = 0; i < thisvid->width * thisvid->height; i++) {
    float val = float_data[i];
    if (val < 0.0f) val = 0.0f;
    else if (val > 1.0f) val = 1.0f;
    uint8_t byte = (uint8_t)(val * 255.0f);
    rgb_buf[i*3 + 0] = byte;
    rgb_buf[i*3 + 1] = byte;
    rgb_buf[i*3 + 2] = byte;
  }

  flushVideo(ptr, rgb_buf);
  av_free(rgb_buf);
}

void saveVideo(void* ptr) {
  struct video* v = (struct video*)ptr;

  avcodec_send_frame(v->c, nullptr);
  int ret;
  while (1) {
    ret = avcodec_receive_packet(v->c, v->pkt);
    if (ret == AVERROR_EOF || ret == AVERROR(EAGAIN)) break;
    if (ret < 0) {
      fmt::println(stderr, "Flush error: {}", av_err2str(ret));
      break;
    }

    if (v->pkt->duration == 0) {
        v->pkt->duration = 1;
    }

    av_packet_rescale_ts(v->pkt, v->c->time_base, v->stream->time_base);
    v->pkt->stream_index = v->stream->index;
    av_write_frame(v->format, v->pkt);
    av_packet_unref(v->pkt);
  }

  av_write_trailer(v->format);

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
