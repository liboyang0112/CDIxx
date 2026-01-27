#include <assert.h>
#include <fmt/base.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fmt/core.h"
#include "memManager.hpp"
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
}
const AVCodec *codec = 0;
//void encode(AVCodecContext *enc_ctx, AVFrame *frame, AVFormatContext *outfile)
#include <SDL2/SDL.h>
struct video{
  AVCodecContext *c;
  AVFormatContext *format;
  AVFrame *frame;
  AVPacket *pkt;
  AVStream* stream;
};

void* encode(void* arg) {
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

    // Rescale PTS/DTS from codec timebase → stream timebase
    av_packet_rescale_ts(v->pkt, v->c->time_base, v->stream->time_base);
    v->pkt->stream_index = v->stream->index;

    // Write packet to MP4 container
    ret = av_write_frame(v->format, v->pkt);
    if (ret < 0) {
      fmt::println(stderr, "Write error: {}", av_err2str(ret));
    }

    av_packet_unref(v->pkt); // reset for reuse; no free()
  }

  // Optional: flush muxer (though av_write_trailer does this)
  av_write_frame(v->format, nullptr);

  return nullptr;
}

void* createVideo(const char* filename, int row, int col, int fps){
  struct video* thisvid = (struct video*)ccmemMngr.borrowCache(sizeof(struct video));
  thisvid->pkt = av_packet_alloc();
  int ret = avformat_alloc_output_context2(&thisvid->format, nullptr, "mp4", filename);
  if (!thisvid->format) {
    fmt::println(stderr, "Could not create output context");
    exit(1);
  }
  //if(!codec) codec = avcodec_find_encoder_by_name("libx265");
  if(!codec) codec = avcodec_find_encoder_by_name("hevc_nvenc");
  //if(!codec) codec = avcodec_find_encoder(AV_CODEC_ID_H265);
  if (!codec) {
    fmt::println(stderr, "Codec libx265 not found");
    exit(1);
  }
  thisvid->stream = avformat_new_stream(thisvid->format, codec);
  if (!thisvid->stream) {
    fmt::println(stderr, "Failed to create stream");
    exit(1);
  }
  AVCodecContext *c = thisvid->c = avcodec_alloc_context3(codec);
  /* put sample parameters */
  c->bit_rate = 800000;
  /* resolution must be a multiple of two */
  c->width = row;
  c->height = col;
  /* frames per second */
  c->time_base = (AVRational){1, fps};
  c->pkt_timebase = (AVRational){1, fps};
  c->framerate = (AVRational){fps, 1};

  c->pix_fmt = AV_PIX_FMT_YUV420P;

  av_opt_set(c->priv_data, "preset", "slow", 0);

  ret = avcodec_parameters_from_context(thisvid->stream->codecpar, thisvid->c);
  if (ret < 0) {
    fmt::println(stderr, "Failed to copy codec params");
    exit(1);
  }
  /* open it */
  ret = avcodec_open2(c, codec, NULL);
  if (ret < 0) {
    fmt::println(stderr, "Could not open codec: {}", av_err2str(ret));
  }
  if (!(thisvid->format->oformat->flags & AVFMT_NOFILE)) {
    ret = avio_open(&thisvid->format->pb, filename, AVIO_FLAG_WRITE);
    if (ret < 0) {
      fmt::println(stderr, "Could not open file: {}", av_err2str(ret));
      exit(1);
    }
  }
  ret = avformat_write_header(thisvid->format, nullptr);
  if (ret < 0) {
    fmt::println(stderr, "Header write failed: {}", av_err2str(ret));
    exit(1);
  }

  AVFrame *frame = thisvid->frame = av_frame_alloc();
  frame->pts = 0;
  if (!frame) {
    fmt::println(stderr, "Could not allocate video frame");
    exit(1);
  }
  frame->format = c->pix_fmt;
  frame->width  = c->width;
  frame->height = c->height;

  ret = av_frame_get_buffer(frame, 0);
  if (ret < 0) {
    fmt::println(stderr, "Could not allocate the video frame data");
    exit(1);
  }
  return thisvid;
}
void flushVideo_float(void* ptr, void* buffer){  //buffer is float
  struct video* thisvid = (struct video*) ptr;
  AVFrame *frame = thisvid->frame;
  AVCodecContext *c = thisvid->c;
  float* bfdata = (float*)buffer;
  int* size = frame->linesize;
  av_frame_make_writable(frame);
  for (int y = 0; y < c->height; y++) {
    for (int x = 0; x < c->width; x++) {
      float fdata = bfdata[y*c->width+x];
      if(fdata > 1) fdata = 1;
      int data = fdata * (1<<24);
      /*
         frame->data[0][y*size[0]+x] = data & 255;  //Y
         data = data>>8;
         if(y%2==0 && x%2==0) {
         frame->data[1][y/2*size[1]+x/2] = (data & 255)/4;
         frame->data[2][y/2*size[2]+x/2] = (data>>8)/4;
         }else{
         frame->data[1][y/2*size[1]+x/2] += (data & 255)/4;
         frame->data[2][y/2*size[2]+x/2] += (data>>8)/4;
         }
         */
      frame->data[0][y*size[0]+x] = (data & 15) + ((data >> 8) & (15<<4));  //Y
      data = data>>4;
      if(y%2==0 && x%2==0) {
        frame->data[1][y/2*size[1]+x/2] = ((data & 15) + ((data >> 8) & (15<<4)))/4;
        data = data>>4;
        frame->data[2][y/2*size[2]+x/2] = ((data & 15) + ((data >> 8) & (15<<4)))/4;
      }else{
        frame->data[1][y/2*size[1]+x/2] += ((data & 15) + ((data >> 8) & (15<<4)))/4;
        data = data>>4;
        frame->data[2][y/2*size[2]+x/2] += ((data & 15) + ((data >> 8) & (15<<4)))/4;
      }
    }
  }
  encode(thisvid);
  frame->pts += 1;
}

void flushVideo(void* ptr, void* buffer){  //buffer is rgb
  struct video* thisvid = (struct video*) ptr;
  unsigned char* dat = (unsigned char*)buffer;
  AVFrame *frame = thisvid->frame;
  AVCodecContext *c = thisvid->c;
  int* size = frame->linesize;
  av_frame_make_writable(frame);
  for (int y = 0; y < c->height; y++) {
    for (int x = 0; x < c->width; x++) {
      unsigned char r,g,b;
      int idx = 3*(y*c->width+x);
      r = dat[idx];
      g = dat[idx+1];
      b = dat[idx+2];
      frame->data[0][y*size[0]+x] = 0.257*r+0.564*g+0.098*b+16;  //Y
      if(y%2==0 && x%2==0) {
        frame->data[1][y/2*size[1]+x/2] = (-0.148*r-0.291*g+0.439*b+128)/4;
        frame->data[2][y/2*size[2]+x/2] = (0.439*r-0.368*g-0.071*b+128)/4;
      }else{
        frame->data[1][y/2*size[1]+x/2] += (-0.148*r-0.291*g+0.439*b+128)/4; //Cb
        frame->data[2][y/2*size[2]+x/2] += (0.439*r-0.368*g-0.071*b+128)/4; //Cr
      }
    }
  }
  encode(thisvid);
  frame->pts += 1;
};
void saveVideo(void* ptr) {
  struct video* v = (struct video*)ptr;

  // Wait for last frame
  // Final flush: send NULL frame
  avcodec_send_frame(v->c, nullptr);
  int ret;
  while (1) {
    ret = avcodec_receive_packet(v->c, v->pkt);
    if (ret == AVERROR_EOF || ret == AVERROR(EAGAIN)) break;
    if (ret < 0) break;

    av_packet_rescale_ts(v->pkt, v->c->time_base, v->stream->time_base);
    v->pkt->stream_index = v->stream->index;
    av_write_frame(v->format, v->pkt);
    av_packet_unref(v->pkt);
  }

  // Finalize file
  av_write_trailer(v->format);

  // Cleanup
  if (!(v->format->oformat->flags & AVFMT_NOFILE)) {
    avio_closep(&v->format->pb);
  }
  avformat_free_context(v->format);
  avcodec_free_context(&v->c);
  av_frame_free(&v->frame);
  av_packet_free(&v->pkt); // ← only free here
  ccmemMngr.returnCache(v);
}
