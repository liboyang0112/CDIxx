#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fmt/core.h"
#include "memManager.hpp"
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <pthread.h>
}


AVPacket *pkt = 0;
const AVCodec *codec = 0;
const unsigned char endcode[4]={0, 0, 1, 0xb7};
//void encode(AVCodecContext *enc_ctx, AVFrame *frame, FILE *outfile)
void* args_encode[3] = {0,0,0};
void* encode(void* args)
{
  AVCodecContext *enc_ctx = ((AVCodecContext**)args)[0];
  AVFrame *frame = ((AVFrame**)args)[1];
  FILE *outfile = ((FILE**)args)[2];
  int ret;

  ret = avcodec_send_frame(enc_ctx, frame);
  if (ret < 0) {
    fmt::println(stderr, "Error sending a frame for encoding");
    exit(1);
  }

  while (ret >= 0) {
    ret = avcodec_receive_packet(enc_ctx, pkt);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
      return 0;
    else if (ret < 0) {
      fmt::println(stderr, "Error during encoding");
      exit(1);
    }

    fwrite(pkt->data, 1, pkt->size, outfile);
    av_packet_unref(pkt);
  }
  return 0;
}

struct video{
  AVCodecContext *c;
  pthread_t stream_thread;
  FILE *f;
  AVFrame *frame;
};

void* createVideo(const char* filename, int row, int col, int fps){
  struct video* thisvid = (struct video*)ccmemMngr.borrowCache(sizeof(struct video));
  thisvid->stream_thread = 0;
  if(!pkt) pkt = av_packet_alloc();
  //if(!codec) codec = avcodec_find_encoder_by_name("libx265");
  if(!codec) codec = avcodec_find_encoder_by_name("hevc_nvenc");
  if (!codec) {
    fmt::println(stderr, "Codec libx265 not found");
    exit(1);
  }
  AVCodecContext *c = thisvid->c = avcodec_alloc_context3(codec);
  /* put sample parameters */
  //c->bit_rate = 400000;
  /* resolution must be a multiple of two */
  c->width = row;
  c->height = col;
  /* frames per second */
  c->time_base = (AVRational){1, fps};
  c->framerate = (AVRational){fps, 1};

  /* emit one intra frame every ten frames
   * check frame pict_type before passing frame
   * to encoder, if frame->pict_type is AV_PICTURE_TYPE_I
   * then gop_size is ignored and the output of encoder
   * will always be I frame irrespective to gop_size
   */
  //c->gop_size = 10;
  //c->max_b_frames = 4;
  c->pix_fmt = AV_PIX_FMT_YUV420P;

  av_opt_set(c->priv_data, "preset", "slow", 0);

  /* open it */
  int ret = avcodec_open2(c, codec, NULL);
  if (ret < 0) {
    fmt::println(stderr, "Could not open codec: {}", av_err2str(ret));
  }
  FILE* f = thisvid->f = fopen(filename, "wb");
  if (!f) {
    fmt::println(stderr, "Could not open {}", filename);
    exit(1);
  }

  AVFrame *frame = thisvid->frame = av_frame_alloc();
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
void* flushVideo_float(void* ptr, void* buffer){  //buffer is float
  struct video* thisvid = (struct video*) ptr;
  AVFrame *frame = thisvid->frame;
  AVCodecContext *c = thisvid->c;
  float* bfdata = (float*) buffer;
  int* size = frame->linesize;
  if(thisvid->stream_thread) pthread_join(thisvid->stream_thread, NULL);
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
  frame->pts += 1;
  args_encode[0] = c;
  args_encode[1] = frame;
  args_encode[2] = ((struct video*)ptr)->f;
  pthread_create(&(thisvid->stream_thread),NULL, encode, args_encode);
  return thisvid;
};
void flushVideo(void* ptr, void* buffer){  //buffer is rgb
  struct video* thisvid = (struct video*) ptr;
  AVFrame *frame = thisvid->frame;
  AVCodecContext *c = thisvid->c;
  unsigned char* rgbdata = (unsigned char*) buffer;
  int* size = frame->linesize;
  if(thisvid->stream_thread) pthread_join(thisvid->stream_thread, NULL);
  av_frame_make_writable(frame);
  for (int y = 0; y < c->height; y++) {
    for (int x = 0; x < c->width; x++) {
      unsigned char r,g,b;
      int idx = 3*(y*c->width+x);
      r = rgbdata[idx];
      g = rgbdata[idx+1];
      b = rgbdata[idx+2];
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
  frame->pts += 1;
  args_encode[0] = c;
  args_encode[1] = frame;
  args_encode[2] = ((struct video*)ptr)->f;
  pthread_create(&(thisvid->stream_thread),NULL, encode, args_encode);
};
void saveVideo(void* ptr){
  struct video* thisvid = (struct video*)ptr;
  if(thisvid->stream_thread) pthread_join(thisvid->stream_thread, NULL);
  FILE *f = thisvid->f;
  if (codec->id == AV_CODEC_ID_MPEG1VIDEO || codec->id == AV_CODEC_ID_MPEG2VIDEO)
    fwrite(endcode, 1, sizeof(endcode), f);
  fclose(f);
  avcodec_free_context(&(thisvid->c));
  av_frame_free(&(thisvid->frame));
  ccmemMngr.returnCache(ptr);
};
