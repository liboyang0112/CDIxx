// streamer.h - MJPEG completely removed
#ifndef STREAMER_H
#define STREAMER_H

void* streamer_create(const char* socket_path, int width, int height, int fps);
void streamer_set_encoder_context(void* handle, void* enc_ctx);
void streamer_send_packet(void* handle, void* pkt);
void streamer_send_frame(void* handle, const void* rgb_buffer);  // NO-OP stub
void streamer_destroy(void* handle);

#endif
