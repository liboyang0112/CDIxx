#ifndef STREAMER_H
#define STREAMER_H

#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#else
#endif

#define STREAMER_MODE_PACKET 0
#define STREAMER_MODE_IMG 1

void* streamer_create(const char* socket_path, int width, int height, int fps);
void streamer_set_encoder_context(void* handle, void* enc_ctx);
void streamer_send_packet(void* handle, void* pkt);
void streamer_send_frame(void* handle, const void* data) ;
void streamer_destroy(void* handle);

#ifdef __cplusplus
}
#endif

#endif
