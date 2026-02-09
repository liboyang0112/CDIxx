#ifndef STREAMER_H
#define STREAMER_H

#ifdef __cplusplus
extern "C" {
#endif

struct AVPacket;
struct AVCodecContext;

void* streamer_create(const char* socket_path, int width, int height, int fps);
void streamer_set_encoder_context(void* handle, struct AVCodecContext* enc_ctx);
void streamer_send_packet(void* handle, struct AVPacket* pkt);
void streamer_destroy(void* handle);

#ifdef __cplusplus
}
#endif

#endif // STREAMER_H
