#include "streamer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <stdint.h>
#include <poll.h>
#include <time.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <turbojpeg.h>

typedef struct Client {
    int fd;
    struct Client* next;
} Client;

typedef struct {
    char socket_path[108];
    int width, height, fps;
    int is_mjpeg;
    
    // MJPEG mode state
    pthread_mutex_t frame_mutex;
    pthread_cond_t frame_cond;
    unsigned char* latest_rgb;
    size_t rgb_size;
    volatile int new_frame;
    
    // Packet mode state
    AVCodecContext* encoder_ctx;
    
    // Common
    int server_fd;
    pthread_t accept_thread;
    volatile int running;
    pthread_mutex_t clients_mutex;
    Client* clients;
} Streamer;

typedef struct {
    Streamer* streamer;
    int client_fd;
} ClientThreadArgs;

static void* accept_thread_func(void* arg);
static void* client_thread_func(void* arg);

void* streamer_create(const char* socket_path, int width, int height, int fps) {
    if (!socket_path || width <= 0 || height <= 0 || fps <= 0) {
        fprintf(stderr, "Invalid streamer parameters\n");
        return NULL;
    }

    Streamer* s = calloc(1, sizeof(Streamer));
    if (!s) return NULL;

    s->width = width;
    s->height = height;
    s->fps = fps;
    s->running = 1;
    s->is_mjpeg = (strncmp(socket_path, "mjpeg://", 8) == 0);
    
    if (s->is_mjpeg) {
        strncpy(s->socket_path, socket_path + 8, sizeof(s->socket_path) - 1);
        s->socket_path[sizeof(s->socket_path) - 1] = '\0';
        pthread_mutex_init(&s->frame_mutex, NULL);
        pthread_cond_init(&s->frame_cond, NULL);
        s->rgb_size = width * height * 3;
        s->latest_rgb = malloc(s->rgb_size);
        if (!s->latest_rgb) {
            free(s);
            return NULL;
        }
        memset(s->latest_rgb, 0, s->rgb_size);
        s->new_frame = 0;
    } else {
        strncpy(s->socket_path, socket_path, sizeof(s->socket_path) - 1);
        s->socket_path[sizeof(s->socket_path) - 1] = '\0';
    }

    unlink(s->socket_path);
    s->server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (s->server_fd < 0) {
        if (s->is_mjpeg) {
            free(s->latest_rgb);
            pthread_mutex_destroy(&s->frame_mutex);
            pthread_cond_destroy(&s->frame_cond);
        }
        free(s);
        return NULL;
    }

    struct sockaddr_un addr = {0};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, s->socket_path, sizeof(addr.sun_path) - 1);
    addr.sun_path[sizeof(addr.sun_path) - 1] = '\0';
    if (bind(s->server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(s->server_fd);
        if (s->is_mjpeg) {
            free(s->latest_rgb);
            pthread_mutex_destroy(&s->frame_mutex);
            pthread_cond_destroy(&s->frame_cond);
        }
        free(s);
        return NULL;
    }

    if (chmod(s->socket_path, 0666) < 0) {
        close(s->server_fd);
        unlink(s->socket_path);
        if (s->is_mjpeg) {
            free(s->latest_rgb);
            pthread_mutex_destroy(&s->frame_mutex);
            pthread_cond_destroy(&s->frame_cond);
        }
        free(s);
        return NULL;
    }

    if (listen(s->server_fd, 5) < 0) {
        close(s->server_fd);
        unlink(s->socket_path);
        if (s->is_mjpeg) {
            free(s->latest_rgb);
            pthread_mutex_destroy(&s->frame_mutex);
            pthread_cond_destroy(&s->frame_cond);
        }
        free(s);
        return NULL;
    }

    pthread_mutex_init(&s->clients_mutex, NULL);
    if (pthread_create(&s->accept_thread, NULL, accept_thread_func, s) != 0) {
        close(s->server_fd);
        unlink(s->socket_path);
        pthread_mutex_destroy(&s->clients_mutex);
        if (s->is_mjpeg) {
            free(s->latest_rgb);
            pthread_mutex_destroy(&s->frame_mutex);
            pthread_cond_destroy(&s->frame_cond);
        }
        free(s);
        return NULL;
    }

    return s;
}

void streamer_set_encoder_context(void* handle, void* enc_ctx) {
    if (!handle) return;
    Streamer* s = handle;
    if (!s->is_mjpeg)
        s->encoder_ctx = enc_ctx;
}

void streamer_send_packet(void* handle, void* pkt) {
    if (!handle || !pkt) return;
    Streamer* s = handle;
    if (s->is_mjpeg || !s->encoder_ctx) return;

    AVPacket* avpkt = pkt;
    if (avpkt->size <= 0) return;

    pthread_mutex_lock(&s->clients_mutex);
    Client* c = s->clients;
    while (c) {
        if (c->fd > 0) {
            AVPacket* clone = av_packet_clone(avpkt);
            if (clone) {
                if (clone->pts != AV_NOPTS_VALUE)
                    clone->pts = av_rescale_q(clone->pts, s->encoder_ctx->time_base, (AVRational){1, 1000});
                if (clone->dts != AV_NOPTS_VALUE)
                    clone->dts = av_rescale_q(clone->dts, s->encoder_ctx->time_base, (AVRational){1, 1000});
                clone->stream_index = 0;
                
                // Note: Direct packet sending over Unix socket isn't standard
                // This is a placeholder - actual implementation would require
                // a custom protocol or using FFmpeg's avio_write() with proper context
                av_packet_free(&clone);
            }
        }
        c = c->next;
    }
    pthread_mutex_unlock(&s->clients_mutex);
}

void streamer_send_frame(void* handle, const void* rgb_buffer) {
    if (!handle || !rgb_buffer) return;
    Streamer* s = handle;
    if (!s->is_mjpeg) return;

    pthread_mutex_lock(&s->frame_mutex);
    memcpy(s->latest_rgb, rgb_buffer, s->rgb_size);
    s->new_frame = 1;
    pthread_cond_broadcast(&s->frame_cond);
    pthread_mutex_unlock(&s->frame_mutex);
}

void streamer_destroy(void* handle) {
    if (!handle) return;
    Streamer* s = handle;
    s->running = 0;

    shutdown(s->server_fd, SHUT_RDWR);
    close(s->server_fd);
    pthread_join(s->accept_thread, NULL);

    pthread_mutex_lock(&s->clients_mutex);
    Client* c = s->clients;
    while (c) {
        Client* next = c->next;
        if (c->fd > 0) close(c->fd);
        free(c);
        c = next;
    }
    s->clients = NULL;
    pthread_mutex_unlock(&s->clients_mutex);

    unlink(s->socket_path);
    pthread_mutex_destroy(&s->clients_mutex);
    
    if (s->is_mjpeg) {
        free(s->latest_rgb);
        pthread_mutex_destroy(&s->frame_mutex);
        pthread_cond_destroy(&s->frame_cond);
    }
    
    free(s);
}

static void* accept_thread_func(void* arg) {
    Streamer* s = arg;

    while (s->running) {
        struct sockaddr_un client_addr;
        socklen_t addr_len = sizeof(client_addr);
        int client_fd = accept(s->server_fd, (struct sockaddr*)&client_addr, &addr_len);
        if (client_fd < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;
            break;
        }

        int flags = fcntl(client_fd, F_GETFL, 0);
        fcntl(client_fd, F_SETFL, flags | O_NONBLOCK);

        if (s->is_mjpeg) {
            // RAW MJPEG MODE: No HTTP handshake - start streaming immediately
            ClientThreadArgs* args = malloc(sizeof(ClientThreadArgs));
            if (!args) {
                close(client_fd);
                continue;
            }
            args->streamer = s;
            args->client_fd = client_fd;
            
            pthread_t client_thread;
            if (pthread_create(&client_thread, NULL, client_thread_func, args) != 0) {
                free(args);
                close(client_fd);
                continue;
            }
            pthread_detach(client_thread);
        } else {
            // Packet mode: store client FD for packet sending
            Client* client = calloc(1, sizeof(Client));
            if (!client) {
                close(client_fd);
                continue;
            }
            client->fd = client_fd;

            pthread_mutex_lock(&s->clients_mutex);
            client->next = s->clients;
            s->clients = client;
            pthread_mutex_unlock(&s->clients_mutex);
        }
    }
    return NULL;
}

static void* client_thread_func(void* arg) {
    ClientThreadArgs* args = (ClientThreadArgs*)arg;
    Streamer* s = args->streamer;
    int client_fd = args->client_fd;
    free(args);

    if (!s || !s->is_mjpeg || !s->running) {
        close(client_fd);
        return NULL;
    }

    tjhandle tj = tjInitCompress();
    if (!tj) {
        close(client_fd);
        return NULL;
    }

    // Simple frame pacing: target FPS
    struct timespec frame_interval = {
        .tv_sec = 0,
        .tv_nsec = 1000000000L / s->fps
    };

    while (s->running) {
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        // Wait for new frame (with timeout)
        pthread_mutex_lock(&s->frame_mutex);
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += 1; // 1s timeout
        
        while (!s->new_frame && s->running) {
            if (pthread_cond_timedwait(&s->frame_cond, &s->frame_mutex, &ts) == ETIMEDOUT)
                break;
        }

        if (!s->new_frame) {
            pthread_mutex_unlock(&s->frame_mutex);
            continue;
        }

        // Copy frame
        unsigned char* rgb_copy = malloc(s->rgb_size);
        if (!rgb_copy) {
            pthread_mutex_unlock(&s->frame_mutex);
            break;
        }
        memcpy(rgb_copy, s->latest_rgb, s->rgb_size);
        s->new_frame = 0;
        pthread_mutex_unlock(&s->frame_mutex);

        // Compress to JPEG
        unsigned char* jpeg_buf = NULL;
        unsigned long jpeg_size = 0;
        int compress_ret = tjCompress2(tj, rgb_copy, s->width, 0, s->height, TJPF_RGB,
                                      &jpeg_buf, &jpeg_size, TJSAMP_420, 85, 0);
        free(rgb_copy);

        if (compress_ret == 0 && jpeg_buf && jpeg_size > 0) {
            // SEND RAW JPEG FRAME (no HTTP headers/boundaries) - ffmpeg compatible
            size_t total_sent = 0;
            while (total_sent < jpeg_size && s->running) {
                ssize_t sent = send(client_fd, jpeg_buf + total_sent, jpeg_size - total_sent, MSG_NOSIGNAL);
                if (sent <= 0) {
                    if (errno == EAGAIN || errno == EWOULDBLOCK) {
                        struct pollfd pfd = {client_fd, POLLOUT, 0};
                        if (poll(&pfd, 1, 100) <= 0) break;
                        continue;
                    }
                    // Client disconnected
                    tjFree(jpeg_buf);
                    tjDestroy(tj);
                    close(client_fd);
                    return NULL;
                }
                total_sent += sent;
            }
            tjFree(jpeg_buf);
        } else {
            if (jpeg_buf) tjFree(jpeg_buf);
        }

        // Frame pacing
        clock_gettime(CLOCK_MONOTONIC, &end);
        long elapsed_ns = (end.tv_sec - start.tv_sec) * 1000000000L + (end.tv_nsec - start.tv_nsec);
        long sleep_ns = frame_interval.tv_nsec - elapsed_ns;
        if (sleep_ns > 0) {
            struct timespec sleep_ts = {.tv_sec = 0, .tv_nsec = sleep_ns};
            nanosleep(&sleep_ts, NULL);
        }
    }

    tjDestroy(tj);
    close(client_fd);
    return NULL;
}
