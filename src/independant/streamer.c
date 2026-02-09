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
#include <time.h>
#include <poll.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>

typedef struct Client {
    int fd;
    AVFormatContext* fmt_ctx;
    uint8_t* io_buffer;
    size_t io_buffer_size;
    struct Client* next;
} Client;

typedef struct {
    char socket_path[108];
    int width, height, fps;
    int server_fd;
    pthread_t accept_thread;
    volatile int running;
    pthread_mutex_t clients_mutex;
    Client* clients;
    
    AVCodecContext* encoder_ctx;
} Streamer;

static void* accept_thread_func(void* arg);
static void handle_client(Streamer* s, Client* client);
static int write_packet(void* opaque, uint8_t* buf, int buf_size);
static void send_data(int fd, const uint8_t* data, size_t size);

// ===== PUBLIC API =====

void* streamer_create(const char* socket_path, int width, int height, int fps) {
    // NOTE: Function name kept for API compatibility - actually creates WebM streamer
    if (!socket_path || width <= 0 || height <= 0 || fps <= 0) {
        fprintf(stderr, "Invalid streamer parameters\n");
        return NULL;
    }
    
    Streamer* s = (Streamer*)calloc(1, sizeof(Streamer));
    if (!s) return NULL;
    
    strncpy(s->socket_path, socket_path, sizeof(s->socket_path) - 1);
    s->width = width;
    s->height = height;
    s->fps = fps;
    s->running = 1;
    pthread_mutex_init(&s->clients_mutex, NULL);
    
    unlink(s->socket_path);
    
    s->server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (s->server_fd < 0) {
        perror("socket");
        free(s);
        return NULL;
    }
    
    struct sockaddr_un addr = {0};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, s->socket_path, sizeof(addr.sun_path) - 1);
    
    if (bind(s->server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(s->server_fd);
        free(s);
        return NULL;
    }
    
    if (chmod(s->socket_path, 0666) < 0) {
        perror("chmod");
        close(s->server_fd);
        unlink(s->socket_path);
        free(s);
        return NULL;
    }
    
    if (listen(s->server_fd, 5) < 0) {
        perror("listen");
        close(s->server_fd);
        unlink(s->socket_path);
        free(s);
        return NULL;
    }
    
    if (pthread_create(&s->accept_thread, NULL, accept_thread_func, s) != 0) {
        close(s->server_fd);
        unlink(s->socket_path);
        free(s);
        return NULL;
    }
    
    fprintf(stderr, "📡 WebM streamer ready at %s (%dx%d@%dfps)\n", 
            s->socket_path, width, height, fps);
    return s;
}

void streamer_set_encoder_context(void* handle, AVCodecContext* enc_ctx) {
    if (!handle || !enc_ctx) return;
    
    Streamer* s = (Streamer*)handle;
    s->encoder_ctx = enc_ctx;
    fprintf(stderr, "   → Encoder context set (WebM streaming)\n");
}

void streamer_send_packet(void* handle, AVPacket* pkt) {
    if (!handle || !pkt || pkt->size <= 0) return;
    
    Streamer* s = (Streamer*)handle;
    
    pthread_mutex_lock(&s->clients_mutex);
    Client* c = s->clients;
    while (c) {
        if (c->fmt_ctx) {
            AVPacket* clone = av_packet_clone(pkt);
            if (clone) {
                // WebM uses simple timestamp scaling (no DTS/PTS conversion headaches)
                if (pkt->pts != AV_NOPTS_VALUE)
                    clone->pts = av_rescale_q(pkt->pts, s->encoder_ctx->time_base, c->fmt_ctx->streams[0]->time_base);
                if (pkt->dts != AV_NOPTS_VALUE)
                    clone->dts = av_rescale_q(pkt->dts, s->encoder_ctx->time_base, c->fmt_ctx->streams[0]->time_base);
                
                if (av_write_frame(c->fmt_ctx, clone) >= 0) {
                    av_write_frame(c->fmt_ctx, NULL);  // Flush
                    send_data(c->fd, c->io_buffer, c->io_buffer_size);
                }
                av_packet_free(&clone);
            }
        }
        c = c->next;
    }
    pthread_mutex_unlock(&s->clients_mutex);
}

void streamer_destroy(void* handle) {
    if (!handle) return;
    
    Streamer* s = (Streamer*)handle;
    s->running = 0;
    
    shutdown(s->server_fd, SHUT_RDWR);
    close(s->server_fd);
    pthread_join(s->accept_thread, NULL);
    
    pthread_mutex_lock(&s->clients_mutex);
    Client* c = s->clients;
    while (c) {
        Client* next = c->next;
        
        if (c->fmt_ctx) {
            av_write_trailer(c->fmt_ctx);
            if (c->fmt_ctx->pb) avio_context_free(&c->fmt_ctx->pb);
            avformat_free_context(c->fmt_ctx);
        }
        if (c->io_buffer) av_free(c->io_buffer);
        close(c->fd);
        free(c);
        c = next;
    }
    pthread_mutex_unlock(&s->clients_mutex);
    
    unlink(s->socket_path);
    pthread_mutex_destroy(&s->clients_mutex);
    free(s);
    
    fprintf(stderr, "🗑️  Streamer destroyed\n");
}

// ===== PRIVATE IMPLEMENTATION =====

static void* accept_thread_func(void* arg) {
    Streamer* s = (Streamer*)arg;
    
    while (s->running) {
        struct sockaddr_un client_addr;
        socklen_t addr_len = sizeof(client_addr);
        int client_fd = accept(s->server_fd, (struct sockaddr*)&client_addr, &addr_len);
        
        if (client_fd < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;
            break;
        }
        
        if (!s->running || !s->encoder_ctx) {
            close(client_fd);
            break;
        }
        
        // Set non-blocking
        int flags = fcntl(client_fd, F_GETFL, 0);
        fcntl(client_fd, F_SETFL, flags | O_NONBLOCK);
        
        // Read HTTP request
        char request[1024] = {0};
        struct pollfd pfd = {client_fd, POLLIN, 0};
        if (poll(&pfd, 1, 5000) <= 0 || !(pfd.revents & POLLIN)) {
            close(client_fd);
            continue;
        }
        
        ssize_t n = recv(client_fd, request, sizeof(request) - 1, 0);
        if (n <= 0 || !strstr(request, "GET /CDIxx/stream")) {
            close(client_fd);
            continue;
        }
        
        // Send HTTP headers with WebM content type
        const char* headers = 
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: video/webm\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n\r\n";
        send(client_fd, headers, strlen(headers), MSG_NOSIGNAL);
        
        // Create client muxer - WEBM (no flags needed!)
        Client* client = (Client*)calloc(1, sizeof(Client));
        client->fd = client_fd;
        client->io_buffer = (uint8_t*)av_malloc(2 * 1024 * 1024);
        if (!client->io_buffer) {
            free(client);
            close(client_fd);
            continue;
        }
        
        // Allocate WebM muxer - works perfectly with non-seekable output
        if (avformat_alloc_output_context2(&client->fmt_ctx, NULL, "webm", NULL) < 0) {
            av_free(client->io_buffer);
            free(client);
            close(client_fd);
            continue;
        }
        
        // Create stream and copy parameters from encoder
        AVStream* stream = avformat_new_stream(client->fmt_ctx, NULL);
        avcodec_parameters_from_context(stream->codecpar, s->encoder_ctx);
        stream->time_base = s->encoder_ctx->time_base;  // WebM preserves encoder timebase
        
        // Setup custom IO context
        client->fmt_ctx->pb = avio_alloc_context(
            client->io_buffer, 2*1024*1024, 1, client,
            NULL, write_packet, NULL
        );
        
        // Write header - WebM writes all headers upfront (no seeking required)
        if (avformat_write_header(client->fmt_ctx, NULL) < 0) {
            avio_context_free(&client->fmt_ctx->pb);
            avformat_free_context(client->fmt_ctx);
            av_free(client->io_buffer);
            free(client);
            close(client_fd);
            continue;
        }
        av_write_frame(client->fmt_ctx, NULL);  // Flush header
        
        // Send header
        if (client->io_buffer_size > 0) {
            send_data(client->fd, client->io_buffer, client->io_buffer_size);
            fprintf(stderr, "   → Sent %zu bytes WebM header\n", client->io_buffer_size);
        }
        
        // Add to client list
        pthread_mutex_lock(&s->clients_mutex);
        client->next = s->clients;
        s->clients = client;
        pthread_mutex_unlock(&s->clients_mutex);
        
        fprintf(stderr, "🎥 Client connected (WebM streaming)\n");
        handle_client(s, client);
    }
    return NULL;
}

static void handle_client(Streamer* s, Client* client) {
    struct pollfd pfd = {client->fd, POLLIN | POLLHUP, 0};
    
    while (s->running) {
        int ret = poll(&pfd, 1, 100);
        if (ret < 0) break;
        if (ret == 0) continue;
        if (pfd.revents & (POLLHUP | POLLERR | POLLNVAL)) break;
        
        // Drain incoming data
        char buf[256];
        while (recv(client->fd, buf, sizeof(buf), MSG_DONTWAIT) > 0);
    }
    
    // Remove client
    pthread_mutex_lock(&s->clients_mutex);
    Client** pp = &s->clients;
    while (*pp) {
        if (*pp == client) {
            *pp = client->next;
            break;
        }
        pp = &(*pp)->next;
    }
    pthread_mutex_unlock(&s->clients_mutex);
    
    fprintf(stderr, "👋 Client disconnected\n");
}

static int write_packet(void* opaque, uint8_t* buf, int buf_size) {
    Client* client = (Client*)opaque;
    if (buf_size < 0 || buf_size > (int)(2 * 1024 * 1024)) return -1;
    memcpy(client->io_buffer, buf, buf_size);
    client->io_buffer_size = buf_size;
    return buf_size;
}

static void send_data(int fd, const uint8_t* data, size_t size) {
    if (size == 0) return;
    send(fd, data, size, MSG_NOSIGNAL);
}
