// streamer.c - MJPEG completely removed, only packet mode stubs
#include "streamer.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <libavcodec/avcodec.h>

typedef struct Client {
    int fd;
    struct Client* next;
} Client;

typedef struct {
    char socket_path[108];
    AVCodecContext* encoder_ctx;
    int server_fd;
    pthread_t accept_thread;
    volatile int running;
    pthread_mutex_t clients_mutex;
    Client* clients;
} Streamer;

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

        Client* client = calloc(1, sizeof(Client));
        client->fd = client_fd;

        pthread_mutex_lock(&s->clients_mutex);
        client->next = s->clients;
        s->clients = client;
        pthread_mutex_unlock(&s->clients_mutex);
    }
    return NULL;
}

void* streamer_create(const char* socket_path, int width, int height, int fps) {
    (void)width; (void)height; (void)fps;  // Not used in packet mode
    
    Streamer* s = calloc(1, sizeof(Streamer));
    if (!s) return NULL;

    s->running = 1;
    strncpy(s->socket_path, socket_path, sizeof(s->socket_path) - 1);
    s->socket_path[sizeof(s->socket_path) - 1] = '\0';

    unlink(s->socket_path);
    s->server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (s->server_fd < 0) { free(s); return NULL; }

    struct sockaddr_un addr = {0};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, s->socket_path, sizeof(addr.sun_path) - 1);
    addr.sun_path[sizeof(addr.sun_path) - 1] = '\0';
    
    if (bind(s->server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(s->server_fd); free(s); return NULL;
    }

    if (chmod(s->socket_path, 0666) < 0) {
        close(s->server_fd); unlink(s->socket_path); free(s); return NULL;
    }

    if (listen(s->server_fd, 5) < 0) {
        close(s->server_fd); unlink(s->socket_path); free(s); return NULL;
    }

    pthread_mutex_init(&s->clients_mutex, NULL);
    if (pthread_create(&s->accept_thread, NULL, accept_thread_func, s) != 0) {
        close(s->server_fd); unlink(s->socket_path); pthread_mutex_destroy(&s->clients_mutex); free(s); return NULL;
    }

    return s;
}

void streamer_set_encoder_context(void* handle, void* enc_ctx) {
    if (!handle) return;
    Streamer* s = handle;
    s->encoder_ctx = enc_ctx;
}

void streamer_send_packet(void* handle, void* pkt) {
    if (!handle || !pkt) return;
    Streamer* s = handle;
    if (!s->encoder_ctx) return;

    AVPacket* avpkt = pkt;
    if (avpkt->size <= 0) return;

    pthread_mutex_lock(&s->clients_mutex);
    for (Client* c = s->clients; c; c = c->next) {
        if (c->fd <= 0) continue;
        // Note: Actual packet sending over Unix socket requires custom protocol
        // This is a stub - real implementation would need framing/serialization
    }
    pthread_mutex_unlock(&s->clients_mutex);
}

// NO-OP stub - MJPEG completely removed
void streamer_send_frame(void* handle, const void* rgb_buffer) {
    (void)handle; (void)rgb_buffer;
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
    pthread_mutex_unlock(&s->clients_mutex);

    unlink(s->socket_path);
    pthread_mutex_destroy(&s->clients_mutex);
    free(s);
}
