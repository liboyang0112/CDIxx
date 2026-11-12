#include <SDL3/SDL.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <poll.h>
#include <fcntl.h>
#include <errno.h>
#include <alloca.h>

// ── SDL local preview (online=false) ─────────────────────────────
struct PreviewContext {
  SDL_Window* window;
  SDL_Renderer* renderer;
  SDL_Texture* texture;
  SDL_Event e;
  char active;
  int width, height;
};

void* createPreview(const char* windowname, int row, int col){
  struct PreviewContext* p = (struct PreviewContext*)malloc(sizeof(struct PreviewContext));
  p->width = row;
  p->height = col;
  assert(SDL_Init(SDL_INIT_VIDEO));
  p->window = SDL_CreateWindow(
      windowname,
      row > 1920 ? 1920 : row,
      col > 1080 ? 1080 : col,
      SDL_WINDOW_RESIZABLE
      );
  assert(p->window != 0);
  p->renderer = SDL_CreateRenderer(p->window, NULL);
  p->texture = SDL_CreateTexture(p->renderer,SDL_PIXELFORMAT_RGB24,SDL_TEXTUREACCESS_STREAMING,row, col);
  p->active = 1;
  return p;
}
void updatePreview(void* ptr, void* buffer){
  struct PreviewContext* p = (struct PreviewContext*)ptr;
  if(!p->active) return;
  SDL_PollEvent(&p->e);
  if (p->e.type == SDL_EVENT_QUIT || (p->e.type == SDL_EVENT_KEY_DOWN && p->e.key.key == SDLK_Q)) {
    SDL_Quit();
  }
  void* pixels;
  int pitch;
  if (SDL_LockTexture(p->texture, NULL, &pixels, &pitch)) {
    memcpy((uint8_t*)pixels,buffer, p->height * p->width * 3);
    SDL_UnlockTexture(p->texture);
    SDL_RenderTexture(p->renderer, p->texture, NULL, NULL);
    SDL_RenderPresent(p->renderer);
  }
};

// ── MJPEG streaming over Unix socket (online=true) ───────────────
// Sends standard MJPEG (multipart/x-mixed-replace) so the browser
// displays it natively with <img src="http://host/live">.
//
// cuPlotter encodes JPEG on GPU via writeJPEG_GPU to a tmpfs file,
// reads the bytes, calls updateStream() which sends them here.

struct StreamContext {
  int srvSock;
  volatile int clientFd;
  char sockPath[108];
  pthread_t acceptThread;
  int active;
};

static void* streamAcceptLoop(void* arg) {
  struct StreamContext* s = (struct StreamContext*)arg;
  while (s->active) {
    struct pollfd pfd = { .fd = s->srvSock, .events = POLLIN };
    if (poll(&pfd, 1, 200) <= 0) continue;

    int cfd = accept(s->srvSock, NULL, NULL);
    if (cfd < 0) continue;

    // Non-blocking: drop frames if consumer is slow
    int flags = fcntl(cfd, F_GETFL, 0);
    fcntl(cfd, F_SETFL, flags | O_NONBLOCK);

    // Buffer large enough for one full frame (2048x2048 JPEG ~1MB)
    // Prevents partial writes that corrupt the stream
    int bufsize = 4 * 1024 * 1024;
    setsockopt(cfd, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize));

    // Single client: replace existing
    int old = s->clientFd;
    s->clientFd = cfd;
    if (old >= 0) close(old);

    const char* hdr =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
        "Connection: keep-alive\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n";
    write(cfd, hdr, strlen(hdr));
    fprintf(stderr, "[Stream] Client connected on %s\n", s->sockPath);
  }
  return NULL;
}

void* createStream(const char* sockPath, int row, int col) {
  (void)row; (void)col;
  struct StreamContext* s = (struct StreamContext*)calloc(1, sizeof(struct StreamContext));
  s->clientFd = -1;
  s->active = 1;
  strncpy(s->sockPath, sockPath, 107);

  unlink(sockPath);

  s->srvSock = socket(AF_UNIX, SOCK_STREAM, 0);
  if (s->srvSock < 0) {
    perror("[Stream] socket");
    free(s);
    return NULL;
  }

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, sockPath, sizeof(addr.sun_path) - 1);

  if (bind(s->srvSock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
    perror("[Stream] bind");
    close(s->srvSock);
    free(s);
    return NULL;
  }
  listen(s->srvSock, 4);

  pthread_create(&s->acceptThread, NULL, streamAcceptLoop, s);
  fprintf(stderr, "[Stream] Listening on %s\n", sockPath);
  return s;
}

// Send one JPEG frame — non-blocking, drops frame if buffer full or too soon
// Min interval: 100ms (10fps) to let browser decode large frames
void updateStream(void* ptr, const unsigned char* jpegData, int jpegSize) {
  struct StreamContext* s = (struct StreamContext*)ptr;
  if (!s || !s->active) return;
  int cfd = s->clientFd;
  if (cfd < 0) return;

  char frame[128];
  int flen = snprintf(frame, sizeof(frame),
    "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n", jpegSize);

  // Build single buffer: header + jpeg + CRLF
  size_t total = flen + jpegSize + 2;
  char* buf = (char*)alloca(total);
  memcpy(buf, frame, flen);
  memcpy(buf + flen, jpegData, jpegSize);
  buf[total - 2] = '\r';
  buf[total - 1] = '\n';

  ssize_t n = write(cfd, buf, total);
  if (n < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) return; // slow consumer, drop frame
    s->clientFd = -1; // real error (EPIPE etc.)
  } else if (n != (ssize_t)total) {
    // Partial write corrupted the stream — disconnect to force resync
    s->clientFd = -1;
  }
}

void destroyStream(void* ptr) {
  struct StreamContext* s = (struct StreamContext*)ptr;
  if (!s) return;
  s->active = 0;
  pthread_join(s->acceptThread, NULL);
  if (s->clientFd >= 0) close(s->clientFd);
  close(s->srvSock);
  unlink(s->sockPath);
  fprintf(stderr, "[Stream] Closed %s\n", s->sockPath);
  free(s);
}
