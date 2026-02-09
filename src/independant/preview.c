#include <SDL2/SDL_thread.h>
#include <assert.h>
#include <string.h>
#include <SDL2/SDL.h>
struct PreviewContext {
  SDL_Window* window;
  SDL_Renderer* renderer;
  SDL_Texture* texture;
  SDL_Thread *renderThread;
  SDL_Event e;
  char active;
  int width, height;
};

void* createPreview(const char* windowname, int row, int col){
  struct PreviewContext* p = (struct PreviewContext*)malloc(sizeof(struct PreviewContext));
  p->width = row;
  p->height = col;
  assert(SDL_Init(SDL_INIT_VIDEO) == 0);
  p->window = SDL_CreateWindow(
      windowname,
      SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
      row > 1920 ? 1920 : row,
      col > 1080 ? 1080 : col,
      SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE
      );
  assert(p->window != 0);
  p->renderer = SDL_CreateRenderer(p->window, -1, SDL_RENDERER_ACCELERATED);
  p->texture = SDL_CreateTexture(p->renderer,SDL_PIXELFORMAT_RGB24,SDL_TEXTUREACCESS_STREAMING,row, col);
  p->active = 1;
  return p;
}
void updatePreview(void* ptr, void* buffer){  //buffer is rgb
  struct PreviewContext* p = (struct PreviewContext*)ptr;
  if(!p->active) return;
  SDL_PollEvent(&p->e);
  if (p->e.type == SDL_QUIT || p->e.type == SDL_KEYDOWN) {
    SDL_Quit();
  }
  void* pixels;
  int pitch;
  if (SDL_LockTexture(p->texture, NULL, &pixels, &pitch) == 0) {
    memcpy((uint8_t*)pixels,buffer, p->height * p->width * 3);
    SDL_UnlockTexture(p->texture);
    SDL_RenderCopy(p->renderer, p->texture, NULL, NULL);
    SDL_RenderPresent(p->renderer);
  }
};
