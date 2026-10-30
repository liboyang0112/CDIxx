// SDL2 image viewer. Pipeline: readImage_c -> to_gpu -> processFloat/processComplex -> RGB24 -> SDL texture.
// Controls: q close | drag pan | scroll zoom
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include "cuPlotter.hpp"
#include "cudaConfig.hpp"
#include "imgio.hpp"
extern "C" {
#include "gui/dataviewerWinWrapper.hpp"
}

static const int BAR_H = 40;
static TTF_Font* g_font = NULL;

struct BarLayout {
  int bar_y, slider_x, slider_w, btn_x[3], btn_y;
  static constexpr int btn_w = 80, btn_h = 28;
  static BarLayout compute(int w, int h) {
    BarLayout l;
    l.bar_y = h - BAR_H;
    l.slider_x = 8;
    l.slider_w = (int)(w * 0.4);
    l.btn_x[0] = l.slider_x + l.slider_w + 12;
    l.btn_x[1] = l.btn_x[0] + btn_w + 12;
    l.btn_x[2] = l.btn_x[1] + btn_w + 12;
    l.btn_y = l.bar_y + (BAR_H - btn_h) / 2;
    return l;
  }
  bool in_slider(int x, int y) const { return y >= bar_y && x >= slider_x && x <= slider_x + slider_w; }
  int  btn_at(int x, int y) const {
    for (int i = 0; i < 3; i++)
      if (x >= btn_x[i] && x <= btn_x[i] + btn_w && y >= btn_y && y <= btn_y + btn_h)
        return i;
    return -1;
  }
  float slider_pos_at(int x) const {
    float p = (float)(x - slider_x) / slider_w;
    return p < 0 ? 0 : (p > 1 ? 1 : p);
  }
};

struct View {
  SDL_Window* win = NULL;
  SDL_Renderer* rend = NULL;
  SDL_Texture* tex = NULL;
  std::vector<unsigned char> rgb;
  bool open = false;
  float zoom = 1.0f, pan_x = 0.0f, pan_y = 0.0f;
  bool dragging = false;
  int drag_last_x = 0, drag_last_y = 0;
  int mouse_x = 0, mouse_y = 0;
  void* d_image = NULL;
  int img_rows = 0, img_cols = 0, img_type = 0;
  float decay = 1.0f, slider_pos = 0.5f;
  bool islog = true, iscolor = true, fft = false, fft_dirty = false;
  void* d_fft = NULL;
  bool slider_dragging = false;
};

static void render_text(SDL_Renderer* rend, const char* text, int cx, int cy, Uint8 r, Uint8 g, Uint8 b) {
  if (!g_font) return;
  SDL_Surface* surf = TTF_RenderText_Solid(g_font, text, SDL_Color{r, g, b, 255});
  if (!surf) return;
  SDL_Texture* tex = SDL_CreateTextureFromSurface(rend, surf);
  if (tex) {
    SDL_Rect dst{cx - surf->w / 2, cy - surf->h / 2, (int)surf->w, (int)surf->h};
    SDL_RenderCopy(rend, tex, NULL, &dst);
    SDL_DestroyTexture(tex);
  }
  SDL_FreeSurface(surf);
}

static void draw_button(SDL_Renderer* rend, int x, int y, int bw, int bh,
                        int fr, int fg, int fb, const char* label) {
  SDL_Rect r{x, y, bw, bh};
  SDL_SetRenderDrawColor(rend, 20, 20, 20, 255);
  SDL_RenderFillRect(rend, &r);
  SDL_SetRenderDrawColor(rend, (Uint8)(fr+40), (Uint8)(fg+40), (Uint8)(fb+40), 255);
  SDL_RenderDrawLine(rend, x, y, x + bw, y);
  SDL_RenderDrawLine(rend, x, y, x, y + bh);
  SDL_SetRenderDrawColor(rend, 10, 10, 10, 255);
  SDL_RenderDrawLine(rend, x, y + bh - 1, x + bw, y + bh - 1);
  SDL_RenderDrawLine(rend, x + bw - 1, y, x + bw - 1, y + bh - 1);
  SDL_Rect inner{x + 2, y + 2, bw - 4, bh - 4};
  SDL_SetRenderDrawColor(rend, (Uint8)fr, (Uint8)fg, (Uint8)fb, 255);
  SDL_RenderFillRect(rend, &inner);
  render_text(rend, label, x + bw / 2, y + bh / 2, 255, 255, 255);
}

static void draw_bar(View* v, int w, int h) {
  BarLayout l = BarLayout::compute(w, h);
  SDL_Rect r;
  SDL_SetRenderDrawColor(v->rend, 40, 40, 40, 255);
  r = {0, l.bar_y, w, BAR_H}; SDL_RenderFillRect(v->rend, &r);
  int track_y = l.bar_y + (BAR_H - 16) / 2;
  SDL_SetRenderDrawColor(v->rend, 80, 80, 80, 255);
  r = {l.slider_x, track_y, l.slider_w, 16}; SDL_RenderFillRect(v->rend, &r);
  int thumb_x = l.slider_x + (int)(v->slider_pos * (l.slider_w - 12));
  SDL_SetRenderDrawColor(v->rend, 200, 200, 200, 255);
  r = {thumb_x, track_y, 12, 16}; SDL_RenderFillRect(v->rend, &r);
  draw_button(v->rend, l.btn_x[0], l.btn_y, BarLayout::btn_w, BarLayout::btn_h,
              v->islog ? 40 : 70, v->islog ? 120 : 70, v->islog ? 40 : 70, "Log");
  draw_button(v->rend, l.btn_x[1], l.btn_y, BarLayout::btn_w, BarLayout::btn_h,
              v->iscolor ? 100 : 70, v->iscolor ? 60 : 70, v->iscolor ? 140 : 70, "Color");
  draw_button(v->rend, l.btn_x[2], l.btn_y, BarLayout::btn_w, BarLayout::btn_h,
              v->fft ? 40 : 70, v->fft ? 40 : 70, v->fft ? 140 : 70, "FFT");
  // cursor coords in image pixels
  float base = SDL_min((float)w / v->img_cols, (float)(h - BAR_H) / v->img_rows) * v->zoom;
  float dx = ((float)w - v->img_cols * base) / 2.0f + v->pan_x;
  float dy = ((float)(h - BAR_H) - v->img_rows * base) / 2.0f + v->pan_y;
  int px = (int)((v->mouse_x - dx) / base), py = (int)((v->mouse_y - dy) / base);
  if (px < 0 || px >= v->img_cols || py < 0 || py >= v->img_rows) px = py = -1;
  char buf[32];
  snprintf(buf, sizeof(buf), "%d, %d", px, py);
  render_text(v->rend, buf, l.btn_x[2] + BarLayout::btn_w + 60, l.btn_y + BarLayout::btn_h / 2, 200, 200, 200);
}

static void reprocess(View* v) {
  if (v->fft) {
    if (v->fft_dirty) {
      if (v->img_type == REALIDX) {
        extendToComplex((Real*)v->d_image, (complexFormat*)v->d_fft);
        myFFT(v->d_fft, v->d_fft);
      } else myFFT(v->d_image, v->d_fft);
      cudaConvertFO((complexFormat*)v->d_fft);
      applyNorm((complexFormat*)v->d_fft, 1.0f / (v->img_rows * v->img_cols));
      v->fft_dirty = false;
    }
    processComplex(v->rgb.data(), v->d_fft, MOD, 0, v->decay, v->islog, 0, v->iscolor);
  } else if (v->img_type == REALIDX)
    processFloat(v->rgb.data(), v->d_image, MOD, 0, v->decay, v->islog, 0, v->iscolor);
  else if (v->img_type == COMPLEXIDX)
    processComplex(v->rgb.data(), v->d_image, MOD, 0, v->decay, v->islog, 0, v->iscolor);
  SDL_UpdateTexture(v->tex, NULL, v->rgb.data(), v->img_cols * 3);
}

static void render_view(View* v) {
  int w, h, tw, th;
  SDL_GetRendererOutputSize(v->rend, &w, &h);
  SDL_QueryTexture(v->tex, NULL, NULL, &tw, &th);
  float s = SDL_min((float)w / tw, (float)(h - BAR_H) / th) * v->zoom;
  int dw = (int)(tw * s), dh = (int)(th * s);
  int dx = (w - dw) / 2 + (int)v->pan_x;
  int dy = ((h - BAR_H) - dh) / 2 + (int)v->pan_y;
  SDL_SetRenderDrawColor(v->rend, 0, 0, 0, 255);
  SDL_RenderClear(v->rend);
  SDL_Rect r;
  r = {0, 0, w, h - BAR_H}; SDL_RenderSetClipRect(v->rend, &r);
  r = {dx, dy, dw, dh}; SDL_RenderCopy(v->rend, v->tex, NULL, &r);
  SDL_RenderSetClipRect(v->rend, NULL);
  draw_bar(v, w, h);
  SDL_RenderPresent(v->rend);
}

static void find_view(std::vector<View>& views, Uint32 id, View** out) {
  for (auto& v : views)
    if (v.open && v.win && SDL_GetWindowID(v.win) == id) { *out = &v; return; }
  *out = NULL;
}

static void close_view(View* v) {
  if (!v->open) return;
  if (v->tex) SDL_DestroyTexture(v->tex);
  if (v->d_fft) myCuFree(v->d_fft);
  if (v->rend) SDL_DestroyRenderer(v->rend);
  if (v->win) SDL_DestroyWindow(v->win);
  v->tex = NULL; v->rend = NULL; v->win = NULL;
  v->open = false;
}

static bool open_file(const char* fname, View* v) {
  struct imageFile f;
  void* image = readImage_c(fname, &f, 0);
  if (!image) { fprintf(stderr, "Failed to load: %s\n", fname); return false; }
  void* d_image = to_gpu(image, &f);
  free(image);
  v->d_image = d_image;
  v->img_rows = f.rows; v->img_cols = f.cols; v->img_type = f.type;
  init_fft(f.rows, f.cols);
  v->d_fft = memMngr.borrowCache((size_t)f.rows * f.cols * sizeof(complexFormat));
  v->rgb.resize((size_t)f.rows * f.cols * 3);
  if (f.type != REALIDX && f.type != COMPLEXIDX) {
    fprintf(stderr, "Unsupported type %d in '%s'\n", f.type, fname);
    return false;
  }
  int w = f.cols, h = f.rows;
  v->win = SDL_CreateWindow(fname, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, w, h, SDL_WINDOW_RESIZABLE);
  if (!v->win) { fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError()); return false; }
  v->rend = SDL_CreateRenderer(v->win, -1, SDL_RENDERER_ACCELERATED);
  if (!v->rend) v->rend = SDL_CreateRenderer(v->win, -1, 0);
  if (!v->rend) { fprintf(stderr, "SDL_CreateRenderer: %s\n", SDL_GetError()); SDL_DestroyWindow(v->win); v->win = NULL; return false; }
  SDL_DisplayMode dm;
  if (SDL_GetDesktopDisplayMode(0, &dm) == 0) {
    if (w > dm.w) w = dm.w;
    if (h > dm.h) h = dm.h;
    SDL_SetWindowSize(v->win, w, h);
  }
  v->tex = SDL_CreateTexture(v->rend, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, f.cols, f.rows);
  if (!v->tex) {
    fprintf(stderr, "SDL_CreateTexture: %s\n", SDL_GetError());
    SDL_DestroyRenderer(v->rend); SDL_DestroyWindow(v->win);
    v->rend = NULL; v->win = NULL;
    return false;
  }
  reprocess(v);
  render_view(v);
  v->open = true;
  printf("Loaded %dx%d '%s'\n", f.cols, f.rows, fname);
  return true;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <image> [...]\ncontrols: q close | drag pan | scroll zoom\n", argv[0]); return 1; }
  if (SDL_Init(SDL_INIT_VIDEO) != 0) { fprintf(stderr, "SDL_Init: %s\n", SDL_GetError()); return 1; }
  TTF_Init();
  g_font = TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16);
  if (!g_font) fprintf(stderr, "Warning: no font\n");

  std::vector<View> views;
  for (int i = 1; i < argc; i++) { View v; if (open_file(argv[i], &v)) views.push_back(std::move(v)); }
  if (views.empty()) { SDL_Quit(); return 1; }

  SDL_Event e;
  bool any_open = true;
  while (any_open) {
    if (!SDL_WaitEvent(&e)) break;
    int need = 0; // 0=none, 1=render, 2=reprocess
    View* rv = NULL;

    do {
      View* v = NULL;
      Uint32 wid = 0;
      switch (e.type) {
        case SDL_KEYDOWN:         wid = e.key.windowID;    break;
        case SDL_MOUSEMOTION:     wid = e.motion.windowID;  break;
        case SDL_MOUSEBUTTONDOWN: wid = e.button.windowID;  break;
        case SDL_MOUSEBUTTONUP:   wid = e.button.windowID;  break;
        case SDL_MOUSEWHEEL:      wid = e.wheel.windowID;   break;
        case SDL_WINDOWEVENT:     wid = e.window.windowID;  break;
        default: break;
      }
      if (wid) find_view(views, wid, &v);
      if (!v) continue;

      if (e.type == SDL_MOUSEMOTION) v->mouse_x = e.motion.x, v->mouse_y = e.motion.y;

      if (e.type == SDL_QUIT) {
        for (auto& vw : views) close_view(&vw);
        any_open = false;
        break;
      } else if (e.type == SDL_KEYDOWN && (e.key.keysym.sym == SDLK_q || e.key.keysym.sym == SDLK_ESCAPE)) {
        close_view(v);
      } else if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT) {
        int w, h; SDL_GetRendererOutputSize(v->rend, &w, &h);
        BarLayout l = BarLayout::compute(w, h);
        if (e.button.y >= l.bar_y) {
          if (l.in_slider(e.button.x, e.button.y)) {
            v->slider_dragging = true;
            v->slider_pos = l.slider_pos_at(e.button.x);
            v->decay = powf(10.0f, -2.0f + 4.0f * v->slider_pos);
            need = 2;
          } else if (l.btn_at(e.button.x, e.button.y) == 0) { v->islog = !v->islog; need = 2; }
          else if (l.btn_at(e.button.x, e.button.y) == 1) { v->iscolor = !v->iscolor; need = 2; }
          else if (l.btn_at(e.button.x, e.button.y) == 2) { v->fft = !v->fft; v->fft_dirty = true; need = 2; }
        } else {
          v->dragging = true; v->drag_last_x = e.button.x; v->drag_last_y = e.button.y;
          SDL_CaptureMouse(SDL_TRUE);
        }
      } else if (e.type == SDL_MOUSEMOTION && v->slider_dragging) {
        int w, h; SDL_GetRendererOutputSize(v->rend, &w, &h);
        BarLayout l = BarLayout::compute(w, h);
        v->slider_pos = l.slider_pos_at(e.motion.x);
        v->decay = powf(10.0f, -2.0f + 4.0f * v->slider_pos);
        need = 2;
      } else if (e.type == SDL_MOUSEMOTION && v->dragging) {
        v->pan_x += e.motion.x - v->drag_last_x;
        v->pan_y += e.motion.y - v->drag_last_y;
        v->drag_last_x = e.motion.x; v->drag_last_y = e.motion.y;
        need = 1;
      } else if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_LEFT) {
        v->dragging = v->slider_dragging = false;
        SDL_CaptureMouse(SDL_FALSE);
      } else if (e.type == SDL_MOUSEWHEEL) {
        const float k = 1.1f;
        v->zoom *= e.wheel.y > 0 ? k : (e.wheel.y < 0 ? 1.0f / k : 1.0f);
        v->zoom = SDL_max(0.1f, SDL_min(50.0f, v->zoom));
        need = 1;
      } else if (e.type == SDL_WINDOWEVENT) {
        if (e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) need = 1;
        else if (e.window.event == SDL_WINDOWEVENT_CLOSE) close_view(v);
      } else if (e.type == SDL_MOUSEMOTION) {
        need = 1; // coordinate update
      }
      if (need > 0) rv = v;
    } while (SDL_PollEvent(&e) && any_open);

    any_open = false;
    for (auto& vw : views) any_open |= vw.open;

    if (rv && rv->open) {
      if (need >= 2) reprocess(rv);
      render_view(rv);
    }
  }
  for (auto& vw : views) close_view(&vw);
  TTF_CloseFont(g_font);
  TTF_Quit();
  SDL_Quit();
  return 0;
}
