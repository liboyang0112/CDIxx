// SDL2-based image viewer (replaces the former GTK/GDK implementation).
// Pipeline: readImage_c -> to_gpu -> processFloat/processComplex
// (cuPlotter CUDA) -> RGB24 buffer -> SDL texture.
// Controls: q close | drag pan | scroll zoom
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <stdio.h>
#include <math.h>
#include <vector>

#include "cuPlotter.hpp"                // enum mode (MOD)
#include "cudaConfig.hpp"               // myFFT, init_fft, cudaConvertFO, applyNorm
#include "imgio.hpp"
extern "C" {
#include "gui/dataviewerWinWrapper.hpp"  // to_gpu, processFloat, processComplex
}

static const int BAR_H = 40;

struct View {
  SDL_Window*   win  = NULL;
  SDL_Renderer* rend = NULL;
  SDL_Texture*  tex  = NULL;
  std::vector<unsigned char> rgb;
  bool   open   = false;
  float  zoom   = 1.0f;   // 1.0 = fit-to-window
  float  pan_x  = 0.0f;   // pan offset (pixels, in window space)
  float  pan_y  = 0.0f;
  bool   dragging = false;
  int    drag_last_x = 0, drag_last_y = 0;
  // control bar state
  void*  d_image  = NULL;
  int    img_rows = 0, img_cols = 0, img_type = 0;
  float  decay    = 1.0f;
  float  slider_pos = 0.5f;
  bool   islog    = true;
  bool   iscolor  = true;
  bool   fft      = false;
  bool   fft_dirty = false;
  void*  d_fft    = NULL;
  bool   slider_dragging = false;
  TTF_Font* font = NULL;
  int    mouse_x = 0, mouse_y = 0;
};

static void reprocess(View* v) {
  if (v->fft) {
    if (v->fft_dirty) {
      if (v->img_type == REALIDX) {
        extendToComplex((Real*)v->d_image, (complexFormat*)v->d_fft);
        myFFT(v->d_fft, v->d_fft);
      } else {
        myFFT(v->d_image, v->d_fft);
      }
      cudaConvertFO((complexFormat*)v->d_fft);
      applyNorm((complexFormat*)v->d_fft, 1.0f / (v->img_rows * v->img_cols));
      v->fft_dirty = false;
    }
    processComplex(v->rgb.data(), v->d_fft, MOD, 0, v->decay,
                   v->islog ? 1 : 0, 0, v->iscolor ? 1 : 0);
  } else {
    if (v->img_type == REALIDX)
      processFloat(v->rgb.data(), v->d_image, MOD, 0, v->decay,
                   v->islog ? 1 : 0, 0, v->iscolor ? 1 : 0);
    else if (v->img_type == COMPLEXIDX)
      processComplex(v->rgb.data(), v->d_image, MOD, 0, v->decay,
                     v->islog ? 1 : 0, 0, v->iscolor ? 1 : 0);
  }
  SDL_UpdateTexture(v->tex, NULL, v->rgb.data(), v->img_cols * 3);
}

static void render_text(View* v, const char* text, int cx, int cy,
                        Uint8 r, Uint8 g, Uint8 b) {
  if (!v->font) return;
  SDL_Color c{r, g, b, 255};
  SDL_Surface* surf = TTF_RenderText_Solid(v->font, text, c);
  if (!surf) return;
  SDL_Texture* tex = SDL_CreateTextureFromSurface(v->rend, surf);
  if (tex) {
    int tw = (int)surf->w, th = (int)surf->h;
    SDL_Rect dst{cx - tw / 2, cy - th / 2, tw, th};
    SDL_RenderCopy(v->rend, tex, NULL, &dst);
    SDL_DestroyTexture(tex);
  }
  SDL_FreeSurface(surf);
}

static void draw_bar(View* v, int w, int h) {
  int bar_y    = h - BAR_H;
  int slider_x = 8;
  int slider_w = (int)(w * 0.4);
  int btn_h    = 28;
  int btn1_x   = slider_x + slider_w + 12;
  int btn2_x   = btn1_x + 80 + 12;
  int btn_y    = bar_y + (BAR_H - btn_h) / 2;
  int track_y  = bar_y + (BAR_H - 16) / 2;

  // bar background
  SDL_Rect bar{0, bar_y, w, BAR_H};
  SDL_SetRenderDrawColor(v->rend, 40, 40, 40, 255);
  SDL_RenderFillRect(v->rend, &bar);

  // slider track
  SDL_Rect track{slider_x, track_y, slider_w, 16};
  SDL_SetRenderDrawColor(v->rend, 80, 80, 80, 255);
  SDL_RenderFillRect(v->rend, &track);

  // slider thumb
  int thumb_w = 12;
  int thumb_x = slider_x + (int)(v->slider_pos * (slider_w - thumb_w));
  SDL_Rect thumb{thumb_x, track_y, thumb_w, 16};
  SDL_SetRenderDrawColor(v->rend, 200, 200, 200, 255);
  SDL_RenderFillRect(v->rend, &thumb);

  // helper: draw a raised button with border
  auto draw_button = [&](int x, int y, int bw, int bh,
                         int fill_r, int fill_g, int fill_b,
                         const char* label) {
    SDL_Rect r{x, y, bw, bh};
    // border (darker)
    SDL_SetRenderDrawColor(v->rend, 20, 20, 20, 255);
    SDL_RenderFillRect(v->rend, &r);
    // top+left highlight
    SDL_SetRenderDrawColor(v->rend, (Uint8)(fill_r+40), (Uint8)(fill_g+40), (Uint8)(fill_b+40), 255);
    SDL_RenderDrawLine(v->rend, x, y, x + bw, y);
    SDL_RenderDrawLine(v->rend, x, y, x, y + bh);
    // bottom+right shadow
    SDL_SetRenderDrawColor(v->rend, 10, 10, 10, 255);
    SDL_RenderDrawLine(v->rend, x, y + bh - 1, x + bw, y + bh - 1);
    SDL_RenderDrawLine(v->rend, x + bw - 1, y, x + bw - 1, y + bh - 1);
    // fill
    SDL_Rect inner{x + 2, y + 2, bw - 4, bh - 4};
    SDL_SetRenderDrawColor(v->rend, (Uint8)fill_r, (Uint8)fill_g, (Uint8)fill_b, 255);
    SDL_RenderFillRect(v->rend, &inner);
    // label
    render_text(v, label, x + bw / 2, y + bh / 2, 255, 255, 255);
  };

  // Log button
  if (v->islog)
    draw_button(btn1_x, btn_y, 80, btn_h, 40, 120, 40, "Log");
  else
    draw_button(btn1_x, btn_y, 80, btn_h, 70, 70, 70, "Log");

  // Color button
  if (v->iscolor)
    draw_button(btn2_x, btn_y, 80, btn_h, 100, 60, 140, "Color");
  else
    draw_button(btn2_x, btn_y, 80, btn_h, 70, 70, 70, "Color");

  // FFT button
  int btn3_x = btn2_x + 80 + 12;
  if (v->fft)
    draw_button(btn3_x, btn_y, 80, btn_h, 40, 40, 140, "FFT");
  else
    draw_button(btn3_x, btn_y, 80, btn_h, 70, 70, 70, "FFT");

  // cursor coordinates (image pixels, accounting for zoom/pan)
  float base = SDL_min((float)w / v->img_cols, (float)(h - BAR_H) / v->img_rows);
  float s = base * v->zoom;
  float dx = ((float)w - v->img_cols * s) / 2.0f + v->pan_x;
  float dy = ((float)(h - BAR_H) - v->img_rows * s) / 2.0f + v->pan_y;
  int px = (int)((v->mouse_x - dx) / s);
  int py = (int)((v->mouse_y - dy) / s);
  if (px < 0 || px >= v->img_cols || py < 0 || py >= v->img_rows)
    px = py = -1;
  char coords[64];
  snprintf(coords, sizeof(coords), "%d, %d", px, py);
  render_text(v, coords, btn2_x + 80 + 12 + 80 + 60, btn_y + btn_h / 2, 200, 200, 200);
}

static void render_view(View* v) {
  int  w, h, tw, th;
  float base, s;
  int   dw, dh, dx, dy;

  SDL_GetRendererOutputSize(v->rend, &w, &h);
  SDL_QueryTexture(v->tex, NULL, NULL, &tw, &th);
  base = SDL_min((float)w / tw, (float)(h - BAR_H) / th);
  s    = base * v->zoom;
  dw   = (int)(tw * s);
  dh   = (int)(th * s);
  dx   = (w - dw) / 2 + (int)v->pan_x;
  dy   = ((h - BAR_H) - dh) / 2 + (int)v->pan_y;

  SDL_SetRenderDrawColor(v->rend, 0, 0, 0, 255);
  SDL_RenderClear(v->rend);

  // clip to image area
  SDL_Rect clip{0, 0, w, h - BAR_H};
  SDL_RenderSetClipRect(v->rend, &clip);
  SDL_Rect dst{dx, dy, dw, dh};
  SDL_RenderCopy(v->rend, v->tex, NULL, &dst);
  SDL_RenderSetClipRect(v->rend, NULL);

  draw_bar(v, w, h);
  SDL_RenderPresent(v->rend);
}

static void find_view(std::vector<View>& views, Uint32 id, View** out) {
  for (auto& v : views)
    if (v.open && v.win && SDL_GetWindowID(v.win) == id) { *out = &v; return; }
  *out = NULL;
}

static bool init_font(View* v) {
  TTF_Init();
  v->font = TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16);
  return v->font != NULL;
}

static bool open_file(const char* fname, View* v) {
  struct imageFile f;
  void* image = readImage_c(fname, &f, 0);
  if (!image) {
    fprintf(stderr, "Failed to load image: %s\n", fname);
    return false;
  }
  void* d_image = to_gpu(image, &f);
  free(image);

  v->d_image  = d_image;
  v->img_rows = f.rows;  v->img_cols = f.cols;  v->img_type = f.type;
  v->decay    = 1.0f;
  v->slider_pos = 0.5f;
  v->islog    = true;
  v->iscolor  = true;
  v->fft      = false;
  init_fft(f.rows, f.cols);
  v->d_fft = memMngr.borrowCache((size_t)f.rows * f.cols * sizeof(complexFormat));

  v->rgb.resize((size_t)f.rows * f.cols * 3);
  if (f.type != REALIDX && f.type != COMPLEXIDX) {
    fprintf(stderr, "Unsupported image type: %d in '%s'\n", f.type, fname);
    return false;
  }

  int w = f.cols, h = f.rows;
  v->win = SDL_CreateWindow(fname, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                            w, h, SDL_WINDOW_RESIZABLE);
  if (!v->win) {
    fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError());
    return false;
  }
  v->rend = SDL_CreateRenderer(v->win, -1, SDL_RENDERER_ACCELERATED);
  if (!v->rend)
    v->rend = SDL_CreateRenderer(v->win, -1, 0);
  if (!v->rend) {
    fprintf(stderr, "SDL_CreateRenderer: %s\n", SDL_GetError());
    SDL_DestroyWindow(v->win);
    v->win = NULL;
    return false;
  }
  SDL_DisplayMode dm;
  if (SDL_GetDesktopDisplayMode(0, &dm) == 0) {
    if (w > dm.w) w = dm.w;
    if (h > dm.h) h = dm.h;
    SDL_SetWindowSize(v->win, w, h);
  }
  v->tex = SDL_CreateTexture(v->rend, SDL_PIXELFORMAT_RGB24,
                             SDL_TEXTUREACCESS_STREAMING, f.cols, f.rows);
  if (!v->tex) {
    fprintf(stderr, "SDL_CreateTexture: %s\n", SDL_GetError());
    SDL_DestroyRenderer(v->rend);
    SDL_DestroyWindow(v->win);
    v->rend = NULL;
    v->win  = NULL;
    return false;
  }
  if (!init_font(v))
    fprintf(stderr, "Warning: could not load font, buttons will be blank\n");
  reprocess(v);
  render_view(v);
  v->open = true;
  printf("Loaded image: %dx%d from '%s'\n", f.cols, f.rows, fname);
  return true;
}

static void close_view(View* v) {
  if (!v->open) return;
  if (v->tex)  SDL_DestroyTexture(v->tex);
  if (v->d_fft) myCuFree(v->d_fft);
  if (v->font) TTF_CloseFont(v->font);
  if (v->rend) SDL_DestroyRenderer(v->rend);
  if (v->win)  SDL_DestroyWindow(v->win);
  v->tex  = NULL;
  v->font = NULL;
  v->rend = NULL;
  v->win  = NULL;
  v->open = false;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <image> [<image>...]\n", argv[0]);
    fprintf(stderr, "controls: q close | drag pan | scroll zoom\n");
    return 1;
  }
  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
    return 1;
  }

  std::vector<View> views;
  for (int i = 1; i < argc; i++) {
    View v;
    if (open_file(argv[i], &v))
      views.push_back(std::move(v));
  }
  if (views.empty()) { SDL_Quit(); return 1; }

  SDL_Event e;
  bool any_open = true;
  while (any_open) {
    if (!SDL_WaitEvent(&e)) break;

    // Find the view this event belongs to
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

    if (e.type == SDL_MOUSEMOTION && v) {
      v->mouse_x = e.motion.x;
      v->mouse_y = e.motion.y;
      if (!v->dragging && !v->slider_dragging)
        render_view(v);
    }

    if (e.type == SDL_QUIT) {
      for (auto& vw : views) close_view(&vw);
      any_open = false;

    } else if (e.type == SDL_KEYDOWN && v &&
               (e.key.keysym.sym == SDLK_q || e.key.keysym.sym == SDLK_ESCAPE)) {
      close_view(v);
      any_open = false;
      for (auto& vw : views) any_open |= vw.open;

    } else if (e.type == SDL_MOUSEBUTTONDOWN && v &&
               e.button.button == SDL_BUTTON_LEFT) {
      int w, h;
      SDL_GetRendererOutputSize(v->rend, &w, &h);
      int bar_y    = h - BAR_H;
      int slider_x = 8;
      int slider_w = (int)(w * 0.4);
      int btn1_x   = slider_x + slider_w + 12;
      int btn2_x   = btn1_x + 80 + 12;
      int btn_y    = bar_y + (BAR_H - 28) / 2;
      int btn_h    = 28;

      if (e.button.y >= bar_y) {
        // in control bar
        if (e.button.x >= slider_x && e.button.x <= slider_x + slider_w) {
          v->slider_dragging = true;
          float pos = (float)(e.button.x - slider_x) / slider_w;
          if (pos < 0) pos = 0;
          if (pos > 1) pos = 1;
          v->slider_pos = pos;
          v->decay = powf(10.0f, -2.0f + 4.0f * pos);
          reprocess(v);
          render_view(v);
        } else if (e.button.x >= btn1_x && e.button.x <= btn1_x + 80 &&
                   e.button.y >= btn_y && e.button.y <= btn_y + btn_h) {
          v->islog = !v->islog;
          reprocess(v);
          render_view(v);
        } else if (e.button.x >= btn2_x && e.button.x <= btn2_x + 80 &&
                   e.button.y >= btn_y && e.button.y <= btn_y + btn_h) {
          v->iscolor = !v->iscolor;
          reprocess(v);
          render_view(v);
        } else if (e.button.x >= btn2_x + 80 + 12 && e.button.x <= btn2_x + 80 + 12 + 80 &&
                   e.button.y >= btn_y && e.button.y <= btn_y + btn_h) {
          v->fft = !v->fft;
          v->fft_dirty = true;
          reprocess(v);
          render_view(v);
        }
      } else {
        v->dragging    = true;
        v->drag_last_x = e.button.x;
        v->drag_last_y = e.button.y;
        SDL_CaptureMouse(SDL_TRUE);
      }

    } else if (e.type == SDL_MOUSEMOTION && v && v->slider_dragging) {
      int w, h;
      SDL_GetRendererOutputSize(v->rend, &w, &h);
      int slider_x = 8;
      int slider_w = (int)(w * 0.4);
      float pos = (float)(e.motion.x - slider_x) / slider_w;
      if (pos < 0) pos = 0;
      if (pos > 1) pos = 1;
      v->slider_pos = pos;
      v->decay = powf(10.0f, -2.0f + 4.0f * pos);
      reprocess(v);
      render_view(v);

    } else if (e.type == SDL_MOUSEMOTION && v && v->dragging) {
      v->pan_x += e.motion.x - v->drag_last_x;
      v->pan_y += e.motion.y - v->drag_last_y;
      v->drag_last_x = e.motion.x;
      v->drag_last_y = e.motion.y;
      render_view(v);

    } else if (e.type == SDL_MOUSEBUTTONUP && v &&
               e.button.button == SDL_BUTTON_LEFT) {
      v->dragging = false;
      v->slider_dragging = false;
      SDL_CaptureMouse(SDL_FALSE);

    } else if (e.type == SDL_MOUSEWHEEL && v) {
      const float k = 1.1f;
      if (e.wheel.y > 0)
        v->zoom *= k;
      else if (e.wheel.y < 0)
        v->zoom /= k;
      if (v->zoom < 0.1f)  v->zoom = 0.1f;
      if (v->zoom > 50.0f) v->zoom = 50.0f;
      render_view(v);

    } else if (e.type == SDL_WINDOWEVENT &&
               e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED && v) {
      render_view(v);

    } else if (e.type == SDL_WINDOWEVENT &&
               e.window.event == SDL_WINDOWEVENT_CLOSE && v) {
      close_view(v);
      any_open = false;
      for (auto& vw : views) any_open |= vw.open;
    }
  }
  for (auto& vw : views) close_view(&vw);
  SDL_Quit();
  return 0;
}
