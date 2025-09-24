#ifndef __IMAGEPAINTER__
#define __IMAGEPAINTER__

#include <gtk/gtk.h>

G_BEGIN_DECLS

#define IMAGE_TYPE_PAINTER (image_painter_get_type())
G_DECLARE_FINAL_TYPE(ImagePainter, image_painter, IMAGE, PAINTER, GtkBox)

typedef enum {
  MOD2,
  MOD,
  REAL,
  IMAG,
  PHASE,
  PHASERAD
} Mode;

ImagePainter* image_painter_new(const char* fname);

G_END_DECLS

#endif // __IMAGEPAINTER__
