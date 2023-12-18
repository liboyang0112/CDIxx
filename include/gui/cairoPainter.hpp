#ifndef __CAIROPAINTER__
#define __CAIROPAINTER__
#include <gtk/gtk.h>
#define GTK_TYPE_CAIRO_PAINTER (cairo_painter_get_type ())
G_DECLARE_FINAL_TYPE(CairoPainter,cairo_painter,CAIRO,PAINTER,GtkBox)
GtkBox* cairo_painter_new(const char*);
#endif
