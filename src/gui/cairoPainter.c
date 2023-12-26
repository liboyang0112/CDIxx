#include "gui/cairoPainter.hpp"
#include "gui/dataviewerWinWrapper.hpp"
#include "imgio.hpp"
enum mode {MOD2,MOD, REAL, IMAG, PHASE, PHASERAD};
struct _CairoPainter{
  GtkBox parent;
  GtkImage* img;
  cairo_t *cr;
  cairo_matrix_t *mat;
  GdkPixbuf *buffer;
  GtkSnapshot* ss;
  GdkPaintable* paintable;
  void* d_image;
  enum mode m;
  char isFrequency;
  char islog;
  char isFlip;
  char isColor;
  Real decay;
};

static void cairo_painter_class_init(CairoPainterClass *class){
}
static void cairo_painter_init(CairoPainter *cp){
  cp->img = g_object_new(GTK_TYPE_IMAGE, "pixel-size", 500, NULL);
  gtk_box_append(GTK_BOX(cp), GTK_WIDGET(cp->img));
}

G_DEFINE_TYPE(CairoPainter, cairo_painter, GTK_TYPE_BOX)
CairoPainter* cairo_painter_new(const char* fname){
  CairoPainter *cp = g_object_new(GTK_TYPE_CAIRO_PAINTER, NULL);
  struct imageFile f;
  void* image = readImage_c(fname, &f, 0);
  cp->d_image = to_gpu(image, &f);
  free(image);
  void* cache = malloc(f.rows*f.cols*3);
  cp->decay = 1;
  cp->islog = 1;
  cp->m = MOD;
  cp->isColor = 1;
  if(f.type == REALIDX) processFloat(cache, cp->d_image, cp->m, cp->isFrequency, cp->decay, cp->islog, cp->isFlip, cp->isColor);
  if(f.type == COMPLEXIDX) processComplex(cache, cp->d_image, cp->m, cp->isFrequency, cp->decay, cp->islog, cp->isFlip, cp->isColor);
  cp->buffer = gdk_pixbuf_new_from_data(cache,GDK_COLORSPACE_RGB, 0, 8, f.rows, f.cols, 3*f.rows, 0, 0);
  cp->mat = (cairo_matrix_t *)malloc(sizeof(cairo_matrix_t));
  cairo_matrix_init_identity(cp->mat);
  cp->ss = gtk_snapshot_new();
  const graphene_rect_t bd = {0,0,f.rows,f.cols};
  cp->cr = gtk_snapshot_append_cairo (cp->ss, &bd);
  gdk_cairo_set_source_pixbuf(cp->cr, cp->buffer, 0, 0);
  cairo_paint(cp->cr);
  cp->paintable = gtk_snapshot_to_paintable(cp->ss,0);
  gtk_image_set_from_paintable(cp->img, GDK_PAINTABLE (cp->paintable));
  gtk_widget_set_visible (GTK_WIDGET (cp), TRUE);
  return cp;
}
/*
  GtkSnapshot* ss = gtk_snapshot_new();
  const graphene_rect_t bd = {0,0,f.rows,f.cols};
  cairo_t *cr = gtk_snapshot_append_cairo (ss, &bd);
  cairo_translate (cr, f.rows / 2.0, f.cols / 2.0);
  cairo_matrix_t mat = {-1, 0, 0, 1, 0, 0};
  //cairo_rotate (cr, M_PI/2);
  cairo_transform(cr, &mat);
  cairo_translate(cr, - f.rows / 2.0, - f.cols / 2.0);
  gdk_cairo_set_source_pixbuf(cr, buffer, 0, 0);
  cairo_paint(cr);
  GdkPaintable* paintable = gtk_snapshot_to_paintable(ss,0);
  */
