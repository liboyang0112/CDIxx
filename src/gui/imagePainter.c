#include "gui/imagePainter.hpp"
#include "gui/dataviewerWinWrapper.hpp"  // assuming this defines readImage_c etc.
#include "imgio.hpp"
// Instance structure
struct _ImagePainter {
  GtkBox parent_instance;

  GtkImage *img;         // child image widget
  void *d_image;         // GPU-side data (optional)
  guchar *pixel_data;    // RGB pixel buffer (owned)
  Mode m;
  int isFrequency;
  int islog;
  int isFlip;
  int isColor;
  Real decay;
};

// Class structure
struct _ImagePainterClass {
  GtkBoxClass parent_class;

  // No need for GApplication stuff here!
};

G_DEFINE_TYPE(ImagePainter, image_painter, GTK_TYPE_BOX)

static void image_painter_finalize(GObject *object) {
  ImagePainter *self = IMAGE_PAINTER(object);

  if (self->pixel_data) {
    g_free(self->pixel_data);
    self->pixel_data = NULL;
  }

//  if (self->d_image) {
//    from_gpu(self->d_image);  // free GPU memory
//    self->d_image = NULL;
//  }

  G_OBJECT_CLASS(image_painter_parent_class)->finalize(object);
}

static void image_painter_init(ImagePainter *cp) {
  cp->img = GTK_IMAGE(g_object_new(GTK_TYPE_IMAGE, "pixel-size", 500, NULL));
  gtk_box_append(GTK_BOX(cp), GTK_WIDGET(cp->img));

  // Initialize defaults
  cp->d_image = NULL;
  cp->pixel_data = NULL;
  cp->m = MOD;
  cp->isFrequency = 0;
  cp->islog = 1;
  cp->isFlip = 0;
  cp->isColor = 1;
  cp->decay = 1.0;
}

static void image_painter_class_init(ImagePainterClass *klass) {
  GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
  gobject_class->finalize = image_painter_finalize;
}

// Public constructor
ImagePainter* image_painter_new(const char* fname) {
  g_return_val_if_fail(fname != NULL, NULL);

  ImagePainter *cp = g_object_new(IMAGE_TYPE_PAINTER, NULL);

  struct imageFile f;
  void *image = readImage_c(fname, &f, 0);
  if (!image) {
    g_warning("Failed to load image: %s", fname);
    gtk_widget_set_visible(GTK_WIDGET(cp), TRUE);
    return cp;
  }
  cp->d_image = to_gpu(image, &f);
  free(image);  // free CPU copy
  cp->pixel_data = g_malloc_n(f.rows, f.cols*3);
  if (!cp->pixel_data) {
    g_warning("Out of memory allocating pixel buffer");
    gtk_widget_set_visible(GTK_WIDGET(cp), TRUE);
    return cp;
  }
  // Process image into RGB
  if (f.type == REALIDX) {
    processFloat(cp->pixel_data, cp->d_image, cp->m, cp->isFrequency, cp->decay,
                 cp->islog, cp->isFlip, cp->isColor);
  } else if (f.type == COMPLEXIDX) {
    processComplex(cp->pixel_data, cp->d_image, cp->m, cp->isFrequency, cp->decay,
                   cp->islog, cp->isFlip, cp->isColor);
  } else {
    g_warning("Unsupported image type: %d", f.type);
    gtk_widget_set_visible(GTK_WIDGET(cp), TRUE);
    return cp;
  }
  GBytes *bytes = g_bytes_new_with_free_func(
      cp->pixel_data,
      f.rows*f.cols*3,
      g_free,
      cp->pixel_data
  );
  // ✅ Step 2: Create GdkTexture from GBytes
  GdkTexture *texture = gdk_memory_texture_new(
      f.cols,
      f.rows,
      GDK_MEMORY_R8G8B8,   // 3-byte RGB
      bytes,               // 👈 now correct type
      f.cols*3
  );
  // ✅ Step 3: GdkTexture is already a GdkPaintable → just cast it
  GdkPaintable *paintable = GDK_PAINTABLE(texture);
  // ✅ Step 4: Set on GtkImage
  gtk_image_set_from_paintable(cp->img, paintable);
  // ✅ Cleanup references
  // GtkImage now holds ref to paintable → unref local ones
  g_object_unref(texture);  // drops one ref; paintable keeps alive
  g_bytes_unref(bytes);     // texture holds reference to bytes
  gtk_widget_set_visible(GTK_WIDGET(cp), TRUE);
  g_message("Loaded image: %dx%d from '%s'", f.cols, f.rows, fname);
  return cp;
}
