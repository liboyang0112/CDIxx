#include <gtk/gtk.h>
#include <stdio.h>
#include "gui/dataviewerWin.hpp"
#include "gui/cairoPainter.hpp"

struct _DataviewerWindow{
  GtkApplicationWindow parent;
  GtkWidget *imgbox;
  GtkWidget *l1box;
};

G_DEFINE_TYPE(DataviewerWindow, dataviewer_window, GTK_TYPE_APPLICATION_WINDOW)
static void dataviewer_window_class_init(DataviewerWindowClass *class){
  gtk_widget_class_set_template_from_resource (GTK_WIDGET_CLASS (class),"/org/gtk/dataViewer/window.ui");
  gtk_widget_class_bind_template_child(GTK_WIDGET_CLASS (class), DataviewerWindow, imgbox);
  gtk_widget_class_bind_template_child(GTK_WIDGET_CLASS (class), DataviewerWindow, l1box);
}
static void dataviewer_window_init(DataviewerWindow *win){
  gtk_widget_init_template (GTK_WIDGET (win));
}

DataviewerWindow* dataviewer_window_new(DataviewerApp* app){
  DataviewerWindow* win = g_object_new (DATAVIEWER_WINDOW_TYPE, "application", app, NULL);
  return win;
}
void dataviewer_window_open(DataviewerWindow* win, GFile* file){
  char* path = g_file_get_path(file);
  CairoPainter* cairopainter = cairo_painter_new(path);
  gtk_box_append(GTK_BOX(win->imgbox), GTK_WIDGET(cairopainter));
}
void gtk_dataviewer_window_save(DataviewerWindow* win){
  GtkSnapshot* ss = gtk_snapshot_new();
  gtk_widget_snapshot_child(GTK_WIDGET(win->l1box), win->imgbox, ss);
}
