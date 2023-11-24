#include <gtk/gtk.h>
#include <stdio.h>
#include "gui/dataviewerWin.h"
#include "gui/cairoPainter.h"

struct _DataviewerWindow{
  GtkApplicationWindow parent;
  GtkWidget *box;
};

G_DEFINE_TYPE(DataviewerWindow, dataviewer_window, GTK_TYPE_APPLICATION_WINDOW)
static void dataviewer_window_class_init(DataviewerWindowClass *class){
   gtk_widget_class_set_template_from_resource (GTK_WIDGET_CLASS (class),"/org/gtk/dataViewer/window.ui");
}
static void dataviewer_window_init(DataviewerWindow *win){
  gtk_widget_init_template (GTK_WIDGET (win));
}

DataviewerWindow* dataviewer_window_new(DataviewerApp* app){
  DataviewerWindow* win = g_object_new (DATAVIEWER_WINDOW_TYPE, "application", app, NULL);
  win->box = gtk_box_new (GTK_ORIENTATION_HORIZONTAL, 0);
  gtk_widget_set_halign (win->box, GTK_ALIGN_CENTER);
  gtk_widget_set_valign (win->box, GTK_ALIGN_CENTER);
  gtk_window_set_child(GTK_WINDOW(win), win->box);
  return win;
}
void dataviewer_window_open(DataviewerWindow* win, GFile* file){
  char* path = g_file_get_path(file);
  GtkBox* cairopainter = cairo_painter_new(path);
  gtk_box_append(GTK_BOX(win->box), GTK_WIDGET(cairopainter));
}
