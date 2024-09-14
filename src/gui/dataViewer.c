#include <gtk/gtk.h>
#include "gui/dataViewer.hpp"
#include "gui/dataviewerWin.hpp"

struct _DataviewerApp{
  GtkApplication parent;
  void* dummy;
};
G_DEFINE_TYPE(DataviewerApp, dataviewer_app, GTK_TYPE_APPLICATION);

static void dataviewer_app_startup(GApplication* app){
  G_APPLICATION_CLASS (dataviewer_app_parent_class)->startup (app);
}

static void dataviewer_app_activate(GApplication* app){
  DataviewerWindow *win = dataviewer_window_new (DATAVIEWER_APP (app));
  gtk_window_present (GTK_WINDOW (win));
}

static void dataviewer_app_open(GApplication* app, GFile** file, int nfile, const char* hint){
  GList *windows;
  DataviewerWindow *win;
  windows = gtk_application_get_windows (GTK_APPLICATION (app));
  if (windows)
    win = DATAVIEWER_WINDOW (windows->data);
  else
    win = dataviewer_window_new (DATAVIEWER_APP(app));
  for(int i = 0; i < nfile; i++)
    dataviewer_window_open (win, file[i]);
  gtk_window_present (GTK_WINDOW (win));
  if(hint) printf("hint=%s", hint);
  //gtk_dataviewer_window_save(win);
}

static void dataviewer_app_class_init (DataviewerAppClass *class)
{
  G_APPLICATION_CLASS (class)->startup = dataviewer_app_startup;
  G_APPLICATION_CLASS (class)->activate = dataviewer_app_activate;
  G_APPLICATION_CLASS (class)->open = dataviewer_app_open;
}
static void dataviewer_app_init(DataviewerApp* app){
    app->dummy = 0;
}

DataviewerApp* dataviewer_app_new()
{
  return g_object_new (DATAVIEWER_APP_TYPE,"application-id","org.gtk.dataviewerapp","flags",
      G_APPLICATION_HANDLES_OPEN,NULL);
}
