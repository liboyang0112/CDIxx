#ifndef __DATAVIEWER_H
#define __DATAVIEWER_H

#include <gtk/gtk.h>

#define DATAVIEWER_APP_TYPE (dataviewer_app_get_type ())
G_DECLARE_FINAL_TYPE (DataviewerApp, dataviewer_app, DATAVIEWER, APP, GtkApplication)
DataviewerApp *dataviewer_app_new();

#endif /* __DATAVIEWER_H */
