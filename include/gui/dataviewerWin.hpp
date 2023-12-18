#ifndef __DATAVIEWERWIN_H
#define __DATAVIEWERWIN_H

#include <gtk/gtk.h>
#include "dataViewer.hpp"

#define DATAVIEWER_WINDOW_TYPE (dataviewer_window_get_type ())
G_DECLARE_FINAL_TYPE (DataviewerWindow, dataviewer_window, DATAVIEWER, WINDOW, GtkApplicationWindow)

DataviewerWindow       *dataviewer_window_new          (DataviewerApp *app);
void                    dataviewer_window_open         (DataviewerWindow *win, GFile* file);

#endif /* __DATAVIEWERWIN_H */
