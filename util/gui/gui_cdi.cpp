#include "cdi.hpp"
extern "C"{
#include "gui/dataViewer.hpp"
}
int main(int argc, char** argv )
{
  DataviewerApp *app;
  int status;
  app = dataviewer_app_new ();
  //g_signal_connect (app, "activate", G_CALLBACK (activate), NULL);
  status = g_application_run (G_APPLICATION (app), argc, argv);
  g_object_unref (app);
}
