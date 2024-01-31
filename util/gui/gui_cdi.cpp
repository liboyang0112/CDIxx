#include "cdi.hpp"
extern "C"{
#include "gui/dataViewer.hpp"
}
int main(int argc, char** argv )
{
  DataviewerApp *app;
  app = dataviewer_app_new ();
  //g_signal_connect (app, "activate", G_CALLBACK (activate), NULL);
  g_application_run (G_APPLICATION (app), argc, argv);
  g_object_unref (app);
}
