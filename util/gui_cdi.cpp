#include "cuPlotter.h"
#include "cdi.h"
#include <gtk/gtk.h>
using namespace std;
static void
activate (GtkApplication *app,
          gpointer        user_data)
{
  int argc;
  char** argv;
  GtkWidget *window;
  GtkWidget *button;
  GtkWidget *image;
  GtkWidget *box;
  button = gtk_button_new_with_label("Show object");

  window = gtk_application_window_new (app);
  gtk_window_set_title (GTK_WINDOW (window), "Window");
  gtk_window_set_default_size (GTK_WINDOW (window), 200, 200);

  box = gtk_box_new (GTK_ORIENTATION_VERTICAL, 0);
  gtk_widget_set_halign (box, GTK_ALIGN_CENTER);
  gtk_widget_set_valign (box, GTK_ALIGN_CENTER);

  //gtk_window_set_child (GTK_WINDOW (window), box);
  gtk_window_set_child (GTK_WINDOW (window), button);

  gtk_box_append (GTK_BOX (box), button);


  gtk_window_present (GTK_WINDOW (window));
}

int main(int argc, char** argv )
{
  GtkApplication *app;
  int status;
  app = gtk_application_new ("org.gtk.example", G_APPLICATION_DEFAULT_FLAGS);
  g_signal_connect (app, "activate", G_CALLBACK (activate), NULL);
  status = g_application_run (G_APPLICATION (app), argc, argv);
  g_object_unref (app);
}
