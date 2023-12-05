/* example1.c                                                      */
/*                                                                 */
/* This small program shows how to print a rotated string with the */
/* FreeType 2 library.                                             */


#include <stdio.h>
#include <string.h>
#include <math.h>
#include <freetype/freetype.h>
#include "freetype.h"
#include "imgio.h"
#include <stdint.h>

/* 这里修改 原来是680 480 太大 */


/* origin is the upper left corner */
uint16_t image[HEIGHT][WIDTH];


/* Replace this function with something useful. */

  void
draw_bitmap( FT_Bitmap* bitmap, FT_Int x, FT_Int y)
{
  FT_Int  i, j, p, q;
  FT_Int  x_max = x + bitmap->width;
  FT_Int  y_max = y + bitmap->rows;


  /* for simplicity, we assume that `bitmap->pixel_mode' */
  /* is `FT_PIXEL_MODE_GRAY' (i.e., not a bitmap font)   */

  for ( i = x, p = 0; i < x_max; i++, p++ )
  {
    for ( j = y, q = 0; j < y_max; j++, q++ )
    {
      if ( i < 0 || j < 0 || i >= WIDTH || j >= HEIGHT )
        continue;

      image[j][i] |= bitmap->buffer[q * bitmap->width + p];
    }
  }
  writePng("freetypetest_s.png", bitmap->buffer, bitmap->rows, bitmap->width, 8, 0);
}


void show_image( void )
{
  int  i, j;


  for ( i = 0; i < HEIGHT; i++ )
  {
    for ( j = 0; j < WIDTH; j++ )
      image[j][i]  *= 255;
  }
  writePng("freetypetest.png", image, HEIGHT, WIDTH, 16, 0);
}


