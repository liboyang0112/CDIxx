#include <stdio.h>
#include <string.h>
#include <math.h>
#include <freetype/freetype.h>
#include "freetype.hpp"
#include "imgio.hpp"
#include <stdint.h>
#include <math.h>
//const char* fontfile= "/usr/share/fonts/msttcore/timesbi.ttf";
const char* fontfile= "/usr/share/fonts/gsfonts/NimbusRoman-BoldItalic.otf";
//const char* fontfile= "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold_Italic.ttf";
static FT_Library    library;
static FT_Face       face = 0;
int putText(const char* text, int initx, int inity, int rows, int cols, void* data, char iscolor, void* rgb)
{


  int num_chars = strlen( text );
  double angle = ( 00.0 / 360 ) * 3.14159 * 2;      /* use 25 degrees     */

  FT_Init_FreeType( &library );              /* initialize library */
  FT_New_Face( library, fontfile, 0, &face );/* create face object */
  FT_Set_Char_Size(face, 0, 10*64, 0,200 );                /* set character size */
  if(!face){
    fprintf(stderr, "Failed to initialize font face, please check if you have the font:\n%s\nIf not, please use another font.\n", fontfile);
    abort();
  }

  FT_GlyphSlot  slot = face->glyph;
  FT_Matrix     matrix;                 /* transformation matrix */
  matrix.xx = (FT_Fixed)( cos( angle ) * 0x10000L );
  matrix.xy = (FT_Fixed)(-sin( angle ) * 0x10000L );
  matrix.yx = (FT_Fixed)( sin( angle ) * 0x10000L );
  matrix.yy = (FT_Fixed)( cos( angle ) * 0x10000L );
  FT_Vector pen = {initx,inity};                    /* untransformed origin  */

  for (int n = 0; n < num_chars; n++ )
  {
    FT_Set_Transform( face, &matrix, &pen );  //rotate + translation
    FT_Load_Char( face, text[n], FT_LOAD_RENDER );
    int y = inity-slot->bitmap_top;
    int x = slot->bitmap_left;
    for (int q = 0; q < slot->bitmap.rows; q++ )
    {
      for (int p = 0; p < slot->bitmap.width; p++ )
      {
        if (p+x < 0 || q+y < 0 || p+x >= rows || q+y >= cols )
          continue;
        if(slot->bitmap.buffer[q * slot->bitmap.width + p] > 128){
          if(iscolor){
            for(int ic = 0; ic < 3; ic++) ((unsigned char*)data)[3*(rows*(q+y)+p+x)+ic] = ((unsigned char*)rgb)[ic];
          }else{
            ((pixeltype*)data)[rows*(q+y)+p+x] = *(pixeltype*)rgb;
          }
        }
      }
    }
    pen.x += slot->advance.x;
    pen.y += slot->advance.y;
  }
  FT_Done_Face    ( face );
  FT_Done_FreeType( library );
  return 0;
}
