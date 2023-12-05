extern "C" {
  #include "freetype.h"
  #include <freetype/freetype.h>
}
#include <math.h>
int main( int argc, char**  argv )
{
  FT_Library    library;
  FT_Face       face;

  FT_GlyphSlot  slot;
  FT_Matrix     matrix;                 /* transformation matrix */
  FT_Vector     pen;                    /* untransformed origin  */
  FT_Error      error;

  char*         text;

  double        angle;
  int           target_height;
  int           n, num_chars;


  if ( argc != 2 )
  {
    fprintf ( stderr, "usage: %s sample-text\n", argv[0] );
    exit( 1 );
  }

  const char* filename= "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf";                           /* first argument     */
  text          = argv[1];                           /* second argument    */
  num_chars     = strlen( text );
  angle         = ( 00.0 / 360 ) * 3.14159 * 2;      /* use 25 degrees     */
  target_height = HEIGHT;

  error = FT_Init_FreeType( &library );              /* initialize library */
  /* error handling omitted */

  error = FT_New_Face( library, filename, 0, &face );/* create face object */
  /* error handling omitted */
#if 1
  /* use 50pt at 100dpi */
  error = FT_Set_Char_Size( face, 10 * 64, 0,
      0,500 );                /* set character size */
  /* error handling omitted */
#else
  error = FT_Set_Pixel_Sizes(
      face,   /* handle to face object */
      0,      /* pixel_width           */
      100 );   /* pixel_height          */    
#endif
  /* cmap selection omitted;                                        */
  /* for simplicity we assume that the font contains a Unicode cmap */

  slot = face->glyph;

  /* set up matrix */
  matrix.xx = (FT_Fixed)( cos( angle ) * 0x10000L );
  matrix.xy = (FT_Fixed)(-sin( angle ) * 0x10000L );
  matrix.yx = (FT_Fixed)( sin( angle ) * 0x10000L );
  matrix.yy = (FT_Fixed)( cos( angle ) * 0x10000L );

  /* the pen position in 26.6 cartesian space coordinates; */
  /* start at (300,200) relative to the upper left corner  */
  /* 这里也要改 因为上面改了 */
  pen.x = 0 * 64;
  pen.y = target_height/2 * 64;

  for ( n = 0; n < num_chars; n++ )
  {
    /* set transformation */
    FT_Set_Transform( face, &matrix, &pen );  //rotate + translation

    /* load glyph image into the slot (erase previous one) */
    FT_Load_Char( face, text[n], FT_LOAD_RENDER );

    /* now, draw to our target surface (convert position) */
    draw_bitmap( &slot->bitmap,
        slot->bitmap_left,
        target_height - slot->bitmap_top );

    /* increment pen position */
    pen.x += slot->advance.x;
    pen.y += slot->advance.y;
  }

  show_image();

  FT_Done_Face    ( face );
  FT_Done_FreeType( library );

  return 0;
}
