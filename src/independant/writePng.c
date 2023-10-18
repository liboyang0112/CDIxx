#include <stdlib.h>
#include <string.h>
#include <png.h>
#include <stdio.h>
png_colorp palette = 0;
int writePng(const char* png_file_name, void* data , int width, int height, int bit_depth, char colored)
{
  unsigned char* pixels = (unsigned char*) data;
  png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if(png_ptr == NULL)
	{
		printf("ERROR:png_create_write_struct/n");
		return 0;
	}
	png_infop info_ptr = png_create_info_struct(png_ptr);
	if(info_ptr == NULL)
	{
		printf("ERROR:png_create_info_struct/n");
		png_destroy_write_struct(&png_ptr, NULL);
		return 0;
	}
	FILE *png_file = fopen(png_file_name, "wb");
	if (!png_file)
	{
		return -1;
	}
	png_init_io(png_ptr, png_file);
	png_set_IHDR(png_ptr, info_ptr, width, height, bit_depth, colored?PNG_COLOR_TYPE_RGB:PNG_COLOR_TYPE_GRAY,
		PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
  if(colored){
    palette = (png_colorp)png_malloc(png_ptr, PNG_MAX_PALETTE_LENGTH * sizeof(png_color));
	  png_set_PLTE(png_ptr, info_ptr, palette, PNG_MAX_PALETTE_LENGTH);
  }
	png_write_info(png_ptr, info_ptr);
  if (bit_depth == 16)
    png_set_swap(png_ptr);
	png_set_packing(png_ptr);
	png_bytepp rows = (png_bytepp)png_malloc(png_ptr, height*sizeof(png_bytep));
	for (int i = 0; i < height; ++i)
	{
		rows[i] = (png_bytep)(pixels + i * png_get_rowbytes(png_ptr, info_ptr));
	}

	png_write_image(png_ptr, rows);
	free(rows);
	png_write_end(png_ptr, info_ptr);
  if(colored){
	  png_free(png_ptr, palette);
  }
	png_destroy_write_struct(&png_ptr, &info_ptr);
	fclose(png_file);
	return 0;
}
int put_formula(const char* formula, int x, int y, int width, void* data, char iscolor, char rgb[3]){
  char cmd[1000] = "texFormula.sh \"";
  strcat(cmd, formula);
  strcat(cmd, "\" tmp");
  system(cmd);
  FILE *f = fopen("out_tmp.png", "rb");
  if (f == NULL){
    fprintf(stderr, "pngpixel: out_tmp.png: could not open file\n");
    abort();
  }
  png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,
      NULL, NULL, NULL);
  png_infop info_ptr = png_create_info_struct(png_ptr);
  int bit_depth, color_type, interlace_method,
      compression_method, filter_method, row, col;
  png_init_io(png_ptr, f);
  png_read_info(png_ptr, info_ptr);
  png_bytep rowbuf = (png_bytep)png_malloc(png_ptr, png_get_rowbytes(png_ptr, info_ptr));
  if (!png_get_IHDR(png_ptr, info_ptr, (unsigned int*)&row, (unsigned int*)&col,
        &bit_depth, &color_type, &interlace_method,
        &compression_method, &filter_method)){
    png_error(png_ptr, "pngpixel: png_get_IHDR failed");
    abort();
  }
  png_start_read_image(png_ptr);
  for(int i = 0; i < col; i++){
    png_read_row(png_ptr, rowbuf, NULL);
    for(int j = 0; j < row; j++){
      unsigned char val = ((unsigned char*)rowbuf)[3*j];
      if(val < 200){
        if(iscolor){
          for(int ic = 0; ic < 3; ic++) ((unsigned char*)data)[3*(i*(width+x)+j+y)+ic] = rgb[ic];
        }else{
          ((unsigned char*)data)[i*(width+x)+j+y] = rgb[0];
        }
      }
    }
  }
  png_free(png_ptr, rowbuf);
  png_destroy_info_struct(png_ptr, &info_ptr);
  png_destroy_read_struct(&png_ptr, NULL, NULL);
  system("rm out_tmp.png");
  system("rm out_tmp.pdf");
  return 0;
}
