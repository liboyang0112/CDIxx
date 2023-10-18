#ifdef __cplusplus
extern "C" {
#endif
int writePng(const char* png_file_name, void* pix , int width, int height, int bit_depth, int colored);
int put_formula(const char* formula, int x, int y, int width, void* data, char iscolor, char rgb[3]);
#ifdef __cplusplus
}
#endif
