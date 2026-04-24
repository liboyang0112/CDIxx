#include "imgio.hpp"
#include "memManager.hpp"
void* ccmalloc(size_t sz){
  return ccmemMngr.borrowCache(sz);
}
Real *readImage(const char* name, int &row, int &col, void* (cmalloc)(size_t)){
  struct imageFile f;
  Real* ptr = readImage_c(name, &f, (void*)(cmalloc?cmalloc:ccmalloc));
  row = f.rows;
  col = f.cols;
  return ptr;
};
Image3 readImage3(const char* name, void* (cmalloc)(size_t)){
  struct imageFile f;
  Real* ptr = readImage3_c(name, &f, (void*)(cmalloc?cmalloc:ccmalloc));
  Image3 img;
  img.rows = f.rows;
  img.cols = f.cols;
  img.r = ptr;
  img.g = ptr + f.rows * f.cols;
  img.b = ptr + 2 * f.rows * f.cols;
  return img;
}
