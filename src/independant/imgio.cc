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
