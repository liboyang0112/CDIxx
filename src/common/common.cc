#include "imgio.h"
#include "memManager.h"
void* ccmalloc(size_t sz){
  return ccmemMngr.borrowCache(sz);
}
Real *readImage(const char* name, int &row, int &col, void* (cmalloc)(size_t)){
  return readImage_c(name, &row, &col, (void*)(cmalloc?cmalloc:ccmalloc));
};
