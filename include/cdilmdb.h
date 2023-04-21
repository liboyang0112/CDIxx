#include <stdlib.h>
extern "C" {
  int initLMDB();
  int saveLMDB();
  int fillLMDB(int *keyval, void* data, size_t data_size, void* label, size_t label_size);
  int readLMDB(void* data, size_t* data_size, void* label, size_t *label_size, int *keyval);
}
