#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif
  int initLMDB(const char* dbDIR);
  int saveLMDB();
  int fillLMDB(int *keyval, void* data, size_t data_size, void* label, size_t label_size);
  void readLMDB(void** data, size_t* data_size, void** label, size_t *label_size, int *keyval);
#ifdef __cplusplus
}
#endif
