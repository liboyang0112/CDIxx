#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif
  int initLMDB(int* handle, const char* dbDIR);
  void saveLMDB(int handle);
  void setCompress(int handle);
  int fillLMDB(int handle, int *keyval, int ndata, void** data, size_t* data_size);
  void readLMDB(int handle, int *ndata, void*** data, size_t** data_size, int *keyval);
#ifdef __cplusplus
}
#endif
