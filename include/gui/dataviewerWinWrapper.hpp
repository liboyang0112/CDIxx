#include "format.hpp"
#include "imageFile.hpp"
void* to_gpu(void*, struct imageFile *f);
void processFloat(void* cache, void* cudaData, char m, char isFrequency, Real decay, char islog, char isFlip, char isColor);
void processComplex(void* cache, void* cudaData, char m, char isFrequency, Real decay, char islog, char isFlip, char isColor);
