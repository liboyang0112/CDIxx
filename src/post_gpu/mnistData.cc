#include "mnistData.hpp"
#include "cudaConfig.hpp"
#include <iostream>
#include <string>
#include "memManager.hpp"
#include "mnistData.hpp"
#include <fstream>
using namespace std;
int ReverseInt(int i)
  {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
  }

mnistData::mnistData(const char* dir){
  std::string filename = "/train-images-idx3-ubyte";
  filename = dir+filename;
  ifstream *file = new ifstream(filename, ios::binary);
  if(file->fail()) {
    fprintf(stderr, "File open failed: %s, check if it exists!", filename.c_str());
    abort();
  }
  dataset = file;
  int magic_number = 0;
  int number_of_images = 0;
  file->read((char*)&magic_number, sizeof(magic_number));
  file->read((char*)&number_of_images, sizeof(number_of_images));
  file->read((char*)&rowraw, sizeof(rowraw));
  file->read((char*)&colraw, sizeof(colraw));
  magic_number = ReverseInt(magic_number);
  number_of_images = ReverseInt(number_of_images);
  rowraw = ReverseInt(rowraw);
  colraw = ReverseInt(colraw);
  output = (Real*)ccmemMngr.borrowCache(rowraw*colraw*sizeof(Real));
  printf("mnist image size: %d x %d\n", rowraw, colraw);
};
Real* mnistData::read(){
  size_t sz =  rowraw*colraw*sizeof(char);
  unsigned char* tmp = (unsigned char*)ccmemMngr.borrowCache(sz);
  size_t startpos = sz * size_t(idat);
  ((ifstream*)dataset)->seekg (sizeof(int)*4 + startpos);
  ((ifstream*)dataset)->read((char*)tmp, sz);
  idat+=1;
  for(int i = 0; i < rowraw*colraw; i++){
    output[i] = Real(tmp[i])/255;
  }
  ccmemMngr.returnCache(tmp);
  return output;
}
mnistData::~mnistData(){
  delete (ifstream*)dataset;
  ccmemMngr.returnCache(output);
};

cuMnist::cuMnist(const char* dir, int nm, int re, int r, int c) : mnistData(dir), refinement(re), row(r), col(c), nmerge(nm){
  cuRaw = memMngr.borrowCache(rowraw*colraw*sizeof(Real));
  rowrf = rowraw*nmerge;
  colrf = colraw*nmerge;
  cuRefine = memMngr.borrowCache(rowrf*colrf*refinement*refinement*sizeof(Real));
  if(refinement!=1){
    createPlan(&handle, rowrf*refinement, colrf*refinement);
    createPlan(&handleraw, rowrf, colrf);
  }
  myCuMalloc(complexFormat, cacheraw, rowrf*colrf);
  myCuMalloc(complexFormat, cache, rowrf*colrf*refinement*refinement);
};
void cuMnist::cuRead(void* out){
  init_cuda_image(65536,1);
  void* media = (refinement==1?cuRefine:out);
  clearCuMem(media,rowrf*colrf*sizeof(Real));
  resize_cuda_image(rowraw, colraw);
  for(int i = 0; i < nmerge; i++){
    for(int j = 0; j < nmerge; j++){
      myMemcpyH2D(cuRaw, read(), rowraw*colraw*sizeof(Real));
      paste( (Real*)media, (Real*)cuRaw, colrf, rowraw*i*2/3, colraw*j*2/3);
    }
  }
  if(refinement!=1){
    resize_cuda_image(rowrf, colrf);
    extendToComplex((Real*)out, cacheraw);
    myFFTM(handleraw,cacheraw, cacheraw);
    resize_cuda_image(rowrf*refinement, colrf*refinement);
    padinner(cacheraw, cache, rowrf, colrf, 1./(rowrf*colrf));
    myIFFTM(handle,cache, cache);
    getMod((Real*)cache, cache);
    applyThreshold((Real*)cache, (Real*)cache, 0.5);
    resize_cuda_image(row, col);
    pad( (Real*)cache, (Real*)out,rowrf*refinement, colrf*refinement);
  }else{
    resize_cuda_image(row, col);
    pad( (Real*)media, (Real*)out,rowrf, colrf);
  }
}
/*
void cuMnist::cuRead(void* out){
  resize_cuda_image(rowrf, colrf);
  init_cuda_image(65536,1);
  auto media = (refinement==1?cuRefine:out);
  clearCuMem(media,rowrf*colrf*sizeof(Real));
  for(int i = 0; i < nmerge; i++){
    for(int j = 0; j < nmerge; j++){
      myMemcpyH2D(cuRaw, read(), rowraw*colraw*sizeof(Real));
      paste( (Real*)media, (Real*)cuRaw, colrf, rowraw*i*2/3, colraw*j*2/3);
    }
  }
  if(refinement!=1){
    resize_cuda_image(rowrf*refinement, colrf*refinement);
    refine((Real*)out, (Real*)cuRefine, refinement);
    resize_cuda_image(row, col);
    pad( (Real*)cuRefine, (Real*)out,rowrf*refinement, colrf*refinement);
  }else{
    resize_cuda_image(row, col);
    pad( (Real*)media, (Real*)out,rowrf, colrf);
  }
}
*/
