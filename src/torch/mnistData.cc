#include <iostream>
#include <vector>
#include <string>
#include "memManager.h"
#include "mnistData.h"
#define mn torch::data::datasets::MNIST
#define dat ((mn*)dataset)
#ifdef DOTORCHMNIST
#include <torch/data/datasets/mnist.h>
mnistData::mnistData(const char* dir, int &row, int &col){
  dataset = new mn(dir);
  auto theData = dat->get(0).data;
  auto foo_a = theData.accessor<float,3>();
  row = foo_a[0].size(0);
  col = foo_a[0][0].size(0);
  idat = 0;
}
Real* mnistData::read(){
  if(idat == dat->size().value()) return 0;
  auto theData = dat->get(idat).data;
  auto foo_a = theData.accessor<float,3>();
  idat++;
  return &(foo_a[0][0][0]);
}
#else
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
};
Real* mnistData::read(){
  unsigned char* tmp = (unsigned char*)ccmemMngr.borrowCache(rowraw*colraw*sizeof(char));
  ((ifstream*)dataset)->read((char*)tmp, rowraw*colraw*sizeof(char));
  for(int i = 0; i < rowraw*colraw; i++){
    output[i] = Real(tmp[i])/255;
  }
  ccmemMngr.returnCache(tmp);
  return output;
}
#endif
mnistData::~mnistData(){
  delete (ifstream*)dataset;
  ccmemMngr.returnCache(output);
};
