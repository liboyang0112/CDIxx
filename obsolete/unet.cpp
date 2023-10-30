#include "train.h"

int main(int argc, char* argv[]){
  torchJob job(argv[1]);
  job.train();
  return 0;
}
