#include <stdio.h>
#include "holo.h"

int main(int argc, char** argv )
{
  holo setups(argv[1]);
  if(argc < 2){
    printf("please feed the object intensity and phase image\n");
  }
  setups.simulate();
  return 0;
}

