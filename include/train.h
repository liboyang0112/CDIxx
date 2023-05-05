#include "torchConfig.h"
class torchJob: public torchConfig{
  public:
    torchJob(const char* configfile) : torchConfig(configfile){};
    int train() ;
};
