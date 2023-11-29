#include "torchConfig.h"
#include  <memory>
#include <vector>
#include <string>
#include <iostream>
#include <string.h>
using namespace std;

// This example reads the configuration file 'example.cfg' and displays
// some of its contents.
#include <libconfig.h>
int torchConfigFile(const char * filename, config_t *cfg)
{
  if(! config_read_file(cfg, filename))
  {
    fprintf(stderr, "%s:%d - %s\n", config_error_file(cfg),
        config_error_line(cfg), config_error_text(cfg));
    config_destroy(cfg);
    return(EXIT_FAILURE);
  }
  // Read the file. If there is an error, report it and exit.

  return EXIT_SUCCESS;
}

torchConfig::torchConfig(const char* configfile){
  config_t cfgs;
  config_t* cfg = &cfgs;
  config_init(cfg);
  int ret = torchConfigFile(configfile, cfg);
  cout << "config file: " << configfile << endl;
  if(ret==EXIT_FAILURE) exit(ret);

#define getValbool(x,y) if(config_setting_lookup_bool(Job, #x, &tmp)) x = tmp;
#define getValint(x,y) config_setting_lookup_int(Job, #x, &x);
#define getValfloat(x,y) config_setting_lookup_float(Job, #x, &x);
#define getValstring(x,y) config_setting_lookup_string(Job, #x, &x);
    config_setting_t *Job = config_lookup(cfg,"Job");
    int tmp;
    BOOLVAR(getValbool);
    INTVAR(getValint);
    REALVAR(getValfloat);
    STRVAR(getValstring);
}
void torchConfig::print(){
#define PRINTBOOL(x,y) std::cout<<"bool: "<<#x<<" = "<<x<<"  (default = "<<y<<")"<<std::endl;
#define PRINTINT(x,y) std::cout<<"int: "<<#x<<" = "<<x<<"  (default = "<<y<<")"<<std::endl;
#define PRINTREAL(x,y) std::cout<<"float: "<<#x<<" = "<<x<<"  (default = "<<y<<")"<<std::endl;
#define PRINTSTR(x,y) std::cout<<"string: "<<#x<<" = "<<x<<"  (default = "<<y<<")"<<std::endl;
  BOOLVAR(PRINTBOOL)
  INTVAR(PRINTINT)
  REALVAR(PRINTREAL)
  STRVAR(PRINTSTR)
}
