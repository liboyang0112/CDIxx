#include <vector>
#include <string>
#include <iostream>
#include <string.h>
#include "fmt/core.h"
#include "format.hpp"
#include "readConfig.hpp"
#define subParsers (*(std::vector<AlgoParser*>*) subParsersp)
#define count (*(std::vector<int>*) countp)
#define algoList (*(std::vector<int>*) algoListp)
using namespace std;

// This example reads the configuration file 'example.cfg' and displays
// some of its contents.
#include <libconfig.h>
int readConfigFile(const char * filename, config_t *cfg)
{
  if(! config_read_file(cfg, filename))
  {
    fmt::println(stderr, "{}:{} - {}", config_error_file(cfg),
        config_error_line(cfg), config_error_text(cfg));
    config_destroy(cfg);
    return(EXIT_FAILURE);
  }
  // Read the file. If there is an error, report it and exit.

  return EXIT_SUCCESS;
}

readConfig::readConfig(const char* configfile){
  config_t cfgs;
  config_t* cfg = &cfgs;
  config_init(cfg);
  int ret = readConfigFile(configfile, cfg);
  fmt::println("config file: {}", configfile);
  if(ret==EXIT_FAILURE) exit(ret);

  // Output a list of all vdWFluids in the inventory.
    config_setting_t *InputImages = config_lookup(cfg,"InputImages");
    config_setting_t *defaultImages= config_setting_lookup(InputImages,"default");
    config_setting_t *pupilImages= config_setting_lookup(InputImages,"pupil");

    config_setting_lookup_string(defaultImages,"Intensity",&common.Intensity);
    config_setting_lookup_string(defaultImages,"Phase",&common.Phase);
    config_setting_lookup_string(defaultImages,"restart",&common.restart);
    config_setting_lookup_string(defaultImages,"Pattern",&common.Pattern);
    config_setting_lookup_string(pupilImages,"Intensity",&pupil.Intensity);
    config_setting_lookup_string(pupilImages,"Phase",&pupil.Phase);
    config_setting_lookup_string(pupilImages,"restart",&pupil.restart);
    config_setting_lookup_string(pupilImages,"Pattern",&pupil.Pattern);
  // Output a list of all vdWFluids in the inventory.
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
void readConfig::print(){
fmt::print("common Intensity={}\n", common.Intensity);
fmt::print("common Phase={}\n", common.Phase);
fmt::print("common restart={}\n", common.restart);
fmt::print("common Pattern={}\n", common.Pattern);
fmt::print("pupil Intensity={}\n", pupil.Intensity);
fmt::print("pupil Phase={}\n", pupil.Phase);
fmt::print("pupil restart={}\n", pupil.restart);
fmt::print("pupil Pattern={}\n", pupil.Pattern);

#define PRINTBOOL(x, y) fmt::print("bool: {} = {}  (default = {})\n", #x, x, y);
#define PRINTINT(x, y)  fmt::print("int: {} = {}  (default = {})\n", #x, x, y);
#define PRINTREAL(x, y) fmt::print("float: {} = {}  (default = {})\n", #x, x, y);
#define PRINTSTR(x, y)  fmt::print("string: {} = {}  (default = {})\n", #x, x, y);
  BOOLVAR(PRINTBOOL)
    INTVAR(PRINTINT)
    REALVAR(PRINTREAL)
    STRVAR(PRINTSTR)
}

AlgoParser::AlgoParser(const char* f){
  nAlgo = cnt;
  subParsersp=new std::vector<AlgoParser*>();
  countp=new std::vector<int>();
  algoListp=new std::vector<int>();
  int j=0;
  std::string formula = f;
  for(int i=0;f[i]!='\0';i++)
  {
    if(f[i]!=' ')
      formula[j++]=f[i];
  }
  formula.resize(j+1);
  formula[j]='\0';
  fmt::println("{}",formula);
  auto position = formula.find("(");
  while(position!= std::string::npos){
    auto positione = formula.find(")");
    auto currentPosition = position;
    currentPosition = formula.find("(",position+1,positione-currentPosition+1);
    while(currentPosition!=std::string::npos){
      positione = formula.find(")",positione+1);
      currentPosition = formula.find("(",currentPosition+1,positione-currentPosition+1);
      fmt::print("{},{},{}\n", position, currentPosition, positione);
    }
    subParsers.push_back(new AlgoParser(formula.substr(position+1, positione-position-1).c_str()));
    formula.replace(position, positione-position+1, "subParser");
    fmt::println("{}",formula);
    position = formula.find("(");
  }
  char* term = const_cast<char*>(formula.c_str());
  char* ptrstore;
  int iParser = 0;
  char* ptr = strtok_r(term,"*",&ptrstore);
  do{
    int num = atoi(ptr);
    ptr = strtok_r(NULL,"+",&ptrstore);
    string str = ptr;
    count.push_back(num);
    if(str=="RAAR") algoList.push_back(RAAR);
    else if(str=="HIO") algoList.push_back(HIO);
    else if(str=="ER") algoList.push_back(ER);
    else if(str=="POSER") algoList.push_back(POSER);
    else if(str=="POSHIO") algoList.push_back(POSHIO);
    else if(str=="FHIO") algoList.push_back(FHIO);
    else if(str=="KKT") algoList.push_back(KKT);
    else if(str=="shrinkWrap") algoList.push_back(shrinkWrap);
    else if(str=="XCORRELATION") algoList.push_back(XCORRELATION);
    else if(str=="subParser") algoList.push_back(nAlgo+iParser++);
    else{
      fmt::println("Algorithm {} not found", str);
      exit(0);
    }
  }while( (ptr = strtok_r(NULL,"*",&ptrstore)));
  restart();
}
void AlgoParser::restart(){
  currentAlgo = 0;
  currentCount = count[0];
  for(auto sub : subParsers){
    sub->restart();
  }
}
int AlgoParser::next(){
  if(currentCount==0){
    if(currentAlgo == algoList.size()-1) return -1; // end of the algorithm
    currentCount = count[++currentAlgo];
  }
  if(algoList[currentAlgo]>=nAlgo) {
    int retVal = subParsers[algoList[currentAlgo]-nAlgo]->next();
    if(retVal==-1) {
      currentCount--;
      subParsers[algoList[currentAlgo]-nAlgo]->restart();
      return next();
    }
    return retVal;
  } else {
    currentCount--;
    return algoList[currentAlgo];
  }
}
// eof

