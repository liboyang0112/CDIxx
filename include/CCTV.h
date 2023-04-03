#include "cudaConfig.h"
#include "experimentConfig.h"

class CCTV : public experimentConfig{
  public:
    CCTV(const char* configfile);
    Real* patternData = 0;
    complexFormat* patternWave = 0;
    Real* support = 0;
    rect *cuda_spt;
    void *mnist_dat = 0;
    std::string save_suffix = "";
    curandStateMRG32k3a *devstates;
    void allocateMem();
    void readObjectWave();
    void readPattern();
    void calculateParameters();
    void readFiles();
    void setPattern(void* pattern);
    void setPattern_c(void* pattern);
    void init();
    void prepareIter();
    void createSupport();
    void initSupport();
    complexFormat* phaseRetrieve();
    void saveState();
};
