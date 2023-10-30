#include "cudaConfig.h"
#include "experimentConfig.h"
#include <string>

class CDI : public experimentConfig{
  public:
    CDI(const char* configfile);
    Real* patternData = 0;
    complexFormat* patternWave = 0;
    complexFormat* autoCorrelation = 0;
    Real* support = 0;
    Real residual = 0;
    rect *cuda_spt;
    void *mnist_dat = 0;
    std::string save_suffix = "";
    curandStateMRG32k3a *devstates;
    void propagatepupil(complexFormat* datain, complexFormat* dataout, bool isforward);
    void propagateMid(complexFormat* datain, complexFormat* dataout, bool isforward);
    void multiplyPatternPhaseMid(complexFormat* amp, Real distance);
    void multiplyFresnelPhaseMid(complexFormat* amp, Real distance);
    void allocateMem();
    void readObjectWave();
    void readPattern();
    void calculateParameters();
    void readFiles();
    void setPattern(void* pattern);
    void setPattern_c(void* pattern);
    void init();
    void prepareIter();
    void checkAutoCorrelation();
    void createSupport();
    void initSupport();
    complexFormat* phaseRetrieve();
    void saveState();
};
void applySupport(complexFormat *gkp1, complexFormat *gkprime, Algorithm algo, Real *spt, int iter = 0, Real fresnelFactor = 0);
