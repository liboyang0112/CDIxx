#include "experimentConfig.h"
#include <string>

class CDI : public experimentConfig{
  public:
    CDI(const char* configfile);
    Real* patternData = 0;
    void* patternWave = 0;
    void* autoCorrelation = 0;
    Real* support = 0;
    Real residual = 0;
    void *cuda_spt;
    void *mnist_dat = 0;
    std::string save_suffix = "";
    void *devstates;
    void multiplyPatternPhaseMid(void* amp, Real distance);
    void multiplyFresnelPhaseMid(void* amp, Real distance);
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
    void* phaseRetrieve();
    void saveState();
};
void applyESWSupport(complexFormat* ESW, complexFormat* ISW, complexFormat* ESWP, Real* length);
void initESW(complexFormat* ESW, Real* mod, complexFormat* amp);
void applyESWMod(complexFormat* ESW, Real* mod, complexFormat* amp, int noiseLevel);
void calcESW(complexFormat* sample, complexFormat* ISW);
void calcO(complexFormat* ESW, complexFormat* ISW);
void applyAutoCorrelationMod(complexFormat* source,complexFormat* target, Real *bs = 0);
