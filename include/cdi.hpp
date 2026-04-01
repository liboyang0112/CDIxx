#include "propagator.hpp"
#include "readConfig.hpp"
#include "format.hpp"
#include <string>

class CDI : public readConfig{
  public:
    CDI(const char* configfile);
    Real* patternData = 0;
    Real* beamstop = 0;
    propagator propagate;
    complexFormat* patternWave = 0;
    complexFormat* autoCorrelation = 0;
    int row = 0, column = 0;
    complexFormat* objectWave = 0;
    Real* support = 0;
    Real residual = 0;
    Real resolution = 0;
    void *cuda_spt;
    void *mnist_dat = 0;
    std::string save_suffix = "";
    void *devstates;
    void allocateMem();
    void readObjectWave();
    void readPattern();
    void readFiles();
    void setPattern(Real* pattern);
    void setPattern_c(complexFormat* pattern);
    void init();
    void prepareIter();
    void checkAutoCorrelation();
    void createSupport();
    void initSupport();
    complexFormat* phaseRetrieve();
    void saveState();
};
void applyESWSupport(complexFormat* ESW, complexFormat* ISW, complexFormat* ESWP);
void initESW(complexFormat* ESW, Real* mod, complexFormat* amp);
void applyESWMod(complexFormat* ESW, Real* mod, complexFormat* amp);
void calcESW(complexFormat* sample, complexFormat* ISW);
void calcO(complexFormat* ESW, complexFormat* ISW);
void applyAutoCorrelationMod(complexFormat* source,complexFormat* target, Real *bs = 0);
void applySupport(void *gkp1, void *gkprime, int algo, Real *spt);
