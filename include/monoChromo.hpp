#include "format.hpp"
#include "broadBand.hpp"
class monoChromo : public broadBand{
  public:
    void **gs=0;  //sample, for each wavelength
    monoChromo():broadBand(){};
    void solveMWL(void* d_input, void* d_patternSum, int noiseLevel = 0, bool restart = 0, int nIter = 200, bool updateX = 1, bool updateY = 0);
};
class monoChromo_constRatio : public broadBand_constRatio{
  public:
    void **gs=0;  //sample, for each wavelength
    monoChromo_constRatio():broadBand_constRatio(){};
    void solveMWL(void* d_input, void* d_patternSum, int noiseLevel = 0, bool restart = 0, int nIter = 200, bool updateX = 1, bool updateY = 0);
};
void updateMomentum(complexFormat* force, complexFormat* mom, Real dx);
void overExposureZeroGrad (complexFormat* deltab, complexFormat* b, int noiseLevel);
