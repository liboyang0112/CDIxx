#include "broadBand.hpp"
class monoChromo : public broadBand{
  public:
    complexFormat **gs=0;  //sample, for each wavelength
    monoChromo():broadBand(){};
    void solveMWL(complexFormat* d_input, complexFormat* d_patternSum, int noiseLevel = 0, bool restart = 0, int nIter = 200, bool updateX = 1, bool updateY = 0);
};
class monoChromo_constRatio : public broadBand_constRatio{
  public:
    complexFormat **gs=0;  //sample, for each wavelength
    monoChromo_constRatio():broadBand_constRatio(){};
    void solveMWL(complexFormat* d_input, complexFormat* d_patternSum, int noiseLevel = 0, bool restart = 0, int nIter = 200, bool updateX = 1, bool updateY = 0);
};
void updateMomentum(complexFormat* force, complexFormat* mom, Real dx);
void overExposureZeroGrad (complexFormat* deltab, complexFormat* b, int noiseLevel);
