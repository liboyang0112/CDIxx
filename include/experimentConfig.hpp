#ifndef __EXPERIMENTCONFIG_H__
#define __EXPERIMENTCONFIG_H__

#include "readConfig.hpp"
#include "format.hpp"
#define verbose(i,a) if(verbose>=i){a;}
#define m_verbose(m,i,a) if(m.verbose>=i){a;}

class experimentConfig : public readConfig{
  public:
    //calculated later by init function, image size dependant
    experimentConfig(const char* configfile):readConfig(configfile){}
    Real enhancement = 0;
    Real forwardFactor = 0;
    Real fresnelFactor = 0;
    Real resolution = 0;
    int row = 0;
    int column = 0;
    void* objectWave = 0;
    void* pupilpatternWave = 0;
    void* pupilobjectWave = 0;
    Real* beamstop = 0;
    Real* pupilpatternData = 0;
    Real enhancementpupil = 0;
    Real fresnelFactorpupil = 0;
    Real enhancementMid = 0;
    Real fresnelFactorMid = 0;
    void createBeamStop();
    void propagate(void* datain, void* dataout, bool isforward);
    void angularPropagate(void* datain, void* dataout, bool isforward);
    void multiplyPatternPhase(void* amp, Real distance);
    void multiplyPatternPhase_reverse(void* amp, Real distance);
    void multiplyFresnelPhase(void* amp, Real distance);
    void multiplyPatternPhase_factor(void* amp, Real factor1, Real factor2);
    void multiplyFresnelPhase_factor(void* amp, Real factor);
    void calculateParameters();
};
void angularSpectrumPropagate(void* input, void* field, Real imagesize_over_lambda, Real z_over_lambda, int n);
void opticalPropagate(void* field, Real lambda, Real d, Real imagesize, int n);
#endif
