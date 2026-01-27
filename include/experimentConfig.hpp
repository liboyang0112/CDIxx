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
    Real forwardFactor = 0;
    Real fresnelFactor = 0;
    Real resolution = 0;
    int row = 0;
    int column = 0;
    complexFormat* objectWave = 0;
    complexFormat* pupilpatternWave = 0;
    complexFormat* pupilobjectWave = 0;
    Real* beamstop = 0;
    Real* pupilpatternData = 0;
    Real fresnelFactorpupil = 0;
    Real fresnelFactorMid = 0;
    void createBeamStop();
    void propagate(complexFormat* datain, complexFormat* dataout, bool isforward);
    void angularPropagate(complexFormat* datain, complexFormat* dataout, bool isforward);
    void multiplyPatternPhase(complexFormat* amp, Real distance);
    void multiplyPatternPhase_reverse(complexFormat* amp, Real distance);
    void multiplyFresnelPhase(complexFormat* amp, Real distance);
    void multiplyPatternPhase_factor(complexFormat* amp, Real factor1, Real factor2);
    void multiplyFresnelPhase_factor(complexFormat* amp, Real factor);
    void calculateParameters();
};
void angularSpectrumPropagate(complexFormat* input, complexFormat* field, Real imagesize_over_lambda, Real z_over_lambda, int n);
void opticalPropagate(complexFormat* field, Real lambda, Real d, Real imagesize, int n);
void multiplyPatternPhase_Device(complexFormat* amp, Real r_d_lambda, Real d_r_lambda);
void multiplyPatternPhaseOblique_Device(complexFormat* amp, Real r_d_lambda, Real d_r_lambda, Real costheta);
void multiplyFresnelPhase_Device(complexFormat* amp, Real phaseFactor);
void multiplyFresnelPhaseOblique_Device(complexFormat* amp, Real phaseFactor, Real costheta_r);
#endif
