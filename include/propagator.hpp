#ifndef __PROPAGATOR_H__
#define __PROPAGATOR_H__
#include "format.hpp"

class propagator{
  public:
    //calculated later by init function, image size dependant
    propagator(){}
    Real pixelsize = 0;
    int row = 0;
    int column = 0;
    Real distance = 0;
    Real lambda = 0;
    void fresnelPropagate(complexFormat* field);
    void multiplyPatternPhase(complexFormat* amp);
    void multiplyPatternPhase_reverse(complexFormat* amp);
    void multiplyFresnelPhase(complexFormat* amp);
    void removeFresnelPhase(complexFormat* amp);
    void multiplyPatternPhase_factor(complexFormat* amp, Real factor1, Real factor2);
    void multiplyFresnelPhase_factor(complexFormat* amp, Real factor);
    void angularSpectrumPropagate(complexFormat* input, complexFormat* output);
    void angularSpectrumPropagateReverse(complexFormat* input, complexFormat* output);
    void fresnelPropagate(complexFormat* field, Real lambda, Real d, Real imagesize, int n);
};
#endif
