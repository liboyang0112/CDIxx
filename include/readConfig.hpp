/**
 * @file 	readConfig.h
 * @brief A config file header.
 * @author Boyang Li
 */

#ifndef __CDIxxCONFIG
#define __CDIxxCONFIG

// This example reads the configuration file 'example.cfg' and displays
// some of its contents.

struct CDIfiles{
  const char* Pattern;
  const char* Intensity;
  const char* Phase;
  const char* restart;
};

#define DECINT(x,y) int x = y;
#define DECBOOL(x,y) bool x = y;
#define DECREAL(x,y) double x = y;
#define DECSTR(x,y) const char* x = y;

#define INTVAR(F) \
F(beamStopSize,5)F(nIter,1)F(nIterpupil,1)F(noiseLevel,0)F(noiseLevel_pupil,0)F(verbose,2)F(mnistN,3)F(cropPattern,0)F(spectrumSamplingStep,10)F(saveVideoEveryIter,0)F(prtfIter, 1000)F(zernikeIter, 0)F(stepSize, 64)

#define BOOLVAR(F) \
F(isFlip,0)F(runSim,0)F(simCCDbit,0)F(isFresnel,0)F(doIteration,0)F(useGaussionLumination,0)F(useGaussionHERALDO,0)F(doCentral,0)F(useRectHERALDO,0)F(dopupil,0)F(useDM,0)F(useBS,0)F(reconAC,0)F(phaseModulation_pupil,0)F(intensityModulation,1)F(restart,0)F(saveIter,0)F(domnist,0)F(saveLMDB, 0)F(solveSpectrum,0)

#define REALVAR(F) \
F(exposure,0)F(exposurepupil,0)F(oversampling,0)F(oversampling_spt,0)F(lambda,0)F(d,0)F(dpupil,0)F(pixelsize,0)F(beamspotsize,0)F(shrinkThreshold,0) F(positionUncertainty,0)F(costheta,1)F(photoncount,1e9)F(phaseModulation,0)

#define STRVAR(F) \
F(mnistData,"data")F(algorithm,"1000*HIO")F(ccd_response,"/home/boyang/SharedNTFS/images/ccd_response.txt")F(spectrum,"/home/boyang/SharedNTFS/images/spectrum/spectrum_after_sample.txt")F(BSimage, "")F(outputDir, "./")
class readConfig
{
public:
  BOOLVAR(DECBOOL)
  INTVAR(DECINT)
  REALVAR(DECREAL)
  STRVAR(DECSTR)
//bool configs
  CDIfiles pupil;
  CDIfiles common;
  readConfig(const char* configfile);
  void print();
  ~readConfig(){};
};

class AlgoParser{
public:
  void* subParsersp;
  void* countp;
  void* algoListp;
  int nAlgo;
  unsigned int currentAlgo;
  int currentCount;
  AlgoParser(const char* formula);
  void restart();
  int next();
};
#endif
