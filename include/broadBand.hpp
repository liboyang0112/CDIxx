#include "format.hpp"
#include<stdint.h>
class broadBand_base{
  public:
    int *locplan;
    double *spectra;  //normalized spectra
    double *lambdas;  //normalized lambdas, 1 is the shortest
    int jump = 20;
    int skip = 10;
    int nlambda;
    int row;
    int column;
    void *devstates = 0;
    broadBand_base(){};
    void writeSpectra(const char* filename, Real scale = 1);
    void resetSpectra();
};
class broadBand : public broadBand_base{
  public:
    int *rows;
    int *cols;
    void init(int nrow, int ncol, double minlambda, double maxlambda);
    void init(int nrow, int ncol, int nlambda_, double* lambdas_, double* spectra_);
    Real init(int nrow, int ncol, double* lambdasi, double* spectrumi, int narray);
    void generateMWL(void* d_input, void* d_patternSum, void* single = 0);
    broadBand() : broadBand_base(){};
};
class broadBand_constRatio : public broadBand_base{
  private:
    complexFormat* cache;
  public:
    int patternptr;
    char nextPattern(complexFormat* currentp, complexFormat* nextp, complexFormat* origin, char transpose = 0);
    void writeSpectra(const char* filename, Real factor);
    void initptr(){patternptr = nmiddle;}
    int nmiddle;
    int thisrow, thiscol, thisrowp, thiscolp;
    void init(int nrow, int ncol, double minlambda, double maxlambda);
    Real init(int nrow, int ncol, double* lambdasi, double* spectrumi, int narray);
    broadBand_constRatio() : broadBand_base(){};
    char skipPattern();
    void reorderSpect();
    void applyAT(complexFormat* src, complexFormat* dest, char zoomout = 0);
    void applyA(complexFormat* src, complexFormat* dest, char zoomout = 0);
    void generateMWL(void* d_input, void* d_patternSum, void* single = 0);
};
void assignRef_d (complexFormat* wavefront, uint32_t* mmap, complexFormat* rf, int n);
void expandRef (complexFormat* rf, complexFormat* amp, uint32_t* mmap, int row, int col, int row0, int col0);
void expandRef (complexFormat* rf, complexFormat* amp, uint32_t* mmap, int row, int col, int row0, int col0, complexFormat a);
void saveRef (complexFormat* rf, complexFormat* amp, uint32_t* mmap, int row, int col, int row0, int col0, Real norm);
void saveRef_Real (complexFormat* rf, complexFormat* amp, uint32_t* mmap, int row, int col, int row0, int col0, int n, Real norm);

