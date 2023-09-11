#include "format.h"
#include "opencv2/core/mat.hpp"

enum mode {MOD2,MOD, REAL, IMAG, PHASE, PHASERAD};
using namespace std;
using namespace cv;
Mat* extend( Mat &src , Real ratio, Real val = 0);
static bool opencv_reverted = 0;
Real getVal(mode m, fftw_format &data);
Real getVal(mode m, Real &data);

Mat* convertFromRealToInteger(Mat *fftwImage, Mat* opencvImage = 0, mode m = MOD, bool isFrequency = 0, Real decay = 1, const char* label= "default",bool islog = 0);

Mat* convertFromComplexToInteger(Mat *fftwImage, Mat* opencvImage = 0, mode m = MOD, bool isFrequency = 0, Real decay = 1, const char* label= "default",bool islog = 0);

Mat* convertFromIntegerToComplex(Mat &image, Mat* cache = 0, bool isFrequency = 0, const char* label= "default");
Mat* convertFromIntegerToReal(Mat &image, Mat* cache = 0, bool isFrequency = 0, const char* label= "default");
Mat* convertFromIntegerToComplex(Mat &image,Mat &phase,Mat* cache = 0);

template<typename functor, typename format=fftw_format>
void imageLoop(Mat* data, void* arg, bool isFrequency = 0){
  int row = data->rows;
  int column = data->cols;
  format *rowp;
  functor* func = static_cast<functor*>(arg);
  for(int x = 0; x < row ; x++){
    int targetx = x;
    if(isFrequency) targetx = x<row/2?x+row/2:(x-row/2);
    rowp = data->ptr<format>(x);
    for(int y = 0; y<column; y++){
      int targety = y;
      if(isFrequency) targety = y<column/2?y+column/2:(y-column/2);
      (*func)(targetx, targety , rowp[y]);
    }
  }
}
template<typename functor, typename format1, typename format2>
void imageLoop(Mat* data, Mat* dataout, void* arg, bool isFrequency = 0){
  int row = data->rows;
  int column = data->cols;
  format1 *rowp;
  format2 *rowo;
  functor* func = static_cast<functor*>(arg);
  for(int x = 0; x < row ; x++){
    int targetx = x;
    if(isFrequency) targetx = x<row/2?x+row/2:(x-row/2);
    rowp = data->ptr<format1>(x);
    rowo = dataout->ptr<format2>(targetx);
    for(int y = 0; y<column; y++){
      int targety = y;
      if(isFrequency) targety = y<column/2?y+column/2:(y-column/2);
      (*func)(targetx, targety , rowp[y], rowo[targety]);
    }
  }
}
Mat* multiWLGen(Mat* original, Mat* merged, Real m, Real step = 1, Real dphaselambda = 0, Real *spectrum = 0);
Mat* multiWLGenAVG(Mat* original, Mat* merged, Real m, Real step = 1, Real *spectrum = 0);
Mat* multiWLGenAVG_MAT(Mat* original, Mat* merged, Real m, Real step = 1, Real *spectrum = 0);
Mat* multiWLGenAVG_MAT_AC(Mat* original, Mat* merged, Real m, Real step = 1, Real *spectrum = 0);
Mat* multiWLGenAVG_MAT_FFT(Mat* original, Mat* merged, Real m, Real step = 1, Real *spectrum = 0);
Mat* multiWLGenAVG_MAT_AC_FFT(Mat* original, Mat* merged, Real m, Real step = 1, Real *spectrum = 0);
Mat* multiWLGenAVG_AC_FFT(Mat* original, Mat* merged, Real m, Real step = 1, Real *spectrum = 0);
template<typename T = complex<Real>>
Mat* convertFO(Mat* mat, Mat* cache = 0){
       int rows = mat->rows;
       int cols = mat->cols;
       if(cache == 0) {
               cache = new Mat();
               mat->copyTo(*cache);
       }
       T *rowi, *rowo;
       for(int x = 0 ; x < rows; x++){
               rowi = mat->ptr<T>(x);
               rowo = cache->ptr<T>((x >= rows/2)? x-rows/2:(x+rows/2));
               for(int y = 0 ; y < cols ; y++){
                       rowo[(y >= cols/2)?y-cols/2:(y+cols/2)] = rowi[y];
               }
       }
       return cache;
}

void plotColor(const char* name, Mat* logged);

