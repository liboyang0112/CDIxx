#include "opencv2/core/mat.hpp"
#include "format.h"

void fftw_init();
cv::Mat* fftw(cv::Mat* in, cv::Mat* out = 0, bool isforward = 1, Real ratio = 0);
