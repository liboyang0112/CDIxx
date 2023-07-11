#include "opencv2/opencv.hpp"
#include "memManager.h"
#include "readCXI.h"
#include "common.h"
#include "fstream"
#include <gsl/gsl_spline.h>
using namespace cv;
using namespace std;
Real* readImage(const char* name, int &row, int &col){
  printf("reading file: %s\n", name);
  Real *data;
  if(string(name).find(".cxi")!=string::npos){
    printf("Input is recognized as cxi file\n");
    Mat *mask;
    Mat imagein = readCXI(name, &mask);  //32FC2, max 65535
    row = imagein.rows;
    col = imagein.cols;
    data = (Real*)ccmemMngr.borrowCache(imagein.total()*sizeof(Real));
    for(int i = 0 ; i < imagein.total() ; i++) data[i] = ((complex<float>*)imagein.data)[i].real()/rcolor;
    return data;
  }
  Mat imagein = imread( name, IMREAD_UNCHANGED  );
  row = imagein.rows;
  col = imagein.cols;
  data = (Real*)ccmemMngr.borrowCache(imagein.total()*sizeof(Real));
  if(imagein.depth() == CV_8U){
    printf("input image nbits: 8, channels=%d\n",imagein.channels());
    if(imagein.channels()>=3){
      Mat image(imagein.rows, imagein.cols, CV_8UC1);
      cv::cvtColor(imagein, image, cv::COLOR_BGR2GRAY);
      for(int i = 0 ; i < image.total() ; i++) data[i] = ((float)(image.data[i]))/255;
    }else{
      for(int i = 0 ; i < imagein.total() ; i++) data[i] = ((float)(imagein.data[i]))/255;
    }
  }else if(imagein.depth() == CV_16U){
    printf("input image nbits: 16\n");
    for(int i = 0 ; i < imagein.total() ; i++) data[i] = ((float)(((uint16_t*)imagein.data)[i]))/65535;
  }else{  //Image data is float
    printf("Image depth %d is not recognized as integer type (%d or %d), Image data is treated as floats\n", imagein.depth(), CV_8U, CV_16U);
    imagein.addref();
    data = (Real*)imagein.data;
    ccmemMngr.registerMem(data, row*col*sizeof(Real));
  }
  return data;
}

void writeComplexImage(const char* name, void* data, int row, int column){
    FileStorage fs(name,FileStorage::WRITE);
    Mat output(row, column, float_cv_format(2));
    auto tmp = output.data;
    output.data = (uchar*)data;
    fs<<"data"<<output;
    fs.release();
    output.data = tmp;
}

void *readComplexImage(const char* name){
  FileStorage fs(name,FileStorage::READ);
  Mat image;
  fs["data"]>>(image);
  fs.release();
  size_t sz = image.rows*image.cols*sizeof(Real)*2;
  void *data = ccmemMngr.borrowCache(sz);
  memcpy(data, image.data, sz);
  return data;
};

void getNormSpectrum(const char* fspectrum, const char* ccd_response, Real &startLambda, Real &endLambda, int &nlambda, double *& outlambda, double *& outspectrum){
  std::vector<double> spectrum_lambda;
  std::vector<double> spectrum;
  std::vector<double> ccd_lambda;
  std::vector<double> ccd_rate;
  std::ifstream file_spectrum, file_ccd_response;
  double threshold = 1e-3;
  file_spectrum.open(fspectrum);
  file_ccd_response.open(ccd_response);
  double lambda, val, maxval;
  maxval = 0;
  while(file_spectrum){
    file_spectrum >> lambda >> val;
    spectrum_lambda.push_back(lambda);
    spectrum.push_back(val);
    if(val > maxval) maxval = val;
  }
  while(file_ccd_response>>lambda>>val){
    ccd_lambda.push_back(lambda);
    ccd_rate.push_back(val);
  }
  endLambda = std::min(Real(spectrum_lambda.back()),endLambda);
  bool isShortest = 1;
  nlambda = 0;
  gsl_interp_accel *acc = gsl_interp_accel_alloc ();
  gsl_spline *spline = gsl_spline_alloc (gsl_interp_cspline, ccd_lambda.size());
  gsl_spline_init (spline, &ccd_lambda[0], &ccd_rate[0], ccd_lambda.size());
  double ccdmax = ccd_lambda.back();
  double ccd_rate_max = ccd_rate.back();
  for(int i = 0; i < spectrum.size(); i++){
    lambda = spectrum_lambda[i];
    if(lambda<startLambda) continue;
    if(lambda>=endLambda) break;
    if(isShortest && spectrum[i] < threshold*maxval) continue;
    if(isShortest) startLambda = lambda;
    isShortest = 0;
    double ccd_rate_i = ccd_rate[0];
    if(lambda >= ccdmax) ccd_rate_i = ccd_rate_max;
    else if(lambda > ccd_lambda[0]) ccd_rate_i = gsl_spline_eval (spline, lambda, acc);
    spectrum_lambda[nlambda] = lambda/startLambda;
    spectrum[nlambda] = spectrum[i]*ccd_rate_i/maxval;
    nlambda++;
  }
  endLambda /= startLambda;
  outlambda = (double*) ccmemMngr.borrowCache(sizeof(double)*nlambda);
  outspectrum = (double*) ccmemMngr.borrowCache(sizeof(double)*nlambda);
  for(int i = 0; i < nlambda; i++){
    outlambda[i] = spectrum_lambda[i];
    outspectrum[i] = spectrum[i];
  }
  gsl_spline_free (spline);
  gsl_interp_accel_free (acc);
}
void getRealSpectrum(const char* ccd_response, int nlambda, double* lambdas, double* spectrum){
  std::vector<double> ccd_lambda;
  std::vector<double> ccd_rate;
  std::ifstream file_ccd_response;
  file_ccd_response.open(ccd_response);
  double lambda, val;
  while(file_ccd_response>>lambda>>val){
    ccd_lambda.push_back(lambda);
    ccd_rate.push_back(val);
  }
  gsl_interp_accel *acc = gsl_interp_accel_alloc ();
  gsl_spline *spline = gsl_spline_alloc (gsl_interp_cspline, ccd_lambda.size());
  gsl_spline_init (spline, &ccd_lambda[0], &ccd_rate[0], ccd_lambda.size());
  for(int i = 0; i < nlambda; i++){
    if(lambdas[i] < ccd_lambda[0]){
      printf("lambda smaller than ccd curve min %f < %f\n", lambdas[i], ccd_lambda[0]);
      spectrum[i] /= ccd_rate[0];
    }else if(lambdas[i] > ccd_lambda.back()){
      printf("lambda larger than ccd curve max %f > %f\n", lambdas[i], ccd_lambda.back());
      spectrum[i] /= ccd_rate.back();
    }else
    spectrum[i] /= gsl_spline_eval (spline, lambdas[i], acc);
  }
  gsl_spline_free (spline);
  gsl_interp_accel_free (acc);
}
