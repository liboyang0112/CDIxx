#include "memManager.h"
#include "common.h"
#include "imageFile.h"
#include "fstream"
#include <gsl/gsl_spline.h>
#include <tiffio.h>
#include <png.h>
#include <setjmp.h>
#include <vector>
/*
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;
Real* readImage_old(const char* name, int &row, int &col){
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
*/
const Real rgb2gray[3] = {0.299,0.587,0.114};
Real* readImage(const char* name, int &row, int &col){
  string fext = string(name).substr(string(name).find_last_of(".")+1);
  Real* ret = 0;
  if(fext == "bin"){
    FILE* fin = fopen(name, "r");
    imageFile fdata;
    fread(&fdata, sizeof(imageFile), 1, fin);
    row = fdata.rows;
    col = fdata.cols;
    size_t datasz = row*col*typeSizes[fdata.type];
    ret = (Real*) ccmemMngr.borrowCache(datasz);
    fread(ret, datasz, 1, fin);
    int ttype = REALIDX;
    if(fdata.type != ttype){  //only save floats with bin;
      fprintf(stderr, "ERROR: FILETYPE unrecognized: %d\n", fdata.type);
      abort();
    }
    fclose(fin);
  }else if(fext == "tiff"){
    TIFF* tif = TIFFOpen(name, "r");
    if(!tif) {
      fprintf(stderr, "ERROR: %s not fould!\n", name);
      abort();
    }
    uint16_t nchann, typesize;
    if(TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &nchann)) {
      fprintf(stderr, "ERROR: get type failed with tiff file %s!\n", name);
      abort();
    }
    if(nchann != 0){  //uint
      printf("File format: %d\n", nchann);
      fprintf(stderr, "ERROR: Please use .bin file to save float or complex image\n");
      abort();
    }
    if(TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &typesize)) {
      //fprintf(stderr, "ERROR: get typesize failed with tiff file %s!\n", name);
      //abort();
    }
    if(TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &nchann)) {
      //fprintf(stderr, "ERROR: get nchann failed with tiff file %s!\n", name);
      //abort();
    }
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &row);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &col);
    tdata_t buf = _TIFFmalloc(TIFFScanlineSize(tif));
    ret = (Real*) ccmemMngr.borrowCache(row*col*sizeof(Real));
    for(int i = 0; i < col; i++){
      TIFFReadScanline(tif, buf, i, 0);
      int idx = i*row;
      for(int j = 0; j < row; j++){
        if(nchann == 1) {
          Real val;
          if(typesize==8) val = Real(((unsigned char*)buf)[j])/255;
          else val = Real(((uint16_t*)buf)[j])/65535;
          ret[idx+j] = val;
        }else if(nchann == 3) {
          Real val;
          ret[idx+j] = 0;
          for(int ic = 0; ic < 3; ic++){
            if(typesize==8) val = Real(((unsigned char*)buf)[3*j+ic])/255;
            else val = Real(((uint16_t*)buf)[3*j+ic])/65535;
            ret[idx+j] += val*rgb2gray[ic];
          }
        }
      }
    }
    TIFFClose(tif);
  }else if(fext == "png"){
    FILE *f = fopen(name, "rb");
    if (f == NULL){
      fprintf(stderr, "pngpixel: %s: could not open file\n", name);
      abort();
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,
        NULL, NULL, NULL);
  
    if (png_ptr == NULL){
      fprintf(stderr, "pngpixel: out of memory allocating png_struct\n");
      abort();
    }
    png_infop info_ptr = png_create_info_struct(png_ptr);
  
    if (info_ptr == NULL){
      fprintf(stderr, "pngpixel: out of memory allocating png_info\n");
      abort();
    }
    if (setjmp(png_jmpbuf(png_ptr)) != 0)
    {
      printf("setjmp error!\n");
      abort();
    }
    int bit_depth, color_type, interlace_method,
        compression_method, filter_method;
    png_init_io(png_ptr, f);
    png_read_info(png_ptr, info_ptr);
    png_bytep rowbuf = (png_bytep)png_malloc(png_ptr, png_get_rowbytes(png_ptr, info_ptr));
    if (!png_get_IHDR(png_ptr, info_ptr, (unsigned int*)&row, (unsigned int*)&col,
          &bit_depth, &color_type, &interlace_method,
          &compression_method, &filter_method)){
      png_error(png_ptr, "pngpixel: png_get_IHDR failed");
      abort();
    }
    png_start_read_image(png_ptr);
    unsigned int typesize = png_get_bit_depth(png_ptr, info_ptr);
    if(typesize == 16) png_set_swap(png_ptr);
    int colortype = png_get_color_type(png_ptr, info_ptr);
    int nchann;
    if(colortype == PNG_COLOR_TYPE_GRAY) nchann = 1;
    else if(colortype == PNG_COLOR_TYPE_RGB) nchann = 3;
    else{
      fprintf(stderr, "ERROR: color type not know\n");
      abort();
    }
    ret = (Real*) ccmemMngr.borrowCache(row*col*sizeof(Real));
    for(int i = 0; i < col; i++){
      png_read_row(png_ptr, rowbuf, NULL);
      int idx = i*row;
      for(int j = 0; j < row; j++){
        if(nchann == 1) {
          Real val;
          if(typesize==8) val = Real(((unsigned char*)rowbuf)[j])/255;
          else val = Real(((uint16_t*)rowbuf)[j])/65535;
          ret[idx+j] = val;
        }else if(nchann == 3) {
          Real val;
          ret[idx+j] = 0;
          for(int ic = 0; ic < 3; ic++){
            if(typesize==8) val = Real(((unsigned char*)rowbuf)[3*j+ic])/255;
            else val = Real(((uint16_t*)rowbuf)[3*j+ic])/65535;
            ret[idx+j] += val*rgb2gray[ic];
          }
        }
      }
    }
    png_free(png_ptr, rowbuf);
    png_destroy_info_struct(png_ptr, &info_ptr);
    png_destroy_read_struct(&png_ptr, NULL, NULL);
  }else{
    fprintf(stderr, "ERROR: file extension .%s not know\n", fext.c_str());
    abort();
  }
  return ret;
}

void writeComplexImage(const char* name, void* data, int row, int column){
  FILE* fout = fopen(name, "w");
  imageFile fdata;
  fdata.type = COMPLEXIDX;
  fdata.rows = row;
  fdata.cols = column;
  fwrite(&fdata, sizeof(imageFile), 1, fout);
  fwrite(data, row*column*sizeof(Real)*2, 1, fout);
}

void getNormSpectrum(const char* fspectrum, const char* ccd_response, Real &startLambda, Real &endLambda, int &nlambda, double *& outlambda, double *& outspectrum){
  std::vector<double> spectrum_lambda;
  std::vector<double> spectrum;
  std::vector<double> ccd_lambda;
  std::vector<double> ccd_rate;
  std::ifstream file_spectrum, file_ccd_response;
  std::ofstream file_out("spectTccd.txt");
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
    //if(lambda >= 940) ccd_rate_i*=2;
    //if(lambda < 800) ccd_rate_i *= 0.9;
    spectrum[nlambda] = spectrum[i]/maxval*ccd_rate_i;
    nlambda++;
  }
  endLambda /= startLambda;
  outlambda = (double*) ccmemMngr.borrowCache(sizeof(double)*nlambda);
  outspectrum = (double*) ccmemMngr.borrowCache(sizeof(double)*nlambda);
  for(int i = 0; i < nlambda; i++){
    outlambda[i] = spectrum_lambda[i];
    outspectrum[i] = spectrum[i];
    file_out << spectrum_lambda[i]*startLambda<<" "<<spectrum[i]<<std::endl;
  }
  file_out.close();
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
  if(0)
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
