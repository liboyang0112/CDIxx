#include <fmt/base.h>
#include <pthread.h>
#include <filesystem>
#include <fmt/core.h>
#include <string>
#include "cuPlotter.hpp"
#include "fmt/core.h"
#include "memManager.hpp"
#include "cudaConfig.hpp"
#include "videoWriter.hpp"
#include "imgio.hpp"
extern "C" {
#include "freetype.hpp"
#include "preview.h"
}
const float TurboData[3][8][3] = {  //color, sector, fit parameterx : a+b*x+c*x*x
  {
    {48.628736, 1.26493952, 0.0179480064},
    {12.9012, 3.17599, -0.0432176},
    {550.736, -13.3905, 0.0850816},
    {-377.035, 4.95565, -0.00546875},
    {-461.663, 6.91765, -0.0158418},
    {-904.194, 12.7805, -0.0352175},
    {-397.815, 7.43316, -0.0210902},
    {-513.745, 8.50106, -0.0235486}
  },{
     {18.3699, 2.98435, -0.00590619},
     {25.5339, 2.60115, -0.000838337},
     {-65.9885, 5.48243, -0.023616},
     {-90.9164, 5.82642, -0.024485},
     {-47.1592, 5.39126, -0.0237502},
     {-65.915, 5.26759, -0.0222375},
     {1648.81, -12.6094, 0.0243797},
     {1160.21, -8.10854, 0.014018}
  },{
     {59.4351, 7.57781, -0.0720408},
     {29.4288, 9.18296, -0.0932682},
     {391.685, -2.3166, -0.0016649},
     {770.071, -8.98098, 0.0266769},
     {606.475, -7.75046, 0.0270971},
     {-638.138, 8.68843, -0.0270811},
     {1017.44, -8.76258, 0.0189386},
     {625.18, -5.14916, 0.0106199}
  }
};
cuPlotter plt;
void getTurboColor(Real x, int bit_depth, char* store){
  //convert to 8 bit;
  x = x / (1<<(bit_depth-8));
  if(x > 255) x = 255;
  int sec = x/32; // deside sector;
  for(int i = 0 ; i < 3; i++){
    int val = TurboData[i][sec][0] + TurboData[i][sec][1]*x + TurboData[i][sec][2]*x*x;
    if(val < 0) val = 0;
    if(val > 255) val = 255;
    store[i] = val;
  }
}

int cuPlotter::initVideo(const char* filename, int fps, bool preview, bool online){
  int handle = nvid;
  if(handle == 100){
    for(int i = 0; i < 100; i++){
      if(videoWriterVec[i] == 0){
        handle = i;
      }
    }
  }else
    nvid++;
  if(handle==100) {
    fmt::print("You created too many videos (100), please release some before create new ones");
    exit(0);
  }
  auto strfname = std::string(filename);
  strfname.erase(strfname.size() - 4);
  if(online){
    videoWriterVec[handle] = createVideo(filename, rows, cols, fps, ("mjpeg:///tmp/CDIxx_" + strfname + ".sock").c_str());
  }else {
    videoWriterVec[handle] = createVideo(filename, rows, cols, fps, NULL);
    if(preview) {
      videoPreview[handle] = createPreview(filename, rows, cols);
    }
  }
  toVideo = handle;
  return handle;
}
void cuPlotter::saveVideo(int handle){
  if(vid_thread) {
    pthread_join(vid_thread, nullptr);
    vid_thread = 0;
  }
  ::saveVideo(videoWriterVec[handle]);
}
bool ensureDirExists(const std::string& path) {
    if (!path.empty()) {
        try {
            if (!std::filesystem::exists(path)) {
                if (std::filesystem::create_directories(path)) {
                    return true; // Successfully created
                } else {
                    return false; // Failed to create (permissions, etc.)
                }
            } else if (std::filesystem::is_directory(path)) {
                return true; // Already exists and is a directory
            } else {
                return false; // Path exists but is not a directory
            }
        } catch (const std::filesystem::filesystem_error&) {
            return false; // Handle error (e.g. permission denied)
        }
    }
    return false; // Empty path or other invalid input
}
void cuPlotter::init(int rows_, int cols_, const char* prefix_){
  if(rows==rows_ && cols==cols_) return;
  freeMem();
  fmt::println("init plot {}, {}",rows_,cols_);
  rows=rows_;
  cols=cols_;
  cv_cache = (uint8_t*)ccmemMngr.borrowCache(rows*cols*3);
  cv_data = ccmemMngr.borrowCache(rows*cols*sizeof(pixeltype));
  cuCache_data = memMngr.borrowCache(rows*cols*3);
  if (prefix) {        // Free previous string if reinitializing
    free(const_cast<char*>(prefix));
  }
  prefix = prefix_ ? strdup(prefix_) : nullptr;
  if(!ensureDirExists(prefix)){
    fmt::print(stderr, "Director failed to create: {}, please check permission or if it's already exists as a file\n", prefix);
  }
}

void cuPlotter::freeMem(){
  if(cv_cache) {
    ccmemMngr.returnCache(cv_cache);
    ccmemMngr.returnCache(cv_data);
    cv_cache = 0;
    cv_data = 0;
  }
  if(cv_float_data) {
    ccmemMngr.returnCache(cv_float_data);
    cv_float_data = 0;
  }
  if(cv_complex_data){
    ccmemMngr.returnCache(cv_complex_data);
    cv_complex_data = 0;
  }
  if(cuCache_data) { memMngr.returnCache(cuCache_data); cuCache_data = 0;}
}

void* cuPlotter::processFloatData(void* cudaData, const mode m, bool isFrequency, Real decay, bool islog, bool isFlip){
  if(!cuCache_data) cuCache_data = (pixeltype*) memMngr.borrowCache(rows*cols*sizeof(pixeltype));
  process<Real>(cudaData, (pixeltype*)cuCache_data, m, isFrequency, decay, islog, isFlip);
  myMemcpyD2H(cv_data, cuCache_data,rows*cols*sizeof(pixeltype));
  return cv_data;
};
void* cuPlotter::processComplexData(void* cudaData, const mode m, bool isFrequency, Real decay, bool islog, bool isFlip){
  if(!cuCache_data) {
    cuCache_data = (pixeltype*) memMngr.borrowCache(rows*cols*sizeof(pixeltype));
  }
  process<complexFormat>(cudaData, (pixeltype*)cuCache_data, m,isFrequency, decay, islog, isFlip);
  myMemcpyD2H(cv_data, cuCache_data,rows*cols*sizeof(pixeltype));
  return cv_data;
};
void* cuPlotter::processComplexColor(void* cudaData, bool isFrequency, Real decay, bool islog, bool isFlip){
  if(vid_thread) {
    pthread_join(vid_thread, nullptr);
    vid_thread = 0;
  }
  if(!cuCache_data) {
    cuCache_data = (col_rgb*) memMngr.borrowCache(rows*cols*sizeof(col_rgb));
  }
  process_rgb(cudaData, (col_rgb*)cuCache_data, isFrequency, decay, islog, isFlip);
  myMemcpyD2H(cv_cache, cuCache_data,rows*cols*sizeof(col_rgb));
  return cv_cache;
};
void* flushVideo_thread(void* args){
  cuPlotter* _plt = (cuPlotter*)args;
  flushVideo(_plt->videoWriterVec[_plt->writeVideo], _plt->cv_cache);
  return nullptr;
}
void cuPlotter::plotComplexColor(void* cudaData, bool isFrequency, Real decay, const char* label,bool islog, bool isFlip){
  if(vid_thread) {
    pthread_join(vid_thread, nullptr);
    vid_thread = 0;
  }
  processComplexColor(cudaData,isFrequency,decay,islog,isFlip);
  char color[3] = {-1,0,0};
  if(label) putText(label, 0, rows-1, rows, cols, cv_cache, 1, color);
  if(toVideo>=0) {
    writeVideo = toVideo;
    pthread_create(&vid_thread, nullptr, flushVideo_thread, this);
    if(videoPreview[toVideo]) updatePreview(videoPreview[toVideo], cv_cache);
  }else
    writePng((prefix + std::string(label)+".png").c_str(), cv_cache, rows, cols, 8, 1);
}
void cuPlotter::plotComplex(void* cudaData, mode m, bool isFrequency, Real decay, const char* label,bool islog, bool isFlip, bool isColor, const char* caption){
  if(vid_thread) {
    pthread_join(vid_thread, nullptr);
    vid_thread = 0;
  }
  processComplexData(cudaData,m,isFrequency,decay,islog,isFlip);
  plot(label, isColor, caption);
}
void cuPlotter::plotFloat(void* cudaData, mode m, bool isFrequency, Real decay, const char* label,bool islog, bool isFlip, bool isColor, const char* caption){
  if(vid_thread) {
    pthread_join(vid_thread, nullptr);
    vid_thread = 0;
  }
  processFloatData(cudaData,m,isFrequency,decay,islog,isFlip);
  plot(label, isColor, caption);
}
void cuPlotter::saveComplex(void* cudaData, const char* label){
  if(!cv_complex_data){
    cv_complex_data = ccmemMngr.borrowCache(rows*cols*sizeof(Real)*2);
  }
  myMemcpyD2H(cv_complex_data, cudaData, rows*cols*sizeof(complexFormat));
  writeComplexImage((std::string(label)+".bin").c_str(), cv_complex_data, rows, cols);
}
void cuPlotter::saveFloat(void* cudaData, const char* label){
  if(!cv_float_data){
    cv_float_data = ccmemMngr.borrowCache(rows*cols*sizeof(Real));
  }
  myMemcpyD2H(cv_float_data, cudaData, rows*cols*sizeof(Real));
  writeFloatImage((prefix + std::string(label)+".bin").c_str(), cv_float_data, rows, cols);
}
void cuPlotter::videoFrame(void* cudaData){
  if(!cv_float_data){
    cv_float_data = ccmemMngr.borrowCache(rows*cols*sizeof(Real));
  }
  myMemcpyD2H(cv_float_data, cudaData, rows*cols*sizeof(Real));
  flushVideo_float(videoWriterVec[toVideo], cv_float_data);
}
void* cuPlotter::cvtTurbo(void* icache){
  char* cache = (char*)(icache?icache:cv_cache);
  for(int i = 0 ; i < rows*cols; i++){
    getTurboColor(((pixeltype*)cv_data)[i], CCDBits, cache);
    cache+=3;
  }
  return cv_cache;
}
void* cuPlotter::cvt8bit(void* icache){
  char* cache = (char*)(icache?icache:cv_cache);
  for(int i = 0 ; i < rows*cols; i++){
    *cache = *(cache+1) = *(cache+2) = ((pixeltype*)cv_data)[i] >> (CCDBits-8);
    cache+=3;
  }
  return cv_cache;
}
void cuPlotter::plot(const char* label, bool iscolor, const char* caption){
  std::string fname = label;
  if(fname.find(".")==std::string::npos) fname+=".png";
  if(iscolor){
    if(vid_thread) {
      pthread_join(vid_thread, nullptr);
      vid_thread = 0;
    }
    cvtTurbo();
    char color[3] = {-1,0,0};
    if(caption) putText(caption, 0, rows-1, rows, cols, cv_cache, 1, color);
    if(toVideo>=0) {
      //put_formula(label, 0,0,cols, cv_cache, 1, color);
      pthread_create(&vid_thread, nullptr, flushVideo_thread, this);
      if(videoPreview[toVideo]) updatePreview(videoPreview[toVideo], cv_cache);
      return;
    }
    else if(std::string(fname).find(".png")!=std::string::npos)
      writePng((prefix + fname).c_str(), cv_cache, rows, cols, 8, 1);
    else if(std::string(fname).find(".jpg")!=std::string::npos)
      writeJPEG((prefix + fname).c_str(), cv_cache, rows, cols, 25);
  }else{
    pixeltype color = -1;
    if(caption) putText(caption, 0, rows-1, rows, cols, cv_data, 0, &color);
    writePng((prefix + fname).c_str(), cv_data, rows, cols, CCDBits, 0);
  }
  fmt::println("written to file {}", prefix + fname);
}
cuPlotter::~cuPlotter(){
  if (prefix) {
    free(const_cast<char*>(prefix));
  }
  freeMem();
}

