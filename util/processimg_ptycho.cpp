#include <complex>
#include <fmt/base.h>
#include <stdio.h>
#include <vector>
#include "fmt/core.h"
#include "imgio.hpp"
#include "cudaConfig.hpp"
#include "cuPlotter.hpp"
#include "cub_wrap.hpp"
#include "misc.hpp"
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

// Global parameters determined from first image
struct AlignParams {
  Real midx, midy;
  Real shiftx, shifty;
  int outrow, finsize;
} params;


// Wrap original processing logic — now takes precomputed alignment parameters
void process_pair(Real* d_bkg, Real* d_sig, int row, int col,
    const char* output_path, int nmerge, const char* mode_str,
    float user_shift_y = 0.0f, float user_shift_x = 0.0f,
    bool use_fixed_params = false) {
  size_t sz = row * col * sizeof(Real);
  resize_cuda_image(row, col);

  add(d_sig, d_bkg, -1);  // sig = sig - bkg


  Real midx, midy, shiftx, shifty;
  int outrow, finsize;

  if (!use_fixed_params) {
    // === Run center detection only for first image ===
    myCuDMalloc(Real, d_bit, sz);
    Real max = findMax(d_sig);
    applyThreshold(d_bit, d_sig, max / 2);
    fmt::println("max = {:f}", max);
    plt.init(row, col);
    plt.plotFloat(d_bit, MOD, 0, 1, "bit", 1, 0, 1);  // Only needed once

    std::complex<Real> mid(findMiddle(d_bit, row * col));
    midx = mid.real() * row;
    midy = mid.imag() * col;
    memMngr.returnCache(d_bit);

    // Apply user offset
    if (user_shift_x != 0 || user_shift_y != 0) {
      midx += user_shift_x;
      midy -= user_shift_y;
    }

    shiftx = int(midx) - midx;
    shifty = int(midy) - midy;
    fmt::println("mid= {:f},{:f}", midx, midy);
    fmt::println("shift = {:f},{:f}", shiftx, shifty);

    int step = nmerge * 4;
    outrow = (row - int(std::abs(midx)) * 2) / step * step;
    int outcol = (col - int(std::abs(midy)) * 2) / step * step;
    outrow = std::min(outrow, outcol);
    finsize = outrow / nmerge;

    // Save parameters for reuse
    params.midx = midx;
    params.midy = midy;
    params.shiftx = shiftx;
    params.shifty = shifty;
    params.outrow = outrow;
    params.finsize = finsize;
  } else {
    // === Reuse cached parameters ===
    midx = params.midx;
    midy = params.midy;
    shiftx = params.shiftx;
    shifty = params.shifty;
    outrow = params.outrow;
    finsize = params.finsize;
  }

  // === Shared processing path below ===
  myCuDMalloc(Real, tmp, outrow * outrow);
  myCuDMalloc(complexFormat, tmp1, outrow * outrow);

  resize_cuda_image(outrow, outrow);
  crop(d_sig, tmp, row, col, midx/row, midy/col);
  extendToComplex(tmp, tmp1);
  init_fft(outrow, outrow);

  if (mode_str[0] == '1') {
    shiftMiddle(tmp1);
  } else {
    //shiftWave(tmp1, shiftx, shifty);
  }

  getReal(tmp, tmp1);
  memMngr.returnCache(tmp1);

  resize_cuda_image(finsize, finsize);
  mergePixel(d_sig, tmp, outrow, nmerge);
  memMngr.returnCache(tmp);
  transpose(d_sig, d_sig);

  plt.init(finsize, finsize);
  myCuDMalloc(complexFormat, xc, finsize * finsize);
  extendToComplex(d_sig, xc);
  init_fft(finsize, finsize);
  myFFT(xc, xc);

  plt.plotFloat(d_sig, MOD, 0, 1, (std::string(output_path) + "_logimage").c_str(), 1, 0, 1);
  plt.plotComplex(xc, MOD2, 1, 1./finsize, (std::string(output_path) + "_autocorrelation").c_str(), 1, 0, 1);
  //initCub();
  //fmt::println("Summation: {}.", size_t(findSum(d_sig)*65535));

  plt.saveFloat(d_sig, output_path);

  memMngr.returnCache(xc);
}

int main(int argc, char** argv) {
  if (argc < 5) {
    fmt::println(stderr, "Usage: {} bkg_dir sig_dir out_dir nmerge [mode] [dy] [dx]", argv[0]);
    return 1;
  }

  std::string input_dir   = argv[1];
  std::string out_dir   = argv[2];
  int nmerge            = atoi(argv[3]);
  const char* mode_str  = argc >= 6 ? argv[4] : "0";
  float shift_y         = argc >= 7 ? std::stof(argv[5]) : 0.0f;
  float shift_x         = argc >= 8 ? std::stof(argv[6]) : 0.0f;


  int row, col;
  fs::create_directories(fs::path(out_dir));

  init_cuda_image();
  Real* host_img = readImage((input_dir+"/bkg.png").c_str(), row, col);
  resize_cuda_image(row, col);
  int sz = row * col * sizeof(Real);
  myCuDMalloc(Real, d_bkg, sz);
  myMemcpyH2D(d_bkg, host_img, sz);
  // === Step 2: Process each signal image using fixed alignment after first ===
  myCuDMalloc(Real, d_sig, sz);

  bool firstSig = false;
  for (const auto& ent : fs::directory_iterator(input_dir)) {
    if (ent.path().extension() != ".png") continue;

    std::string sig_path = ent.path().string();
    std::string base_name = ent.path().stem().string();
    if(base_name == "bkg") continue;
    std::string out_path = (fs::path(out_dir) / base_name).string();

    Real* sig = readImage(sig_path.c_str(), row, col);
    myMemcpyH2D(d_sig, sig, sz);
    ccmemMngr.returnCache(sig);

    // Pass whether we should use fixed (cached) alignment params
    process_pair(d_bkg, d_sig, row, col,
        out_path.c_str(),
        nmerge, mode_str,
        shift_y, shift_x,
        firstSig);  // use_fixed_params = true for non-first

    if (!firstSig) firstSig = true;
  }

  memMngr.returnCache(d_bkg);
  return 0;
}
