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

// Process one signal against background with shared alignment
void process_pair(Real* d_bkg, Real* d_sig, int row, int col,
    const char* output_path, int nmerge,
    float user_shift_y = 0.0f, float user_shift_x = 0.0f,
    bool use_fixed_params = false) {
  resize_cuda_image(row, col);

  add(d_sig, d_bkg, -1);  // sig = sig - bkg

  Real midx, midy, shiftx, shifty;
  int outrow, finsize;

  if (!use_fixed_params) {
    // === Center detection on center 1/3 x 1/3 region only ===
    int srow = row / 3;
    int scol = col / 3;

    myCuDMalloc(Real, d_center, (size_t)srow * scol);
    myCuDMalloc(Real, d_bit, (size_t)srow * scol);
    resize_cuda_image(srow, scol);
    crop(d_sig, d_center, row, col);  // extract center 1/3 x 1/3

    Real max = findMax(d_center);
    applyThreshold(d_bit, d_center, max / 2);
    fmt::println("max = {:f} (center {}x{})", max, srow, scol);
    plt.init(srow, scol);
    plt.plotFloat(d_bit, MOD, 0, 1, "bit", 1, 0, 1);

    complexFormat mid = findMiddle(d_bit, srow * scol);
    midx = mid.x * srow;
    midy = mid.y * scol;
    myCuFree(d_bit);
    myCuFree(d_center);
    resize_cuda_image(row, col);

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

  getReal(tmp, tmp1);
  myCuFree(tmp1);

  resize_cuda_image(finsize, finsize);
  mergePixel(d_sig, tmp, outrow, nmerge);
  myCuFree(tmp);
  transpose(d_sig, d_sig);

  plt.init(finsize, finsize);
  myCuDMalloc(complexFormat, xc, finsize * finsize);
  extendToComplex(d_sig, xc);
  init_fft(finsize, finsize);
  myFFT(xc, xc);

  plt.plotFloat(d_sig, MOD, 0, 1, (std::string(output_path) + "_logimage").c_str(), 1, 0, 1);
  plt.plotComplex(xc, MOD2, 1, 1./finsize, (std::string(output_path) + "_autocorrelation").c_str(), 1, 0, 1);

  // Save raw data (plt.saveFloat writes .bin in the correct format)
  plt.saveFloat(d_sig, output_path);
  fmt::println("Saved: {}", output_path);

  myCuFree(xc);
}

int main(int argc, char** argv) {
  if (argc < 7) {
    fmt::println(stderr, "Usage: {} bkg_file pattern_file gratings_file out_pattern_bin out_gratings_bin nmerge [shiftx] [shifty]", argv[0]);
    return 1;
  }

  std::string bkg_file     = argv[1];
  std::string pattern_file = argv[2];
  std::string gratings_file= argv[3];
  std::string out_pattern  = argv[4];
  std::string out_gratings = argv[5];
  int nmerge               = atoi(argv[6]);
  float shift_x            = argc >= 8 ? std::stof(argv[7]) : 0.0f;
  float shift_y            = argc >= 9 ? std::stof(argv[8]) : 0.0f;

  int row, col;

  // Create output directories if needed
  if (fs::path(out_pattern).has_parent_path())
    fs::create_directories(fs::path(out_pattern).parent_path());
  if (fs::path(out_gratings).has_parent_path())
    fs::create_directories(fs::path(out_gratings).parent_path());

  init_cuda_image();
  Real* h_bkg = readImage(bkg_file.c_str(), row, col);
  size_t sz = (size_t)row * col * sizeof(Real);
  myCuDMalloc(Real, d_bkg, sz);
  myMemcpyH2D(d_bkg, h_bkg, sz);
  myFree(h_bkg);

  // Process pattern first — computes the alignment
  Real* h_pattern = readImage(pattern_file.c_str(), row, col);
  myCuDMalloc(Real, d_pattern, sz);
  myMemcpyH2D(d_pattern, h_pattern, sz);
  myFree(h_pattern);
  process_pair(d_bkg, d_pattern, row, col,
      out_pattern.c_str(), nmerge,
      shift_y, shift_x,
      false);  // compute alignment
  myCuFree(d_pattern);

  // Process gratings — reuses alignment from pattern
  Real* h_gratings = readImage(gratings_file.c_str(), row, col);
  myCuDMalloc(Real, d_gratings, sz);
  myMemcpyH2D(d_gratings, h_gratings, sz);
  myFree(h_gratings);
  process_pair(d_bkg, d_gratings, row, col,
      out_gratings.c_str(), nmerge,
      shift_y, shift_x,
      true);  // use cached alignment
  myCuFree(d_gratings);

  myCuFree(d_bkg);
  fmt::println("Done. Alignment computed from pattern and applied to both outputs.");
  return 0;
}
