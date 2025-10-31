#include "cudaConfig.hpp"
#include "cudaDefs_h.cu"
#include <cmath>
#include <complex.h>
#include <fmt/base.h>

__forceinline__ __device__ void pixel_circle_coverage_approx(
    Real dx, Real dy,  // rectangle center relative to circle (which is at origin)
    Real r,
    Real& area,        // output: covered area of pixel
    Real& cent_x,      // centroid x, relative to circle (origin)
    Real& cent_y       // centroid y, relative to circle (origin)
) {
    Real d_sq = dx * dx + dy * dy;
    cent_x = dx;
    cent_y = dy;
    if (d_sq < 1.0e-12f) {
        area = 1.0f;
        return;
    }
    Real d_inv = rsqrtf(d_sq);
    Real d_norm = d_sq * d_inv;  // |vec| = sqrt(d_sq)
    if (d_norm >= r+0.7071067811865475 || fabs(dx) >= r+0.5 || fabs(dy) >= r+0.5){
        area = 0.f;
        return;
    }
    if (d_norm <= r-0.7071067811865475){
        area = 1.f;
        return;
    }
    Real nx = -dx * d_inv;
    Real ny = -dy * d_inv;
    Real c_line = -r - (nx * dx + ny * dy);
    Real corners_x[4] = {-0.5f, 0.5f, 0.5f, -0.5f};
    Real corners_y[4] = {-0.5f, -0.5f, 0.5f, 0.5f};
    int num_in = 0;
    for (int i = 0; i < 4; ++i) {
        Real val = nx * corners_x[i] + ny * corners_y[i];
        num_in += (val >= c_line);  // bool → int: efficient!
    }
    if (num_in == 0) {
        area = 0.0f;
        return;
    }
    if (num_in == 4) {
        area = 1.0f;
        return;
    }
    Real px[6], py[6];
    int n_out = 0;
    for (int i = 0; i < 4; ++i) {
        int j = (i + 3) % 4;
        Real x1 = corners_x[j], y1 = corners_y[j];
        Real x2 = corners_x[i], y2 = corners_y[i];
        Real d1 = nx * x1 + ny * y1 - c_line;
        Real d2 = nx * x2 + ny * y2 - c_line;
        bool in1 = (d1 >= 0.0f);
        bool in2 = (d2 >= 0.0f);
        if (in1 != in2) {
            Real t = d1 / (d1 - d2);
            t = fmaxf(0.0f, fminf(1.0f, t));
            px[n_out] = x1 + t * (x2 - x1);
            py[n_out] = y1 + t * (y2 - y1);
            ++n_out;
        }
        if (in2) {
            px[n_out] = x2;
            py[n_out] = y2;
            ++n_out;
        }
    }
    if (n_out < 3) {
        area = 0.0f;
        return;
    }
    Real A = 0.0f;
    Real Cx = 0.0f;
    Real Cy = 0.0f;
    for (int i = 0; i < n_out; ++i) {
        int j = (i + 1) % n_out;
        Real cross = px[i] * py[j] - px[j] * py[i];
        A += cross;
        Cx += (px[i] + px[j]) * cross;
        Cy += (py[i] + py[j]) * cross;
    }
    area = fabsf(A*0.5);
    if (area > 1.0e-10f) {
      cent_x += Cx / (6.0f * A);
      cent_y += Cy / (6.0f * A);
    }
}
__forceinline__ __device__ Real Hermit(Real x,int n) {
    if (n < 0) return 0;
    Real e = expf(-0.5f * x*x), p0 = 0.751127f * e;
    if (n == 0) return p0;
    Real p1 = 1.224744f * x * p0;
    if (n == 1) return p1;
    Real p = p1;
    for (int k = 2; k <= n; ++k) {
        p = sqrtf(2.f/k) * x * p1 - sqrtf((k-1.f)/k) * p0;
        p0 = p1; p1 = p;
    }
    return p;
}

__forceinline__ __device__ cuComplex multiplyAM(Real r, Real theta) {
  Real c, s;
  sincosf(theta, &s, &c);
  return make_cuComplex(r * c, r * s);
}
__forceinline__ __device__ Real laguerre_gaussian_R(Real z, int p, int m) {
  const int alpha = (m > 0) ? m : -m;
  if (z == 0.0f && alpha > 0) return 0.0f;
  Real f0 = 0.7978845608028654f*expf(-0.5f * z);
  for (int i = 1; i <= alpha; ++i) f0 *= rsqrtf(i);
  for (int i = 0; i < (alpha>>1); ++i) f0 *= z;
  if(alpha & 1) f0 *= sqrtf(z);
  if (p == 0) return f0;
  Real fact = rsqrtf((1 + alpha));
  Real f1 = f0*(1.0f + alpha - z)*fact;
  for (int k = 2; k <= p; ++k) {
    Real coef2 = 1./fact;
    fact = rsqrtf(k * (k + alpha));
    coef2 *= fact;
    Real coef1 = (2.0f*k + alpha - 1.0f - z) * fact;
    Real f = __fmaf_rn(coef1, f1, -coef2 * f0);
    f0 = f1;
    f1 = f;
  }
  return f1;
}
__forceinline__ __device__ cuComplex laguerre_gaussian(Real x, Real y, int p, int m) {
  return multiplyAM(laguerre_gaussian_R(2*(x*x + y*y), p, m), m*atan2f(y, x));
}
__forceinline__ __device__ Real zernike_R(Real z, int n, int m_abs) {
  Real r = sqrtf(z);
  Real R_prev = powf(r,m_abs);
  if(n == m_abs) return R_prev;
  Real R_n = R_prev * ((m_abs + 2) * z - (m_abs + 1));
  if (n == m_abs + 2) return  R_n;
  for (int i = m_abs+4; i <= n; i += 2) {
    Real K1 = ((i + m_abs) * (i - m_abs) * (i - 2.0f)) * 0.5f;
    Real K2 = 2.0f * i * (i - 1.0f) * (i - 2.0f);
    Real K3 = -m_abs * m_abs * (i - 1.0f) - i * (i - 1.0f) * (i - 2.0f);
    Real K4 = -0.5f * i * (i + m_abs - 2.0f) * (i - m_abs - 2.0f);
    Real numerator = (K2 * r * r + K3) * R_n + K4 * R_prev;
    R_prev = R_n;
    R_n = numerator/K1;
  }
  return R_n;
}
__forceinline__ __device__ cuComplex zernike_complex(Real x, Real y, int n, int m) {
  Real z = x*x+y*y;
  if (z > 1.0f || n < 0) return cuComplex();
  const int m_abs = (m < 0) ? -m : m;
  if (m_abs > n || (n - m_abs) % 2 != 0)
    return cuComplex();
  return multiplyAM(zernike_R(z, n, m_abs) * sqrtf(n + 1), m*atan2f(y, x));
}
__forceinline__ __device__ cuComplex zernike_complex_polar(Real r, Real theta, int n, int m) {
  Real z = r*r;
  const int m_abs = (m < 0) ? -m : m;
  if (m_abs > n || (n - m_abs) % 2 != 0)
    return cuComplex();
  return multiplyAM(zernike_R(z, n, m_abs) * sqrtf(n + 1), m*theta);
}
cuFuncc(multiplyHermit,(complexFormat* store, complexFormat* data, Real pupilsize, int n, int m),(cuComplex* store, cuComplex* data, Real pupilsize, int n, int m),((cuComplex*)store, (cuComplex*)data,pupilsize, n, m),{
    cudaIdx()
    Real xp = Real(x - (cuda_row>>1))/pupilsize;
    Real yp = Real(y - (cuda_column>>1))/pupilsize;
    Real factor = Hermit(xp, n) * Hermit(yp, m);
    store[index].x = factor*data[index].x;
    store[index].y = factor*data[index].y;
    })

cuFunc(multiplyHermit,(Real* store, Real* data, Real pupilsize, int n, int m),(store, data,pupilsize, n, m),{
    cudaIdx()
    Real xp = Real(x - (cuda_row>>1))/pupilsize;
    Real yp = Real(y - (cuda_column>>1))/pupilsize;
    Real factor = Hermit(xp, n) * Hermit(yp, m);
    store[index] = factor*data[index];
    })

cuFuncc(multiplyZernikeConj,(complexFormat* store,complexFormat* data, Real pupilsize, int n, int m),(cuComplex* store, cuComplex* data, Real pupilsize, int n, int m),((cuComplex*)store, (cuComplex*)data,pupilsize, n, m),{
    cudaIdx()
    Real dx = x - Real(cuda_row-1)/2;
    Real dy = y - Real(cuda_column-1)/2;
    Real centx, centy, factor;
    pixel_circle_coverage_approx(dx, dy, pupilsize, factor, centx, centy);
    Real dr = sqrtf(centx*centx+centy*centy);
    if(n < 0 || factor < 2e-20) {
      store[index].x = store[index].y = 0;
      return;
    }
    cuComplex out = cuCmulf(data[index], cuConjf(zernike_complex_polar(fminf(dr/pupilsize,1), atan2f(centy, centx), n, m)));
    store[index].x = out.x;
    store[index].y = out.y;
    })

cuFuncc(multiplyLaguerreConj,(complexFormat* store,complexFormat* data, Real pupilsize, int n, int m),(cuComplex* store, cuComplex* data, Real pupilsize, int n, int m),((cuComplex*)store, (cuComplex*)data,pupilsize, n, m),{
    cudaIdx()
    store[index] = cuCmulf(data[index], cuConjf(laguerre_gaussian(Real(x - (cuda_row>>1))/pupilsize, Real(y - (cuda_column>>1))/pupilsize, n, m)));
    })

cuFuncc(addZernike,(complexFormat* store, complexFormat coefficient, Real pupilsize, int n, int m),(cuComplex* store, cuComplex coefficient, Real pupilsize, int n, int m),((cuComplex*)store, *(cuComplex*)&coefficient, pupilsize, n, m),{
    cudaIdx()
    Real dx = x - Real(cuda_row-1)/2;
    Real dy = y - Real(cuda_column-1)/2;
    Real factor, centx, centy;
    pixel_circle_coverage_approx(dx, dy, pupilsize, factor, centx, centy);
    if(n < 0 || factor < 2e-20) return;
    Real dr = sqrtf(centx*centx+centy+centy);
    cuComplex out = cuCmulf(coefficient, zernike_complex_polar(fminf(dr/pupilsize,1), atan2f(centy, centx), n, m));
    store[index].x += out.x*factor;
    store[index].y += out.y*factor;
    })

cuFuncc(addLaguerre,(complexFormat* store, complexFormat coefficient, Real pupilsize, int n, int m),(cuComplex* store, cuComplex coefficient, Real pupilsize, int n, int m),((cuComplex*)store, *(cuComplex*)&coefficient, pupilsize, n, m),{
    cudaIdx()
    store[index] = cuCaddf(store[index], cuCmulf(coefficient, laguerre_gaussian(Real(x - (cuda_row>>1))/pupilsize, Real(y - (cuda_column>>1))/pupilsize, n, m)));
    })


struct ZernikeHandle {
  int maxN;
  int nmodes;
  int nblocks;
  int width, height;
  size_t shared_mem_size;
  cuComplex* block_coeff;
  cuComplex* final_coeff;
  complexFormat* host_coeff;
  int nthread;
  bool initialized;
};

void* zernike_init(int width, int height, int maxN, int max_blocks) {
  ZernikeHandle* handle = new ZernikeHandle();
  handle->width = width;
  handle->height = height;
  handle->maxN = maxN;
  handle->nmodes = (maxN + 1) * (maxN + 2) / 2;
  handle->nthread = 128;
  int npixels = width * height;
  handle->nblocks = (npixels + handle->nthread-1) / handle->nthread;
  if (max_blocks > 0 && handle->nblocks > max_blocks) handle->nblocks = max_blocks;
  handle->shared_mem_size = (handle->nthread + 1) * 14 * 2 * sizeof(Real);

  cudaError_t err1 = cudaMalloc(&handle->block_coeff, handle->nmodes * handle->nblocks * sizeof(cuComplex));
  cudaError_t err2 = cudaSuccess;
  if (err1 == cudaSuccess) {
    handle->host_coeff = (complexFormat*)malloc(handle->nmodes * sizeof(complexFormat));
    err2 = cudaMalloc(&handle->final_coeff, handle->nmodes * sizeof(cuComplex));
  }
  handle->initialized = (err1 == cudaSuccess && err2 == cudaSuccess);
  if (!handle->initialized) {
    if (err1 == cudaSuccess) cudaFree(handle->block_coeff);
    delete handle;
    return nullptr;
  }
  return static_cast<void*>(handle);
}

complexFormat* zernike_coeff(void* handle_ptr) {
  if (!handle_ptr) return nullptr;
  ZernikeHandle* handle = static_cast<ZernikeHandle*>(handle_ptr);
  if (!handle->initialized) return nullptr;
  cudaMemcpy(handle->host_coeff, 
      handle->final_coeff, 
      handle->nmodes * sizeof(complexFormat), 
      cudaMemcpyDeviceToHost);
  return handle->host_coeff;
}

void zernike_destroy(void* handle_ptr) {
  if (!handle_ptr) return;
  ZernikeHandle* handle = static_cast<ZernikeHandle*>(handle_ptr);
  if (handle->initialized) {
    cudaFree(handle->block_coeff);
    cudaFree(handle->final_coeff);
    free(handle->host_coeff);
  }
  delete handle;
}
__global__ void zernike_project_mode(
    const cuComplex* phi,
    int width, int height,
    Real cx, Real cy, Real norm,
    int maxN,
    cuComplex* block_coeff_mode
    ) {
  extern __shared__ char shared_raw[];
  const int modes_in_shared_mem = 14;  // 必须与 zernike_init 中的 shared_mem_size 一致
  const int SHMEM_STRIDE = blockDim.x + 1;
  Real* shared_real = (Real*)shared_raw;
  Real* shared_imag = shared_real + SHMEM_STRIDE * modes_in_shared_mem;

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;

  Real rho = 10.0f, theta = 0.0f;
  Real rho_pow = 1.0f;
  cuComplex phi_val = make_cuComplex(0.0f, 0.0f);
  if (gid < width * height) {
    int x = gid % width;
    int y = gid / width;
    Real dx = x - cx;
    Real dy = y - cy;
    pixel_circle_coverage_approx(dx, dy, norm, rho_pow, dx, dy);
    rho = sqrtf(dx*dx + dy*dy);
    if (rho_pow > 2e-20) {
      rho = fminf(rho/norm, 1);
      theta = atan2f(dy, dx);
      phi_val = phi[gid];
    }
  }

  int nmodes = (maxN + 1) * (maxN + 2) / 2;
  int imode = 0;
  int shared_n = 0;

  // Helper lambda to flush current batch (C++11 required)
  auto flush_batch = [&]() {
    __syncthreads();
    int batch_start = imode - shared_n;
    if (tid < shared_n) {
      Real sum_r = 0.0f, sum_i = 0.0f;
      for (unsigned int t = 0; t < blockDim.x; t++) {
        sum_r += shared_real[t + SHMEM_STRIDE * tid];
        sum_i += shared_imag[t + SHMEM_STRIDE * tid];
      }
      int mode_idx = batch_start + tid;
      if (mode_idx < nmodes) {
        block_coeff_mode[bid * nmodes + mode_idx] = make_cuComplex(sum_r, sum_i);
      }
    }
    shared_n = 0;
  };

  for (int m_abs = 0; m_abs <= maxN; m_abs++) {
    Real cos_m = 1.0f, sin_m = 0.0f;
    if (rho <= 1.0f && m_abs > 0) {
      sincosf(m_abs*theta, &sin_m, &cos_m);
    }

    Real R_nm2 = 0.0f;
    Real R_nm4 = 0.0f;
    Real m2 = m_abs*m_abs;
    for (int n = m_abs; n <= maxN; n += 2) {
      Real R_curr;
      if(rho>1.0f) R_curr = 0;
      else{
        if (n == m_abs) {
          R_curr = rho_pow;
        } else if (n == m_abs + 2) {
          R_curr = R_nm2 * ((m_abs + 2) * rho*rho - (m_abs + 1));
        } else {
          Real K1 = n*0.5;
          Real num = n-2.f;
          Real K4 = -K1 * (num*num - m2);
          Real K2 = n * (n - 1.0f) * num;
          K1 = (n*n-m2) * (K1 - 1.0f);
          Real K3 = -m2 * (n - 1.0f) - K2;
          num = (2 * K2 * rho*rho + K3) * R_nm2 + K4 * R_nm4;
          R_curr = num / K1;
        }
        R_nm4 = R_nm2;
        R_nm2 = R_curr;
        R_curr *= m_abs == 0 ? sqrtf(n+1) : sqrtf(2*(n+1));
      }

      // Determine how many slots needed for this (n, m_abs)
      int needed = (m_abs == 0) ? 1 : 2;

      // Flush if not enough space
      if (shared_n + needed > modes_in_shared_mem) {
        flush_batch();
        __syncthreads();
      }

      // Now safe to write
      if (m_abs == 0) {
        shared_real[tid + SHMEM_STRIDE * shared_n] = phi_val.x * R_curr;
        shared_imag[tid + SHMEM_STRIDE * shared_n] = phi_val.y * R_curr;
        shared_n++;
        imode++;
      } else {
        Real Rc = R_curr*cos_m, Rs = R_curr*sin_m;
        shared_real[tid + SHMEM_STRIDE * shared_n] = phi_val.x * Rc;
        shared_imag[tid + SHMEM_STRIDE * shared_n] = phi_val.y * Rc;
        shared_n++;
        shared_real[tid + SHMEM_STRIDE * shared_n] = phi_val.x * Rs;
        shared_imag[tid + SHMEM_STRIDE * shared_n] = phi_val.y * Rs;
        shared_n++;
        imode += 2;
      }
    }
    rho_pow *= rho;
  }

  // Flush remaining modes
  if (shared_n > 0) {
    flush_batch();
  }
}

__global__ void reduce_coefficients(
    const cuComplex* block_coeff_mode,
    int nblocks,
    int nmodes,
    cuComplex* final_coeff,
    Real norm = 1
    ) {
  int mode = blockIdx.x * blockDim.x + threadIdx.x;
  if (mode >= nmodes) return;
  cuComplex sum = make_cuComplex(0.0f, 0.0f);
  for (int b = 0; b < nblocks; b++) {
    sum = cuCaddf(sum, block_coeff_mode[b * nmodes + mode]);
  }
  final_coeff[mode].x = sum.x*norm;
  final_coeff[mode].y = sum.y*norm;
}

__global__ void regularize_zernike_coefficients(
    int max_l,
    Real regl,
    Real regm,
    cuComplex* final_coeff
    ) {
  int mode = blockIdx.x * blockDim.x + threadIdx.x;
  if (mode >= (max_l+2)*(max_l+1)/2) return;
  int p,m = 0;
  if(mode <= (max_l>>1)){
    p = 2*mode;
  }else{
    if(max_l%2==1 && mode < (max_l+1)*3/2){
      m = 1;
      p = 1+((mode - (max_l>>1)-1)>>1<<1);
    }else{
      p = ceilf((1+sqrtf((3+max_l)*max_l-2*mode+3))/2)*2;
      int k = mode - ((3+max_l)*max_l + (2-p)*p)/2 - 1;
      m = 3 + (max_l - p);
      if(k >= p-2) {
        m++;
        k-=p-2;
      }
      p = m+(k>>1<<1);
    }
  }
  Real factor = 0;
  if(p>=10) factor = regl*p*p + regm*m*m;
  Real tmp = final_coeff[mode].x;
  final_coeff[mode].x = copysignf(fmaxf(fabs(tmp)-factor, 0),tmp);
  tmp = final_coeff[mode].y;
  final_coeff[mode].y = copysignf(fmaxf(fabs(tmp)-factor, 0),tmp);
}

__global__ void zernike_reconstruct_kernel(cuComplex* phi_out, int width, int height, Real cx, Real cy, Real radius, int maxN, const cuComplex* coeff, int nmodes) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int npixels = width * height;
  if (gid >= npixels) return;
  int x = gid % width, y = gid / width;
  Real dx = x - cx, dy = y - cy;
  Real fraction;
  pixel_circle_coverage_approx(dx, dy, radius, fraction, dx, dy);
  Real rho = sqrtf(dx*dx + dy*dy);
  if (fraction < 2e-20) {
    phi_out[gid] = make_cuComplex(0.0f, 0.0f);
    return;
  }
  rho = fminf(rho/radius,1);
  Real theta = (rho > 2e-20) ? atan2f(dy, dx) : 0.0f;
  Real sum_real = 0.0f, sum_imag = 0.0f;
  int mode_idx = 0;
  Real rho_pow = 1.0f;
  for (int m_abs = 0; m_abs <= maxN && mode_idx < nmodes; m_abs++) {
    Real cos_m = 1.0f, sin_m = 0.0f;
    if (m_abs > 0) {
      sincosf(m_abs*theta, &sin_m, &cos_m);
    }
    Real R_nm2 = 0.0f, R_nm4 = 0.0f;
    for (int n = m_abs; n <= maxN && mode_idx < nmodes; n += 2) {
      Real R_curr;
      if (n == m_abs) R_curr = rho_pow;
      else if (n == m_abs + 2) R_curr = R_nm2 * ((m_abs + 2)*rho*rho - (m_abs + 1));
      else {
        Real K1 = 0.5f*(n + m_abs)*(n - m_abs)*(n - 2.0f);
        Real K2 = 2.0f*n*(n - 1.0f)*(n - 2.0f);
        Real K3 = -m_abs*m_abs*(n - 1.0f) - n*(n - 1.0f)*(n - 2.0f);
        Real K4 = -0.5f*n*(n + m_abs - 2.0f)*(n - m_abs - 2.0f);
        R_curr = ((K2*rho*rho + K3)*R_nm2 + K4*R_nm4) / K1;
      }
      R_nm4 = R_nm2; R_nm2 = R_curr;
      R_curr *= (m_abs == 0) ? sqrtf(n + 1.0f) : sqrtf(2.0f*(n + 1.0f));
      if (m_abs == 0) {
        sum_real += coeff[mode_idx].x * R_curr;
        sum_imag += coeff[mode_idx].y * R_curr;
        mode_idx++;
      } else {
        Real cosR = R_curr * cos_m, sinR = R_curr * sin_m;
        sum_real += coeff[mode_idx].x * cosR + coeff[mode_idx+1].x * sinR;
        sum_imag += coeff[mode_idx].y * cosR + coeff[mode_idx+1].y * sinR;
        mode_idx += 2;
      }
    }
    rho_pow *= rho;
  }
  phi_out[gid] = make_cuComplex(sum_real, sum_imag);
}

complexFormat* zernike_compute(
    void* handle_ptr,
    complexFormat* phi,
    Real cx, Real cy, Real radius
    ) {
  if (!handle_ptr) return nullptr;
  ZernikeHandle* handle = static_cast<ZernikeHandle*>(handle_ptr);
  if (!handle->initialized) return nullptr;

  zernike_project_mode<<<handle->nblocks, handle->nthread, handle->shared_mem_size>>>(
      (cuComplex*)phi, handle->width, handle->height, cx, cy, radius, handle->maxN, handle->block_coeff
      );
  int reduce_threads = handle->nthread;
  int reduce_blocks = (handle->nmodes + reduce_threads - 1) / reduce_threads;
  reduce_coefficients<<<reduce_blocks, reduce_threads, 0>>>(
      handle->block_coeff, handle->nblocks, handle->nmodes, handle->final_coeff, 1./(M_PI*radius*radius)
      );
  //regularize_zernike_coefficients<<<reduce_blocks, reduce_threads, 0>>>(
  //    handle->nmodes, 1e-8, 1e-8, handle->final_coeff
  //    );

  return (complexFormat*)handle->final_coeff;  // return device pointer
}

void zernike_reconstruct(void* handle_ptr, complexFormat* phi_out, Real radius) {
  if (!handle_ptr || !phi_out) return;
  ZernikeHandle* handle = static_cast<ZernikeHandle*>(handle_ptr);
  if (!handle->initialized) return;

  int npixels = handle->width * handle->height;
  dim3 blockSize(512);
  dim3 gridSize((npixels + blockSize.x - 1) / blockSize.x);

  zernike_reconstruct_kernel<<<gridSize, blockSize>>>(
      (cuComplex*)phi_out,
      handle->width,
      handle->height,
      Real(handle->width-1)/2,   // cx
      Real(handle->height-1)/2.0f,  // cy
      radius,
      handle->maxN,
      handle->final_coeff,    // ← internal coefficient cache
      handle->nmodes
      );
}

__global__ void laguerre_project_mode(
    const cuComplex* phi,
    int width, int height, Real radius,
    Real cx, Real cy,
    int maxN, int maxM,
    cuComplex* block_coeff_mode
    ) {
  extern __shared__ char shared_raw[];
  const int modes_in_shared_mem = 14;  // 必须与 zernike_init 中的 shared_mem_size 一致
  const int SHMEM_STRIDE = blockDim.x + 1;
  Real* shared_real = (Real*)shared_raw;
  Real* shared_imag = shared_real + SHMEM_STRIDE * modes_in_shared_mem;

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;

  Real z = 0.0f, theta = 0.0f;
  cuComplex phi_val = make_cuComplex(0.0f, 0.0f);
  radius  = 1./radius;
  if (gid < width * height) {
    int x = gid % width;
    int y = gid / width;
    Real dx = (x - cx)*radius;
    Real dy = (y - cy)*radius;
    z = 2*(dx*dx + dy*dy);
    theta = atan2f(dy, dx);
    phi_val = phi[gid];
  }
  Real norm = radius*radius;

  int nmodes = (maxN+1) * (2 * maxM + 1);
  int imode = 0;
  int shared_n = 0;

  // Helper lambda to flush current batch (C++11 required)
  auto flush_batch = [&]() {
    __syncthreads();
    int batch_start = imode - shared_n;
    if (tid < shared_n) {
      Real sum_r = 0.0f, sum_i = 0.0f;
      for (unsigned int t = 0; t < blockDim.x; t++) {
        sum_r += shared_real[t + SHMEM_STRIDE * tid];
        sum_i += shared_imag[t + SHMEM_STRIDE * tid];
      }
      int mode_idx = batch_start + tid;
      if (mode_idx < nmodes) {
        block_coeff_mode[bid * nmodes + mode_idx] = make_cuComplex(sum_r*norm, sum_i*norm);
      }
    }
    shared_n = 0;
  };

  Real sqrt2rho = sqrt(z);
  Real init = 1.1283791670955126f*expf(-0.5f * z); //2/sqrt(pi) this is because we are using sin and cos as bases, so the sqrt(2) factor compared to the exp(i mtheta)
  for (int m_abs = 0; m_abs <= maxM; m_abs++) {
    Real cos_m = 1.0f, sin_m = 0.0f;
    if (m_abs > 0) {
      sincosf(m_abs*theta, &sin_m, &cos_m);
      init *= rsqrt(Real(m_abs));
    }
    Real f0 = init;
    if(m_abs == 0) f0 *= 0.7071067811865475; // no cos sin -> exp problem for m=0
    Real fact = rsqrtf((1 + m_abs));
    Real fact0 = m_abs-z;
    Real f1 = f0*(1.0f + fact0)*fact;
    fact0-=1;
    for (int n = 0; n <= maxN; n ++) {
      Real f = f0;
      if(n == 1){ f = f1;}
      else if(n > 1){
        Real coef2 = f0/fact;
        fact = rsqrtf(n * (n + m_abs));
        f = __fmaf_rn(fact0*fact, f1, -coef2*fact);
        f0 = f1;
        f1 = f;
      }
      fact0 += 2;
      // Determine how many slots needed for this (n, m_abs)
      int needed = (m_abs == 0) ? 1 : 2;
      if (shared_n + needed > modes_in_shared_mem) {
        flush_batch();
        __syncthreads();
      }
      if (m_abs == 0) {
        shared_real[tid + SHMEM_STRIDE * shared_n] = phi_val.x * f;
        shared_imag[tid + SHMEM_STRIDE * shared_n] = phi_val.y * f;
        shared_n++;
        imode++;
      } else {
        Real Rc = f*cos_m, Rs = f*sin_m;
        shared_real[tid + SHMEM_STRIDE * shared_n] = phi_val.x * Rc;
        shared_imag[tid + SHMEM_STRIDE * shared_n] = phi_val.y * Rc;
        shared_n++;
        shared_real[tid + SHMEM_STRIDE * shared_n] = phi_val.x * Rs;
        shared_imag[tid + SHMEM_STRIDE * shared_n] = phi_val.y * Rs;
        shared_n++;
        imode += 2;
      }
    }
    init *= sqrt2rho;
  }

  // Flush remaining modes
  if (shared_n > 0) {
    flush_batch();
  }
}
struct LaguerreHandle {
  int maxN;
  int maxM;
  int nmodes;
  int nblocks;
  int width, height;
  size_t shared_mem_size;
  cuComplex* block_coeff;
  cuComplex* final_coeff;
  complexFormat* host_coeff;
  int nthread;
  bool initialized;
};
void* laguerre_init(int width, int height, int maxN, int maxM, int max_blocks) {
  LaguerreHandle* handle = new LaguerreHandle();
  handle->width = width;
  handle->height = height;
  handle->maxN = maxN;
  handle->maxM = maxM;
  handle->nmodes = (maxN+1) * (2 * maxM + 1);
  handle->nthread = 128;
  int npixels = width * height;
  handle->nblocks = (npixels + handle->nthread - 1) / handle->nthread;
  if (max_blocks > 0 && handle->nblocks > max_blocks) handle->nblocks = max_blocks;
  handle->shared_mem_size = (handle->nthread + 1) * 14 * 2 * sizeof(Real);

  cudaError_t err1 = cudaMalloc(&handle->block_coeff, handle->nmodes * handle->nblocks * sizeof(cuComplex));
  cudaError_t err2 = cudaSuccess;
  if (err1 == cudaSuccess) {
    handle->host_coeff = (complexFormat*)malloc(handle->nmodes * sizeof(complexFormat));
    err2 = cudaMalloc(&handle->final_coeff, handle->nmodes * sizeof(cuComplex));
  }
  handle->initialized = (err1 == cudaSuccess && err2 == cudaSuccess);
  if (!handle->initialized) {
    if (err1 == cudaSuccess) cudaFree(handle->block_coeff);
    delete handle;
    return nullptr;
  }
  return static_cast<void*>(handle);
}
complexFormat* laguerre_compute(void* handle_ptr, complexFormat* phi, Real cx, Real cy, Real radius) {
  if (!handle_ptr) return nullptr;
  LaguerreHandle* handle = static_cast<LaguerreHandle*>(handle_ptr);
  if (!handle->initialized) return nullptr;

  laguerre_project_mode<<<handle->nblocks, handle->nthread, handle->shared_mem_size>>>(
      (cuComplex*)phi,
      handle->width, handle->height, radius,
      cx, cy,
      handle->maxN, handle->maxM,
      handle->block_coeff
      );

  int reduce_threads = handle->nthread;
  int reduce_blocks = (handle->nmodes + reduce_threads - 1) / reduce_threads;
  reduce_coefficients<<<reduce_blocks, reduce_threads, 0>>>(
      handle->block_coeff, handle->nblocks, handle->nmodes, handle->final_coeff
      );

  return (complexFormat*)handle->final_coeff; // device pointer
}
complexFormat* laguerre_coeff(void* handle_ptr) {
  if (!handle_ptr) return nullptr;
  LaguerreHandle* handle = static_cast<LaguerreHandle*>(handle_ptr);
  if (!handle->initialized) return nullptr;
  cudaMemcpy(handle->host_coeff,
      handle->final_coeff,
      handle->nmodes * sizeof(complexFormat),
      cudaMemcpyDeviceToHost);
  return handle->host_coeff;
}
void laguerre_destroy(void* handle_ptr) {
  if (!handle_ptr) return;
  LaguerreHandle* handle = static_cast<LaguerreHandle*>(handle_ptr);
  if (handle->initialized) {
    cudaFree(handle->block_coeff);
    cudaFree(handle->final_coeff);
    free(handle->host_coeff);
  }
  delete handle;
}
__global__ void laguerre_reconstruct_kernel(
    cuComplex* phi_out,
    int width, int height,
    Real radius,
    Real cx, Real cy,
    int maxN, int maxM,
    const cuComplex* coeff,
    int nmodes
    ) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int npixels = width * height;
  if (gid >= npixels) return;

  int x = gid % width;
  int y = gid / width;
  Real dx = (x - cx) / radius;
  Real dy = (y - cy) / radius;
  Real z = 2.0f * (dx*dx + dy*dy);
  Real theta = atan2f(dy, dx);

  // Early exit if outside support (optional, since Laguerre modes decay but are defined everywhere)
  // We keep all pixels for completeness.

  Real sum_real = 0.0f, sum_imag = 0.0f;
  int mode_idx = 0;

  Real sqrt2rho = sqrtf(z);
  Real init = 1.1283791670955126f * expf(-0.5f * z); // 2 / sqrt(pi)
  for (int m_abs = 0; m_abs <= maxM && mode_idx < nmodes; m_abs++) {
    Real cos_m = 1.0f, sin_m = 0.0f;
    if (m_abs > 0) {
      sincosf(m_abs*theta, &sin_m, &cos_m);
      init *= rsqrtf(Real(m_abs));
    }

    Real f0 = init;
    if (m_abs == 0) {
      f0 *= 0.7071067811865475f; // 1/sqrt(2) for m=0 (no sin/cos pair)
    }

    Real fact = rsqrtf(1.0f + m_abs);
    Real f1 = f0 * (1.0f + m_abs - z) * fact;

    for (int n = 0; n <= maxN && mode_idx < nmodes; n++) {
      Real f = f0;
      if (n == 1) {
        f = f1;
      } else if (n > 1) {
        Real coef2 = 1.0f / fact;
        fact = rsqrtf(Real(n) * (n + m_abs));
        coef2 *= fact;
        Real coef1 = (2.0f * n + m_abs - 1.0f - z) * fact;
        f = __fmaf_rn(coef1, f1, -coef2 * f0);
        f0 = f1;
        f1 = f;
      }

      if (m_abs == 0) {
        sum_real += coeff[mode_idx].x * f;
        sum_imag += coeff[mode_idx].y * f;
        mode_idx++;
      } else {
        Real cosR = f * cos_m;
        Real sinR = f * sin_m;
        sum_real += coeff[mode_idx].x * cosR + coeff[mode_idx + 1].x * sinR;
        sum_imag += coeff[mode_idx].y * cosR + coeff[mode_idx + 1].y * sinR;
        mode_idx += 2;
      }

      // Update f0/f1 for next n (if needed)
      if (n == 0) {
        f0 = f;
      } else if (n == 1) {
        // f1 already set above; f0 was previous f
        // No extra update needed beyond loop logic
      }
    }

    init *= sqrt2rho;
  }

  phi_out[gid] = make_cuComplex(sum_real, sum_imag);
}

void laguerre_reconstruct(
    void* handle_ptr,
    complexFormat* phi_out,
    Real radius
    ) {
  if (!handle_ptr || !phi_out) return;
  LaguerreHandle* handle = static_cast<LaguerreHandle*>(handle_ptr);
  if (!handle->initialized) return;

  int npixels = handle->width * handle->height;
  dim3 blockSize(512);
  dim3 gridSize((npixels + blockSize.x - 1) / blockSize.x);

  laguerre_reconstruct_kernel<<<gridSize, blockSize>>>(
      (cuComplex*)phi_out,
      handle->width,
      handle->height,
      radius,                    // ← passed as argument
      handle->width / 2.0f,      // cx (centered by default)
      handle->height / 2.0f,     // cy
      handle->maxN,
      handle->maxM,
      handle->final_coeff,
      handle->nmodes
      );
}
