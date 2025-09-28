#include "cudaConfig.hpp"
#include "cudaDefs_h.cu"
#include <complex.h>
#include <fmt/base.h>

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

__forceinline__ __device__ cuComplex multiplyAM(Real r, int m, Real theta) {
  Real c, s;
  sincosf(m * theta, &s, &c);
  return make_cuComplex(r * c, r * s);
}
__forceinline__ __device__ Real laguerre_gaussian_R(Real z, int p, int m) {
  int alpha = m>0? m:-m;
  Real L0 = sqrt(2./M_PI)*expf(-0.5f * z)*powf(z, Real(alpha)/2);
  for(int i = p+1; i <= p+alpha; i++) { L0 /= sqrtf(i); }
  if (p == 0) return L0;
  Real L1 = (1.0f + alpha - z)*L0;
  Real L = L1;
  for (int k = 2; k <= p; ++k) {
    L = __fmaf_rn(2.0f*k + alpha - 1.0f - z, L1, -(k - 1 + alpha) * L0) / k;
    L0 = L1;
    L1 = L;
  }
  return  L;
}
__forceinline__ __device__ cuComplex laguerre_gaussian(Real x, Real y, int p, int m) {
    return multiplyAM(laguerre_gaussian_R(2*(x*x + y*y), p, m), m, atan2(y, x));
}
__forceinline__ __device__ Real zernike_R(Real z, int n, int m_abs) {
    Real K = (n - m_abs) / 2;
    Real C = 1;
    Real b = min(K, n - K);
    for (int i = 0; i < b; ++i) {
      C = C * (n - i) / (i + 1);
    }
    Real radial = C;
    Real a_half = (n + m_abs) >> 1;
    for (int k = 0; k < K; ++k) {
      C = -(C * (a_half - k) * (K - k)) / ((n - k) * (k + 1));
      radial = __fmaf_rn(z, radial, C);
    }
    z = sqrt(z);
    for (int i = 0; i < m_abs; ++i) {
      radial *= z;
    }
    return radial;
}
__forceinline__ __device__ cuComplex zernike_complex(Real x, Real y, int n, int m) {
    Real z = x*x+y*y;
    if (z > 1.0f || n < 0) return cuComplex();
    const int m_abs = (m < 0) ? -m : m;
    if (m_abs > n || (n - m_abs) % 2 != 0)
        return cuComplex();
    return multiplyAM(zernike_R(z, n, m_abs) * sqrtf(n + 1), m, atan2(y, x));
}
cuFuncc(multiplyHermit,(complexFormat* store, complexFormat* data, Real pupilsize, int n, int m),(cuComplex* store, cuComplex* data, Real pupilsize, int n, int m),((cuComplex*)store, (cuComplex*)data,pupilsize, n, m),{
    cudaIdx()
    Real xp = Real(x - (cuda_row>>1))/pupilsize;
    Real yp = Real(y - (cuda_column>>1))/pupilsize;
    Real factor = Hermit(xp, n) * Hermit(yp, m);
    store[index].x = factor*data[index].x;
    store[index].y = factor*data[index].y;
    })

cuFuncc(multiplyZernikeConj,(complexFormat* store,complexFormat* data, Real pupilsize, int n, int m),(cuComplex* store, cuComplex* data, Real pupilsize, int n, int m),((cuComplex*)store, (cuComplex*)data,pupilsize, n, m),{
    cudaIdx()
    store[index] = cuCmulf(data[index], cuConjf(zernike_complex(Real(x - (cuda_row>>1))/pupilsize, Real(y - (cuda_column>>1))/pupilsize, n, m)));
    })

cuFuncc(multiplyLaguerreConj,(complexFormat* store,complexFormat* data, Real pupilsize, int n, int m),(cuComplex* store, cuComplex* data, Real pupilsize, int n, int m),((cuComplex*)store, (cuComplex*)data,pupilsize, n, m),{
    cudaIdx()
    store[index] = cuCmulf(data[index], cuConjf(laguerre_gaussian(Real(x - (cuda_row>>1))/pupilsize, Real(y - (cuda_column>>1))/pupilsize, n, m)));
    })

cuFuncc(addZernike,(complexFormat* store, complexFormat coefficient, Real pupilsize, int n, int m),(cuComplex* store, cuComplex coefficient, Real pupilsize, int n, int m),((cuComplex*)store, *(cuComplex*)&coefficient, pupilsize, n, m),{
    cudaIdx()
    Real xp = Real(x - (cuda_row>>1))/pupilsize;
    Real yp = Real(y - (cuda_column>>1))/pupilsize;
    store[index] = cuCaddf(store[index],cuCmulf(coefficient, zernike_complex(xp, yp, n, m)));
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
    bool initialized;
};

void* zernike_init(int width, int height, int maxN, int max_blocks) {
    ZernikeHandle* handle = new ZernikeHandle();
    handle->width = width;
    handle->height = height;
    handle->maxN = maxN;
    handle->nmodes = (maxN + 1) * (maxN + 2) / 2;
    int npixels = width * height;
    handle->nblocks = (npixels + 511) / 512;
    if (max_blocks > 0 && handle->nblocks > max_blocks) handle->nblocks = max_blocks;
    handle->shared_mem_size = (512 + 1) * 10 * 2 * sizeof(Real);
    
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
    Real cx, Real cy,
    int maxN,
    int nblocks,
    cuComplex* block_coeff_mode
) {
    extern __shared__ char shared_raw[];
    const int modes_in_shared_mem = 10;  // 必须与 zernike_init 中的 shared_mem_size 一致
    const int SHMEM_STRIDE = blockDim.x + 1;
    Real* shared_real = (Real*)shared_raw;
    Real* shared_imag = shared_real + SHMEM_STRIDE * modes_in_shared_mem;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = bid * blockDim.x + tid;

    Real rho = 0.0f, theta = 0.0f;
    cuComplex phi_val = make_cuComplex(0.0f, 0.0f);
    Real norm = fminf(width>>1, height>>1);
    if (gid < width * height) {
        int x = gid % width;
        int y = gid / width;
        Real dx = (x - cx) / norm;
        Real dy = (y - cy) / norm;
        Real rho2 = dx*dx + dy*dy;
        if (rho2 <= 1.0f) {
            rho = sqrtf(rho2);
            theta = atan2f(dy, dx);
            phi_val = phi[gid];
        }
    }
    norm = 1./(M_PI*norm*norm);

    int nmodes = (maxN + 1) * (maxN + 2) / 2;
    int imode = 0;
    int shared_n = 0;
    Real rho_pow = 1.0f;

    // Helper lambda to flush current batch (C++11 required)
    auto flush_batch = [&]() {
        if (shared_n == 0) return;
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
                block_coeff_mode[mode_idx * nblocks + bid] = make_cuComplex(sum_r*norm, sum_i*norm);
            }
        }
        shared_n = 0;
    };

    for (int m_abs = 0; m_abs <= maxN; m_abs++) {
        Real cos_m = 1.0f, sin_m = 0.0f;
        if (rho <= 1.0f && m_abs > 0) {
            Real m_theta = m_abs * theta;
            cos_m = cosf(m_theta);
            sin_m = sinf(m_theta);
        }

        Real R_nm2 = 0.0f;
        Real R_nm4 = 0.0f;
        for (int n = m_abs; n <= maxN; n += 2) {
            Real R_curr;
            if (n == m_abs) {
                R_curr = (rho <= 1.0f) ? rho_pow : 0.0f;
            } else if (n == m_abs + 2) {
              R_curr = R_nm2 * ((m_abs + 2) * rho * rho - (m_abs + 1));
            } else {
              Real K1 = ((n + m_abs) * (n - m_abs) * (n - 2.0f)) * 0.5f;
              Real K2 = 2.0f * n * (n - 1.0f) * (n - 2.0f);
              Real K3 = -m_abs * m_abs * (n - 1.0f) - n * (n - 1.0f) * (n - 2.0f);
              Real K4 = -0.5f * n * (n + m_abs - 2.0f) * (n - m_abs - 2.0f);
              Real numerator = (K2 * rho * rho + K3) * R_nm2 + K4 * R_nm4;
              R_curr = numerator / K1;
            }
            R_nm4 = R_nm2;
            R_nm2 = R_curr;

            // Determine how many slots needed for this (n, m_abs)
            int needed = (m_abs == 0) ? 1 : 2;

            // Flush if not enough space
            if (shared_n + needed > modes_in_shared_mem) {
              flush_batch();
              __syncthreads();
            }

            // Now safe to write
            R_curr *= m_abs == 0 ? sqrt(n+1) : sqrt(2*(n+1));
            if (m_abs == 0) {
              shared_real[tid + SHMEM_STRIDE * shared_n] = cuCrealf(phi_val) * R_curr;
              shared_imag[tid + SHMEM_STRIDE * shared_n] = cuCimagf(phi_val) * R_curr;
              shared_n++;
              imode++;
            } else {
              shared_real[tid + SHMEM_STRIDE * shared_n] = cuCrealf(phi_val) * R_curr * cos_m;
              shared_imag[tid + SHMEM_STRIDE * shared_n] = cuCimagf(phi_val) * R_curr * cos_m;
              shared_n++;
              shared_real[tid + SHMEM_STRIDE * shared_n] = cuCrealf(phi_val) * R_curr * sin_m;
              shared_imag[tid + SHMEM_STRIDE * shared_n] = cuCimagf(phi_val) * R_curr * sin_m;
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
    cuComplex* final_coeff
    ) {
  int mode = blockIdx.x * blockDim.x + threadIdx.x;
  if (mode >= nmodes) return;
  cuComplex sum = make_cuComplex(0.0f, 0.0f);
  for (int b = 0; b < nblocks; b++) {
    sum = cuCaddf(sum, block_coeff_mode[mode * nblocks + b]);
  }
  final_coeff[mode] = sum;
}

__global__ void zernike_reconstruct_kernel(cuComplex* phi_out, int width, int height, Real cx, Real cy, int maxN, const cuComplex* coeff, int nmodes) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int npixels = width * height;
    if (gid >= npixels) return;
    int x = gid % width, y = gid / width;
    Real norm_val = fminf(width >> 1, height >> 1);
    Real dx = (x - cx) / norm_val, dy = (y - cy) / norm_val;
    Real rho2 = dx*dx + dy*dy;
    if (rho2 > 1.0f) {
        phi_out[gid] = make_cuComplex(0.0f, 0.0f);
        return;
    }
    Real rho = sqrtf(rho2);
    Real theta = (rho2 > 0.0f) ? atan2f(dy, dx) : 0.0f;
    Real sum_real = 0.0f, sum_imag = 0.0f;
    int mode_idx = 0;
    Real rho_pow = 1.0f;
    for (int m_abs = 0; m_abs <= maxN && mode_idx < nmodes; m_abs++) {
        Real cos_m = 1.0f, sin_m = 0.0f;
        if (m_abs > 0) {
            Real m_theta = m_abs * theta;
            cos_m = cosf(m_theta); sin_m = sinf(m_theta);
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
    Real cx, Real cy
    ) {
  if (!handle_ptr) return nullptr;
  ZernikeHandle* handle = static_cast<ZernikeHandle*>(handle_ptr);
  if (!handle->initialized) return nullptr;

  zernike_project_mode<<<handle->nblocks, 512, handle->shared_mem_size>>>(
      (cuComplex*)phi, handle->width, handle->height, cx, cy, handle->maxN, handle->nblocks, handle->block_coeff
      );
  int reduce_threads = 512;
  int reduce_blocks = (handle->nmodes + reduce_threads - 1) / reduce_threads;
  reduce_coefficients<<<reduce_blocks, reduce_threads, 0>>>(
      handle->block_coeff, handle->nblocks, handle->nmodes, handle->final_coeff
      );

  return (complexFormat*)handle->final_coeff;  // return device pointer
}

void zernike_reconstruct(void* handle_ptr, complexFormat* phi_out) {
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
        handle->width / 2.0f,   // cx
        handle->height / 2.0f,  // cy
        handle->maxN,
        handle->final_coeff,    // ← internal coefficient cache
        handle->nmodes
    );
}
