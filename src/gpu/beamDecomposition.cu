#include "cudaConfig.hpp"
#include <cuComplex.h>
#include "cudaDefs_h.cu"
#include "memManager.hpp"
#include <cmath>
#include <complex.h>
#include <fmt/core.h>

typedef struct { float x, y; } Point;
typedef struct { float area, cx, cy; } PixelRes;
__forceinline__ __device__ PixelRes compute_pixel(float px, float py, float r) {
  float r2 = r*r;
  float d2 = px*px + py*py;
  float sqrt2 = sqrtf(2);
  if(d2 > (r+sqrt2)*(r+sqrt2)) return (PixelRes){0.0f, px, py};
  if(d2 < (r-sqrt2)*(r-sqrt2)) return (PixelRes){1.0f, px, py};
  float x0=px-0.5f, x1=px+0.5f, y0=py-0.5f, y1=py+0.5f;
  int c[4] = {(x0*x0+y0*y0<r2), (x1*x1+y0*y0<r2), (x1*x1+y1*y1<r2), (x0*x0+y1*y1<r2)};
  int n_in = c[0]+c[1]+c[2]+c[3];
  if(n_in==4) return (PixelRes){1.0f, px, py};
  if(n_in==0) return (PixelRes){0.0f, px, py};
  Point ints[8]; int n_int=0;
  float eps = 1e-5f;
  for(int i=0; i<2; i++) {
    float x = (i==0) ? x0 : x1;
    float val = (r - x) * (r + x);
    if(val >= -eps) {
      val = (val < 0) ? 0 : val;
      float y = sqrtf(val);
      if(y >= y0-eps && y <= y1+eps) {
        bool dup = false;
        for(int k=0; k<n_int; k++) if(fabsf(ints[k].x-x)<eps && fabsf(ints[k].y-y)<eps) dup=true;
        if(!dup) { ints[n_int].x=x; ints[n_int].y=y; n_int++; }
      }
      if(n_int < 8 && y > eps && -y >= y0-eps && -y <= y1+eps) {
        bool dup = false;
        for(int k=0; k<n_int; k++) if(fabsf(ints[k].x-x)<eps && fabsf(ints[k].y+y)<eps) dup=true;
        if(!dup) { ints[n_int].x=x; ints[n_int].y=-y; n_int++; }
      }
    }
  }
  for(int i=0; i<2; i++) {
    float y = (i==0) ? y0 : y1;
    float val = (r - y) * (r + y);
    if(val >= -eps) {
      val = (val < 0) ? 0 : val;
      float x = sqrtf(val);
      if(x >= x0-eps && x <= x1+eps) {
        bool dup = false;
        for(int k=0; k<n_int; k++) if(fabsf(ints[k].x-x)<eps && fabsf(ints[k].y-y)<eps) dup=true;
        if(!dup) { ints[n_int].x=x; ints[n_int].y=y; n_int++; }
      }
      if(n_int < 8 && x > eps && -x >= x0-eps && -x <= x1+eps) {
        bool dup = false;
        for(int k=0; k<n_int; k++) if(fabsf(ints[k].x+x)<eps && fabsf(ints[k].y-y)<eps) dup=true;
        if(!dup) { ints[n_int].x=-x; ints[n_int].y=y; n_int++; }
      }
    }
  }
  Point verts[12]; int nc=0;
  float vx[4]={x0,x1,x1,x0}, vy[4]={y0,y0,y1,y1};
  for(int i=0; i<4; i++) if(c[i]) { verts[nc].x=vx[i]; verts[nc].y=vy[i]; nc++; }
  for(int i=0; i<n_int; i++) { verts[nc].x=ints[i].x; verts[nc].y=ints[i].y; nc++; }
  if(nc < 3) return (PixelRes){(n_in>0?1.0f:0.0f), px, py};
  int idx[12]; for(int i=0; i<nc; i++) idx[i]=i;
  for(int i=0; i<nc-1; i++) for(int j=i+1; j<nc; j++) {
    float a1=atan2f(verts[idx[i]].y-py, verts[idx[i]].x-px);
    float a2=atan2f(verts[idx[j]].y-py, verts[idx[j]].x-px);
    if(a1>a2) { int t=idx[i]; idx[i]=idx[j]; idx[j]=t; }
  }
  float A=0.0f, Mx=0.0f, My=0.0f;
  for(int i=0; i<nc; i++) {
    Point p1=verts[idx[i]], p2=verts[idx[(i+1)%nc]];
    double cross = double(p1.x) * p2.y - double(p2.x) * p1.y;
    A += cross; Mx += (p1.x + p2.x) * cross; My += (p1.y + p2.y) * cross;
  }
  A *= 0.5f; Mx /= 6.0f; My /= 6.0f;
  if(A < 0.0f) { A=-A; Mx=-Mx; My=-My; }
  if(n_int == 2) {
    float mx = (ints[0].x+ints[1].x)*0.5f, my = (ints[0].y+ints[1].y)*0.5f;
    float dx_c = ints[0].x-ints[1].x, dy_c = ints[0].y-ints[1].y;
    float L = sqrtf(dx_c*dx_c + dy_c*dy_c);
    float h = sqrtf(mx*mx + my*my);
    float theta = 2.0f * atan2f(0.5f*L, h);
    float theta_minus_sin, sin_half_cubed;
    if(theta < 1e-4f) {
      float t2 = theta*theta, t3 = t2*theta, t5 = t3*t2;
      theta_minus_sin = t3/6.0f - t5/120.0f;
      float sh = 0.5f*theta; sin_half_cubed = sh*sh*sh;
    } else {
      theta_minus_sin = theta - sinf(theta);
      sin_half_cubed = powf(sinf(0.5f*theta), 3.0f);
    }
    float A_s = 0.5f * r2 * theta_minus_sin;
    float d_s = (theta_minus_sin > 1e-10f) ? (4.0f * r * sin_half_cubed) / (3.0f * theta_minus_sin) : r;
    float norm = h > 1e-9f ? h : 1.0f;
    float ux = mx/norm, uy = my/norm;
    float Mx_s = A_s * d_s * ux, My_s = A_s * d_s * uy;
    A += A_s; Mx += Mx_s; My += My_s;
  }
  float Cx = (A > 1e-9f) ? Mx/A : px;
  float Cy = (A > 1e-9f) ? My/A : py;
  return (PixelRes){A, Cx, Cy};
}

__forceinline__ __device__ void pixel_circle_coverage_approx(
    Real dx, Real dy,  // rectangle center relative to circle (which is at origin)
    Real r,
    Real& area,        // output: covered area of pixel
    Real& cent_x,      // centroid x, relative to circle (origin)
    Real& cent_y       // centroid y, relative to circle (origin)
) {
    PixelRes res = compute_pixel(dx, dy, r);
    area = res.area;
    cent_x = res.cx;
    cent_y = res.cy;
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

__forceinline__ __device__ cuComplex make_polar(Real r, Real theta) {
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
  return make_polar(laguerre_gaussian_R(2*(x*x + y*y), p, m), m*atan2f(y, x));
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
  return make_polar(zernike_R(z, n, m_abs) * sqrtf(n + 1), m*atan2f(y, x));
}
__forceinline__ __device__ cuComplex zernike_complex_polar(Real r, Real theta, int n, int m) {
  Real z = r*r;
  const int m_abs = (m < 0) ? -m : m;
  if (m_abs > n || (n - m_abs) % 2 != 0)
    return cuComplex();
  return make_polar(zernike_R(z, n, m_abs) * sqrtf(n + 1), m*theta);
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
  int width;
  size_t shared_mem_size;
  cuComplex* block_coeff;
  cuComplex* final_coeff;
  complexFormat* host_coeff;
  int nthread;
  bool initialized;
};


void* zernike_init(int width, int maxN, int max_blocks) {
  ZernikeHandle* handle = new ZernikeHandle();
  handle->width = width;
  handle->maxN = maxN;
  handle->nmodes = (maxN + 1) * (maxN + 2) / 2;
  handle->nthread = 128;
  int npixels = width * width;
  handle->nblocks = (npixels + handle->nthread-1) / handle->nthread;
  if (max_blocks > 0 && handle->nblocks > max_blocks) handle->nblocks = max_blocks;
  handle->shared_mem_size = (handle->nthread + 1) * 14 * 2 * sizeof(Real);

  myCuMalloc(cuComplex, handle->block_coeff, handle->nmodes * handle->nblocks);
  myMalloc(complexFormat, handle->host_coeff, handle->nmodes);
  myCuMalloc(cuComplex, handle->final_coeff, handle->nmodes);
  handle->initialized = true;
  return static_cast<void*>(handle);
}

void* zernike_init_shared_mem(void* handle_shared, int width, int maxN, int max_blocks) {
  ZernikeHandle* handle = new ZernikeHandle();
  handle->width = width;
  handle->maxN = maxN;
  handle->nmodes = (maxN + 1) * (maxN + 2) / 2;
  handle->nthread = 128;
  int npixels = width * width;
  handle->nblocks = (npixels + handle->nthread-1) / handle->nthread;
  if (max_blocks > 0 && handle->nblocks > max_blocks) handle->nblocks = max_blocks;
  handle->shared_mem_size = (handle->nthread + 1) * 14 * 2 * sizeof(Real);

  handle->block_coeff = ((ZernikeHandle*)handle_shared)->block_coeff;
  handle->host_coeff = ((ZernikeHandle*)handle_shared)->host_coeff;
  handle->final_coeff = ((ZernikeHandle*)handle_shared)->final_coeff;
  handle->initialized = true;
  return static_cast<void*>(handle);
}

void** zernike_init_group(int* widths, int maxN, int n, int max_blocks) {
  myDMalloc(void*, handles, n);
  int max_width = 0, max_idx = 0;
  for (int i = 0 ; i < n ; i++) {
    if(widths[i] > max_width) {
      max_width = widths[i];
      max_idx = 0;
    }
  }
  handles[max_idx] = zernike_init(max_width, maxN, max_blocks);
  for (int i = 0 ; i < n ; i++) {
    if(i == max_idx) {
      continue;
    }
    handles[i] = zernike_init_shared_mem(handles[max_idx], widths[i], maxN, max_blocks);
  }
  return handles;
}

complexFormat* zernike_coeff(void* handle_ptr) {
  if (!handle_ptr) return nullptr;
  ZernikeHandle* handle = static_cast<ZernikeHandle*>(handle_ptr);
  if (!handle->initialized) return nullptr;
  myMemcpyD2H(handle->host_coeff, 
      handle->final_coeff, 
      handle->nmodes * sizeof(complexFormat)); 
  return handle->host_coeff;
}

void zernike_destroy(void* handle_ptr) {
  if (!handle_ptr) return;
  ZernikeHandle* handle = static_cast<ZernikeHandle*>(handle_ptr);
  if (handle->initialized) {
    myCuFree(handle->block_coeff);
    myCuFree(handle->final_coeff);
    free(handle->host_coeff);
  }
  delete handle;
}
__global__ void zernike_project_mode(
    const cuComplex* phi,
    int width,
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
  if (gid < width * width) {
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
    double norm = 1
    ) {
  int mode = blockIdx.x * blockDim.x + threadIdx.x;
  if (mode >= nmodes) return;
  double sumx = 0, sumy = 0;
  for (int b = 0; b < nblocks; b++) {
    sumx += block_coeff_mode[b * nmodes + mode].x;
    sumy += block_coeff_mode[b * nmodes + mode].y;
  }
  final_coeff[mode].x = sumx*norm;
  final_coeff[mode].y = sumy*norm;
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
  factor = regl*p*p + regm*m*m;
  Real tmp = final_coeff[mode].x;
  final_coeff[mode].x = copysignf(fmaxf(fabs(tmp)-factor, 0),tmp);
  tmp = final_coeff[mode].y;
  final_coeff[mode].y = copysignf(fmaxf(fabs(tmp)-factor, 0),tmp);
}

__global__ void zernike_reconstruct_kernel(cuComplex* phi_out, int width, Real cx, Real cy, Real radius, int maxN, const cuComplex* coeff, int nmodes) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int npixels = width * width;
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
      (cuComplex*)phi, handle->width, cx, cy, radius, handle->maxN, handle->block_coeff
      );
  int reduce_threads = handle->nthread;
  int reduce_blocks = (handle->nmodes + reduce_threads - 1) / reduce_threads;
  reduce_coefficients<<<reduce_blocks, reduce_threads, 0>>>(
      handle->block_coeff, handle->nblocks, handle->nmodes, handle->final_coeff, 1./(M_PI*radius*radius)
      );
  regularize_zernike_coefficients<<<reduce_blocks, reduce_threads, 0>>>(
      handle->nmodes, 5e-5, 5e-8, handle->final_coeff
      //handle->nmodes, 1e-7, 0, handle->final_coeff
      );

  return (complexFormat*)handle->final_coeff;  // return device pointer
}

void zernike_reconstruct(void* handle_ptr, complexFormat* phi_out, Real radius) {
  if (!handle_ptr || !phi_out) return;
  ZernikeHandle* handle = static_cast<ZernikeHandle*>(handle_ptr);
  if (!handle->initialized) return;

  int npixels = handle->width * handle->width;
  dim3 blockSize(512);
  dim3 gridSize((npixels + blockSize.x - 1) / blockSize.x);

  zernike_reconstruct_kernel<<<gridSize, blockSize>>>(
      (cuComplex*)phi_out,
      handle->width,
      Real(handle->width-1)/2,   // cx
      Real(handle->width-1)/2,   // cy
      radius,
      handle->maxN,
      handle->final_coeff,    // ← internal coefficient cache
      handle->nmodes
      );
}

__global__ void laguerre_project_mode(
    const cuComplex* phi,
    int width, Real radius,
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
  if (gid < width * width) {
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
  int width;
  size_t shared_mem_size;
  cuComplex* block_coeff;
  cuComplex* final_coeff;
  complexFormat* host_coeff;
  int nthread;
  bool initialized;
};
void* laguerre_init(int width, int maxN, int maxM, int max_blocks) {
  LaguerreHandle* handle = new LaguerreHandle();
  handle->width = width;
  handle->maxN = maxN;
  handle->maxM = maxM;
  handle->nmodes = (maxN+1) * (2 * maxM + 1);
  handle->nthread = 128;
  int npixels = width * width;
  handle->nblocks = (npixels + handle->nthread - 1) / handle->nthread;
  if (max_blocks > 0 && handle->nblocks > max_blocks) handle->nblocks = max_blocks;
  handle->shared_mem_size = (handle->nthread + 1) * 14 * 2 * sizeof(Real);

  myCuMalloc(cuComplex, handle->block_coeff, handle->nmodes * handle->nblocks)
  myMalloc(complexFormat, handle->host_coeff, handle->nmodes);
  myCuMalloc(cuComplex, handle->final_coeff, handle->nmodes);
  handle->initialized = true;
  return static_cast<void*>(handle);
}
complexFormat* laguerre_compute(void* handle_ptr, complexFormat* phi, Real cx, Real cy, Real radius) {
  if (!handle_ptr) return nullptr;
  LaguerreHandle* handle = static_cast<LaguerreHandle*>(handle_ptr);
  if (!handle->initialized) return nullptr;

  laguerre_project_mode<<<handle->nblocks, handle->nthread, handle->shared_mem_size>>>(
      (cuComplex*)phi,
      handle->width, radius,
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
  myMemcpyD2H(handle->host_coeff,
      handle->final_coeff,
      handle->nmodes * sizeof(complexFormat));
  return handle->host_coeff;
}
void laguerre_destroy(void* handle_ptr) {
  if (!handle_ptr) return;
  LaguerreHandle* handle = static_cast<LaguerreHandle*>(handle_ptr);
  if (handle->initialized) {
    myCuFree(handle->block_coeff);
    myCuFree(handle->final_coeff);
    myFree(handle->host_coeff);
  }
  delete handle;
}
__global__ void laguerre_reconstruct_kernel(
    cuComplex* phi_out,
    int width,
    Real radius,
    Real cx, Real cy,
    int maxN, int maxM,
    const cuComplex* coeff,
    int nmodes
    ) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int npixels = width * width;
  if (gid >= npixels) return;

  int x = gid % width;
  int y = gid / width;
  Real dx = (x - cx) / radius;
  Real dy = (y - cy) / radius;
  Real z = 2.0f * (dx*dx + dy*dy);
  Real theta = atan2f(dy, dx);
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
      if (n == 0) {
        f0 = f;
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

  int npixels = handle->width * handle->width;
  dim3 blockSize(512);
  dim3 gridSize((npixels + blockSize.x - 1) / blockSize.x);

  laguerre_reconstruct_kernel<<<gridSize, blockSize>>>(
      (cuComplex*)phi_out,
      handle->width,
      radius,                    // ← passed as argument
      handle->width / 2.0f,      // cx (centered by default)
      handle->width / 2.0f,      // cy (centered by default)
      handle->maxN,
      handle->maxM,
      handle->final_coeff,
      handle->nmodes
      );
}
