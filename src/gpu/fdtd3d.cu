#include <cstring>
#include <math.h>
#include "cudaDefs_h.cu"
#include "fdtd3d.hpp"
__device__ Real getmH(int x, int y, int z){
  Real ret = 0.4;
  return ret;
}
__device__ Real getmE(int x, int y, int z){
  //if(sqSum(x-nx/2,y-ny/2,z-nz/2) < 100) mE = 0.1;
  //if(sqSum(sqrt(sqSum(x-100,y-100))-70,z-100)<100) return 2./9;
  //if(sqSum(y-100,z-100)<100) return 2./9;
  //if(abs(y-100)<90) return 0.2;
  //if(y>120 && y < 140) return 2./9;
  Real ret = 0.5;
  return ret;
  //return 0.2;
}
cuFunc(updateH,(Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez), (Hx, Hy, Hz, Ex, Ey, Ez), {
  cuda3Idx();
#ifdef secondOrder
  Real ismid = x>=3&&x<cuda_row-3&&y>=3&&y<cuda_column-3&&z>=3&&z<cuda_height-3;
#endif
  Real mH = getmH(x,y,z);
  if(z < cuda_height-1 && x > 0 && y < cuda_column-1){
    Real dH = Ez[index+cuda_row]-Ez[index]-Ey[index+cuda_row*cuda_column]+Ey[index];
#ifdef secondOrder
    if(ismid)
      dH = fact1*(Ez[index-cuda_row]-Ez[index+2*cuda_row]+Ey[index+2*cuda_row*cuda_column]-Ey[index-cuda_row*cuda_column]) + fact2*dH;
#endif
    Hx[index] -= mH*dH;
  }
  if(z < cuda_height-1 && x < cuda_row-1 && y > 0){
    Real dH = Ex[index+cuda_row*cuda_column]-Ex[index]-Ez[index+1]+Ez[index];
#ifdef secondOrder
    if(ismid)
      dH = fact1*(Ex[index-cuda_row*cuda_column]-Ex[index+2*cuda_row*cuda_column]+Ez[index+2]-Ez[index-1]) + fact2*dH;
#endif
    Hy[index] -= mH*dH; //dEz/dy
  }
  if(z > 0 && x < cuda_row-1 && y < cuda_column-1){
    Real dH = Ey[index+1]-Ey[index]-Ex[index+cuda_row]+Ex[index];
#ifdef secondOrder
    if(ismid)
      dH = fact1*(Ey[index-1]-Ey[index+2]+Ex[index+2*cuda_row]-Ex[index-cuda_row]) + fact2*dH;
#endif
    Hz[index] -= mH*dH; //dEz/dx
  }
})
cuFunc(updateE,(Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez), (Hx, Hy, Hz, Ex, Ey, Ez),{
  cuda3Idx();
  Real mE = getmE(x,y,z);
#ifdef secondOrder
  Real ismid = x>=3&&x<cuda_row-3&&y>=3&&y<cuda_column-3&&z>=3&&z<cuda_height-3;
#endif
  if(z > 0 && x < cuda_row-1 && y > 0){
    Real dE = Hz[index]-Hz[index-cuda_row]-Hy[index]+Hy[index-cuda_row*cuda_column];
#ifdef secondOrder
    if(ismid)
      dE = fact1*(Hz[index-2*cuda_row]-Hz[index+cuda_row]+Hy[index+cuda_row*cuda_column]-Hy[index-2*cuda_row*cuda_column]) + fact2*dE;
#endif
    Ex[index] += mE*dE; //dEz/dy
  }
  if(z > 0 && x > 0 && y > 0){
    Real dE = Hx[index]-Hx[index-cuda_row*cuda_column]-Hz[index]+Hz[index-1];
#ifdef secondOrder
    if(ismid)
      dE = fact1*(Hx[index-2*cuda_row*cuda_column]-Hx[index+cuda_row*cuda_column]+Hz[index+1]-Hz[index-2]) + fact2*dE;
#endif
    Ey[index] += mE*dE;
  }
  if(z < cuda_height-1 && x > 0 && y > 0){
    Real dE = Hy[index]-Hy[index-1]-Hx[index]+Hx[index-cuda_row];
#ifdef secondOrder
    if(ismid)
      dE = fact1*(Hy[index-2]-Hy[index+1]+Hx[index+cuda_row]-Hx[index-2*cuda_row]) + fact2*dE;
#endif
    Ez[index] += mE*dE;
  }
})
cuFunc(applyPMLx0_d, (Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez), (Hx, Hy, Hz, Ex, Ey, Ez), {
  cuda3Idx();
  index = x + 1 + cuda_row*y + cuda_row*cuda_column*z;
  Real sf = b_PML+x*k_PML;
  Ez[index] *= sf;
  Ex[index] *= sf;
  Ey[index] *= sf;
  Hy[index] *= sf;
  Hx[index] *= sf;
  Hz[index] *= sf;
})
cuFunc(applyPMLx1_d, (Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez, int nx), (Hx, Hy, Hz, Ex, Ey, Ez, nx), {
  cuda3Idx();
  index = nx-x-2 + nx*y + cuda_row*cuda_column*z;
  Real sf = b_PML+x*k_PML;
  Ez[index] *= sf;
  Ex[index] *= sf;
  Ey[index] *= sf;
  Hy[index] *= sf;
  Hx[index] *= sf;
  Hz[index] *= sf;
})
cuFunc(applyPMLx1, (Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez, Real* EyBdx1, Real* EzBdx1, int nx),(Hx, Hy, Hz, Ex, Ey, Ez, EyBdx1, EzBdx1, nx),
{
  cudaIdx()
  if(y < cuda_column*2/3) return;
  index = x*nx*cuda_row + y*nx + nx - 50;  //large stride, unavoidable
  Real mH = getmH(nx-1,y,x);
  Real mE = getmE(nx-1,y,x);
  Real rat = sqrt(mH/mE);
  Real a = Ez[index];
  Real dt = 0.5/(mE*rat)-0.5;
  Hy[index-1] += rat*(dt*EzBdx1[y+x*cuda_row] + (1-dt)*a);
  EzBdx1[y+x*cuda_row] = a;
  Ez[index] = 0;
  a = Ey[index];
  Hz[index-1] -= rat*(dt*EyBdx1[y+x*cuda_row] + (1-dt)*a);
  EyBdx1[y+x*cuda_row] = a;
  Ey[index] = 0;
  Ex[index-1] = 0;
})
cuFunc( applyPMLx0, (Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez, Real* HyBdx0, Real* HzBdx0, int nx), (Hx, Hy, Hz, Ex, Ey, Ez, HyBdx0, HzBdx0, nx), {
  cudaIdx();
  index = x*nx*cuda_row + y*nx;  //large stride, unavoidable
  Real a = Hz[index];
  Real mH = getmH(0,y,x);
  Real mE = getmE(0,y,x);
  Real rat = sqrt(mE/mH);
  Real dt = 0.5/(rat*mH)-0.5;
  Ey[index+1] += rat*(HzBdx0[y+x*cuda_row]*dt + (1-dt)*a);
  HzBdx0[y+x*cuda_row] = a;
  Hz[index] = 0;
  a = Hy[index];
  Ez[index+1] -= rat*(HyBdx0[y+x*cuda_row]*dt + (1-dt)*a);
  HyBdx0[y+x*cuda_row] = a;
  Hy[index] = 0;
  Hx[index+1] = 0;
})
cuFunc( applyPMLy1, (Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez, Real* ExBdy1, Real* EzBdy1, int ny), (Hx, Hy, Hz, Ex, Ey, Ez, ExBdy1, EzBdy1, ny), {
  cudaIdx()
  Real mH = getmH(y,ny-1,x);
  Real mE = getmE(y,ny-1,x);
  Real rat = sqrt(mH/mE);
  Real dt = 0.5/(mE*rat)-0.5;
  index = cuda_row*ny*x + cuda_row*(ny-1)+ y;
  Real a = Ez[index];
  Hx[index-cuda_row] -= rat*(dt*EzBdy1[y+cuda_row*x] + (1-dt)*a);
  EzBdy1[y+cuda_row*x] = a;
  Ez[index] = 0;
  a = Ex[index];
  Hz[index-cuda_row] += rat*(dt*ExBdy1[y+cuda_row*x] + (1-dt)*a);
  ExBdy1[y+cuda_row*x] = a;
  Ex[index] = 0;
  Ey[index-cuda_row] = 0;
  Real sf = b_PML;
  for(int y = 0; y < n_PML; y++) {
    Ez[index] *= sf;
    Ex[index] *= sf;
    Ey[index] *= sf;
    Hy[index] *= sf;
    Hx[index] *= sf;
    Hz[index] *= sf;
    sf+=k_PML;
    index-=cuda_row;
  }
})
cuFunc(applyPMLy0, (Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez, Real* HxBdy0, Real* HzBdy0, int ny), (Hx, Hy, Hz, Ex, Ey, Ez, HxBdy0, HzBdy0, ny), {
  cuda1Idx();
  int x = index%cuda_row;
  int z = index/cuda_row;
  index = cuda_row*ny*z + x;
  Real mH = getmH(x,0,z);
  Real mE = getmE(x,0,z);
  Real rat = sqrt(mE/mH);
  Real dt = 0.5/(rat*mH)-0.5;
  Real a = Hz[index];
  Ex[index+cuda_row] -= rat*(dt*HzBdy0[x+cuda_row*z] + (1-dt)*a);
  HzBdy0[x+cuda_row*z] = a;
  Hz[index] = 0;
  a = Hx[index];
  Ez[index+cuda_row] += rat*(dt*HxBdy0[x+cuda_row*z] + (1-dt)*a);
  HxBdy0[x+cuda_row*z] = a;
  Hx[index] = 0;
  Hy[index+cuda_row] = 0;
  Real sf = b_PML;
  for(int y = 0; y < n_PML; y++) {
    Ez[index] *= sf;
    Ey[index] *= sf;
    Ex[index] *= sf;
    Hy[index] *= sf;
    Hx[index] *= sf;
    Hz[index] *= sf;
    index+=cuda_row;
    sf += k_PML;
  }
})
cuFunc( applyPMLz1, (Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez, Real* ExBdz1, Real* EyBdz1, int nz), (Hx, Hy, Hz, Ex, Ey, Ez, ExBdz1, EyBdz1, nz), {
  cudaIdx();
  int step = cuda_row*cuda_column;
  index = (nz-1)*step + cuda_row*x + y;
  Real mH = getmH(y,x,nz-1);
  Real mE = getmE(y,x,nz-1);
  Real rat = sqrt(mH/mE);
  Real dt = 0.5/(mE*rat)-0.5;
  int idxp1 = index-step;
  Real a = Ex[index];
  Hy[idxp1] -= rat*(dt*ExBdz1[y+cuda_row*x] + (1-dt)*a);
  ExBdz1[y+cuda_row*x] = a;
  Ex[index] = 0;
  a = Ey[index];
  Hx[idxp1] += rat*(dt*EyBdz1[y+cuda_row*x] + (1-dt)*a);
  EyBdz1[y+cuda_row*x] = a;
  Ey[index] = 0;
  Ez[idxp1] = 0;
  Real sf = b_PML;
  for(int z = 0; z < n_PML; z++) {
    Ez[index] *= sf;
    Ey[index] *= sf;
    Ex[index] *= sf;
    Hy[index] *= sf;
    Hx[index] *= sf;
    Hz[index] *= sf;
    index-=step;
    sf+=k_PML;
  }
})
cuFunc(applyPMLz0,(Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez, Real* HxBdz0, Real* HyBdz0, int nz), (Hx, Hy, Hz, Ex, Ey, Ez, HxBdz0, HyBdz0, nz),{
    cudaIdx();
  int step = cuda_row*cuda_column;
  index = cuda_row*y + x;
  Real mH = getmH(x,y,0);
  Real mE = getmE(x,y,0);
  Real rat = sqrt(mE/mH);
  Real dt = 0.5/(rat*mH)-0.5;
  int idxp1 = index+step;
  Real a = Hx[index];
  Ey[idxp1] -= rat*(dt*HxBdz0[x+cuda_row*y] + (1-dt)*a);
  HxBdz0[x+cuda_row*y] = a;
  Hx[index] = 0;
  a = Hy[index];
  Ex[idxp1] += rat*(dt*HyBdz0[x+cuda_row*y] + (1-dt)*a);
  HyBdz0[x+cuda_row*y] = a;
  Hy[index] = 0;
  Hz[idxp1] = 0;
  Real sf = b_PML;
  for(int z = 0; z < n_PML; z++) {
    Ez[index] *= sf;
    Ey[index] *= sf;
    Ex[index] *= sf;
    Hy[index] *= sf;
    Hx[index] *= sf;
    Hz[index] *= sf;
    index+=step;
    sf+=k_PML;
  }
})
__global__ void applyPeriodicx_H(Real* Hx, Real* Hy, Real* Hz, int nx, int ny, int nz){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int z = blockIdx.y * blockDim.y + threadIdx.y;
  if(y >= ny || z >= nz) return;
  int index = z*nx*ny + y*nx;  //large stride, unavoidable
  int index1 = index + nx - 1;
  Hx[index] = Hx[index1];
  Hy[index1] = Hy[index];
  Hz[index1] = Hz[index];
}
__global__ void applyPeriodicx_E(Real* Ex, Real* Ey, Real* Ez, int nx, int ny, int nz){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int z = blockIdx.y * blockDim.y + threadIdx.y;
  if(y >= ny || z >= nz) return;
  int index = z*nx*ny + y*nx;  //large stride, unavoidable
  int index1 = index + nx - 1;
  Ez[index] = Ez[index1];
  Ey[index] = Ey[index1];
  Ex[index1] = Ex[index];
}
__global__ void applyPeriodicy_H(Real* Hx, Real* Hy, Real* Hz, int nx, int ny, int nz){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int z = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= nx || z >= nz) return;
  int index = z*nx*ny + x;  //large stride, unavoidable
  int index1 = index + (ny - 1)*nx;
  Hx[index1] = Hx[index];
  Hy[index] = Hy[index1];
  Hz[index1] = Hz[index];
}
__global__ void applyPeriodicy_E(Real* Ex, Real* Ey, Real* Ez, int nx, int ny, int nz){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int z = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= nx || z >= nz) return;
  int index = z*nx*ny + x;  //large stride, unavoidable
  int index1 = index + (ny - 1)*nx;
  Ez[index] = Ez[index1];
  Ex[index] = Ex[index1];
  Ey[index1] = Ey[index];
}
__global__ void applyPeriodicz_H(Real* Hx, Real* Hy, Real* Hz, int nx, int ny, int nz){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= nx || y >= ny) return;
  int index = y*nx + x;  //large stride, unavoidable
  int index1 = index + (nz - 1)*nx*ny;
  Hx[index1] = Hx[index];
  Hy[index1] = Hy[index];
  Hz[index] = Hz[index1];
}
__global__ void applyPeriodicz_E(Real* Ex, Real* Ey, Real* Ez, int nx, int ny, int nz){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= nx || y >= ny) return;
  int index = y*nx + x;  //large stride, unavoidable
  int index1 = index + (nz - 1)*nx*ny;
  Ez[index1] = Ez[index];
  Ex[index] = Ex[index1];
  Ey[index] = Ey[index1];
}
cuFunc(applySource,(Real* Ez, size_t idx, Real val), (Ez, idx, val), {
  Ez[idx] += val;
})
__global__ void applySourceV(Real* Ez, int nx, int ny, int nz, int pos, Real val){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int z = blockIdx.y * blockDim.y + threadIdx.y;
  if(y >= ny || z >= nz) return;
  int idx = z*nx*ny + y*nx + pos;
  Ez[idx] += val;
}
