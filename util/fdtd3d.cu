#include <fstream>
#include <cstring>
#include <math.h>
#include "cudaDefs.hpp"
#include "cudaConfig.hpp"
#include "cuPlotter.hpp"
#define n_PML 10
#define k_PML 0.0035
#define secondOrder
#ifdef secondOrder
#define fact1 1./24
#define fact2 9./8
#endif
using namespace std;

const Real b_PML = 1-k_PML*n_PML;

__device__ Real sqSum3(Real a, Real b, Real c){
  return a*a+b*b+c*c;
}
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
__global__ void updateH(Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez, int nx, int ny, int nz)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= nx*ny*nz) return;
  int x = index%nx;
  int y = (index/nx)%ny;
  int z = index/(nx*ny);
  Real ismid = x>=3&&x<nx-3&&y>=3&&y<ny-3&&z>=3&&z<nz-3;
  Real mH = getmH(x,y,z);
  if(z < nz-1 && x > 0 && y < ny-1){
    Real dH = Ez[index+nx]-Ez[index]-Ey[index+nx*ny]+Ey[index];
#ifdef secondOrder
    if(ismid)
      dH = fact1*(Ez[index-nx]-Ez[index+2*nx]+Ey[index+2*nx*ny]-Ey[index-nx*ny]) + fact2*dH;
#endif
    Hx[index] -= mH*dH;
  }
  if(z < nz-1 && x < nx-1 && y > 0){
    Real dH = Ex[index+nx*ny]-Ex[index]-Ez[index+1]+Ez[index];
#ifdef secondOrder
    if(ismid)
      dH = fact1*(Ex[index-nx*ny]-Ex[index+2*nx*ny]+Ez[index+2]-Ez[index-1]) + fact2*dH;
#endif
    Hy[index] -= mH*dH; //dEz/dy
  }
  if(z > 0 && x < nx-1 && y < ny-1){
    Real dH = Ey[index+1]-Ey[index]-Ex[index+nx]+Ex[index];
#ifdef secondOrder
    if(ismid)
      dH = fact1*(Ey[index-1]-Ey[index+2]+Ex[index+2*nx]-Ex[index-nx]) + fact2*dH;
#endif
    Hz[index] -= mH*dH; //dEz/dx
  }
}
__global__ void updateE(Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez, int nx, int ny, int nz)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= nx*ny*nz) return;
  int x = index%nx;
  int y = (index/nx)%ny;
  int z = index/(nx*ny);
  Real mE = getmE(x,y,z);
  Real ismid = x>=3&&x<nx-3&&y>=3&&y<ny-3&&z>=3&&z<nz-3;
  if(z > 0 && x < nx-1 && y > 0){
    Real dE = Hz[index]-Hz[index-nx]-Hy[index]+Hy[index-nx*ny];
#ifdef secondOrder
    if(ismid)
      dE = fact1*(Hz[index-2*nx]-Hz[index+nx]+Hy[index+nx*ny]-Hy[index-2*nx*ny]) + fact2*dE;
#endif
    Ex[index] += mE*dE; //dEz/dy
  }
  if(z > 0 && x > 0 && y > 0){
    Real dE = Hx[index]-Hx[index-nx*ny]-Hz[index]+Hz[index-1];
#ifdef secondOrder
    if(ismid)
      dE = fact1*(Hx[index-2*nx*ny]-Hx[index+nx*ny]+Hz[index+1]-Hz[index-2]) + fact2*dE;
#endif
    Ey[index] += mE*dE;
  }
  if(z < nz-1 && x > 0 && y > 0){
    Real dE = Hy[index]-Hy[index-1]-Hx[index]+Hx[index-nx];
#ifdef secondOrder
    if(ismid)
      dE = fact1*(Hy[index-2]-Hy[index+1]+Hx[index+nx]-Hx[index-2*nx]) + fact2*dE;
#endif
    Ez[index] += mE*dE;
  }
}
__global__ void applyPMLx0_d(Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez, int nx, int ny, int nz){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= nx*ny*n_PML) return;
  int x = index%n_PML;
  int y = (index/nx)%ny;
  int z = index/(nx*ny);
  index = x + 1 + nx*y + nx*ny*z;
  Real sf = b_PML+x*k_PML;
  Ez[index] *= sf;
  Ex[index] *= sf;
  Ey[index] *= sf;
  Hy[index] *= sf;
  Hx[index] *= sf;
  Hz[index] *= sf;
}
__global__ void applyPMLx1_d(Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez, int nx, int ny, int nz){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= nx*ny*n_PML) return;
  int x = index%n_PML;
  int y = (index/nx)%ny;
  int z = index/(nx*ny);
  index = nx-x-2 + nx*y + nx*ny*z;
  Real sf = b_PML+x*k_PML;
  Ez[index] *= sf;
  Ex[index] *= sf;
  Ey[index] *= sf;
  Hy[index] *= sf;
  Hx[index] *= sf;
  Hz[index] *= sf;
}
__global__ void applyPMLx1(Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez, Real* EyBdx1, Real* EzBdx1, int nx, int ny, int nz){
  int edgeIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if(edgeIdx >= ny*nz) return;
  int y = edgeIdx%ny;
  int z = edgeIdx/ny;
  edgeIdx = z*nx*ny + y*nx + nx - 1;  //large stride, unavoidable
  Real mH = getmH(nx-1,y,z);
  Real mE = getmE(nx-1,y,z);
  Real rat = sqrt(mH/mE);
  Real dt = 0.5/(mE*rat)-0.5;
  Real a = Ez[edgeIdx];
  Hy[edgeIdx-1] += rat*(dt*EzBdx1[y+z*ny] + (1-dt)*a);
  EzBdx1[y+z*ny] = a;
  Ez[edgeIdx] = 0;
  Ex[edgeIdx-1] = 0;
  a = Ey[edgeIdx];
  Hz[edgeIdx-1] -= rat*(dt*EyBdx1[y+z*ny] + (1-dt)*a);
  EyBdx1[y+z*ny] = a;
  Ey[edgeIdx] = 0;
}
__global__ void applyPMLx0(Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez, Real* HyBdx0, Real* HzBdx0, int nx, int ny, int nz){
  int edgeIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if(edgeIdx >= ny*nz) return;
  int y = edgeIdx%ny;
  int z = edgeIdx/ny;
  edgeIdx = z*nx*ny + y*nx;  //large stride, unavoidable
  Real a = Hz[edgeIdx];
  Real mH = getmH(0,y,z);
  Real mE = getmE(0,y,z);
  Real rat = sqrt(mE/mH);
  Real dt = 0.5/(rat*mH)-0.5;
  Ey[edgeIdx+1] += rat*(HzBdx0[y+z*ny]*dt + (1-dt)*a);
  HzBdx0[y+z*ny] = a;
  Hz[edgeIdx] = 0;
  a = Hy[edgeIdx];
  Ez[edgeIdx+1] -= rat*(HyBdx0[y+z*ny]*dt + (1-dt)*a);
  HyBdx0[y+z*ny] = a;
  Hy[edgeIdx] = 0;
  Hx[edgeIdx+1] = 0;
}
__global__ void applyPMLy1(Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez, Real* ExBdy1, Real* EzBdy1, int nx, int ny, int nz){
  int edgeIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if(edgeIdx >= nx*nz) return;
  int x = edgeIdx%nx;
  int z = edgeIdx/nx;
  Real mH = getmH(x,ny-1,z);
  Real mE = getmE(x,ny-1,z);
  Real rat = sqrt(mH/mE);
  Real dt = 0.5/(mE*rat)-0.5;
  edgeIdx = nx*ny*z + nx*(ny-1)+ x;
  Real a = Ez[edgeIdx];
  Hx[edgeIdx-nx] -= rat*(dt*EzBdy1[x+nx*z] + (1-dt)*a);
  EzBdy1[x+nx*z] = a;
  Ez[edgeIdx] = 0;
  a = Ex[edgeIdx];
  Hz[edgeIdx-nx] += rat*(dt*ExBdy1[x+nx*z] + (1-dt)*a);
  ExBdy1[x+nx*z] = a;
  Ex[edgeIdx] = 0;
  Ey[edgeIdx-nx] = 0;
  Real sf = b_PML;
  for(int y = 0; y < n_PML; y++) {
    Ez[edgeIdx] *= sf;
    Ex[edgeIdx] *= sf;
    Ey[edgeIdx] *= sf;
    Hy[edgeIdx] *= sf;
    Hx[edgeIdx] *= sf;
    Hz[edgeIdx] *= sf;
    sf+=k_PML;
    edgeIdx-=nx;
  }
}
__global__ void applyPMLy0(Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez, Real* HxBdy0, Real* HzBdy0, int nx, int ny, int nz){
  int edgeIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if(edgeIdx >= nx*nz) return;
  int x = edgeIdx%nx;
  int z = edgeIdx/nx;
  edgeIdx = nx*ny*z + x;
  Real mH = getmH(x,0,z);
  Real mE = getmE(x,0,z);
  Real rat = sqrt(mE/mH);
  Real dt = 0.5/(rat*mH)-0.5;
  Real a = Hz[edgeIdx];
  Ex[edgeIdx+nx] -= rat*(dt*HzBdy0[x+nx*z] + (1-dt)*a);
  HzBdy0[x+nx*z] = a;
  Hz[edgeIdx] = 0;
  a = Hx[edgeIdx];  // change to Ez[index] in cr
  Ez[edgeIdx+nx] += rat*(dt*HxBdy0[x+nx*z] + (1-dt)*a);
  HxBdy0[x+nx*z] = a;
  Hx[edgeIdx] = 0;
  Hy[edgeIdx+nx] = 0;
  Real sf = b_PML;
  for(int y = 0; y < n_PML; y++) {
    Ez[edgeIdx] *= sf;
    Ey[edgeIdx] *= sf;
    Ex[edgeIdx] *= sf;
    Hy[edgeIdx] *= sf;
    Hx[edgeIdx] *= sf;
    Hz[edgeIdx] *= sf;
    edgeIdx+=nx;
    sf += k_PML;
  }
}
__global__ void applyPMLz1(Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez, Real* ExBdz1, Real* EyBdz1, int nx, int ny, int nz){
  int edgeIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if(edgeIdx >= nx*ny) return;
  int x = edgeIdx%ny;
  int y = edgeIdx/ny;
  int step = nx*ny;
  edgeIdx = (nz-1)*step + nx*y + x;
  Real mH = getmH(x,y,nz-1);
  Real mE = getmE(x,y,nz-1);
  Real rat = sqrt(mH/mE);
  Real dt = 0.5/(mE*rat)-0.5;
  int idxp1 = edgeIdx-step;
  Real a = Ex[edgeIdx];
  Hy[idxp1] -= rat*(dt*ExBdz1[x+nx*y] + (1-dt)*a);
  ExBdz1[x+nx*y] = a;
  Ex[edgeIdx] = 0;
  a = Ey[edgeIdx];
  Hx[idxp1] += rat*(dt*EyBdz1[x+nx*y] + (1-dt)*a);
  EyBdz1[x+nx*y] = a;
  Ey[edgeIdx] = 0;
  Ez[idxp1] = 0;
  Real sf = b_PML;
  for(int z = 0; z < n_PML; z++) {
    Ez[edgeIdx] *= sf;
    Ey[edgeIdx] *= sf;
    Ex[edgeIdx] *= sf;
    Hy[edgeIdx] *= sf;
    Hx[edgeIdx] *= sf;
    Hz[edgeIdx] *= sf;
    edgeIdx-=step;
    sf+=k_PML;
  }
}
__global__ void applyPMLz0(Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez, Real* HxBdz0, Real* HyBdz0, int nx, int ny, int nz){
  int edgeIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if(edgeIdx >= nx*ny) return;
  int x = edgeIdx%ny;
  int y = edgeIdx/ny;
  int step = nx*ny;
  edgeIdx = nx*y + x;
  Real mH = getmH(x,y,0);
  Real mE = getmE(x,y,0);
  Real rat = sqrt(mE/mH);
  Real dt = 0.5/(rat*mH)-0.5;
  int idxp1 = edgeIdx+step;
  Real a = Hx[edgeIdx];
  Ey[idxp1] -= rat*(dt*HxBdz0[x+nx*y] + (1-dt)*a);
  HxBdz0[x+nx*y] = a;
  Hx[edgeIdx] = 0;
  a = Hy[edgeIdx];
  Ex[idxp1] += rat*(dt*HyBdz0[x+nx*y] + (1-dt)*a);
  HyBdz0[x+nx*y] = a;
  Hy[edgeIdx] = 0;
  Hz[idxp1] = 0;
  Real sf = b_PML;
  for(int z = 0; z < n_PML; z++) {
    Ez[edgeIdx] *= sf;
    Ey[edgeIdx] *= sf;
    Ex[edgeIdx] *= sf;
    Hy[edgeIdx] *= sf;
    Hx[edgeIdx] *= sf;
    Hz[edgeIdx] *= sf;
    edgeIdx+=step;
    sf+=k_PML;
  }
}
__global__ void applyPeriodicx_H(Real* Hx, Real* Hy, Real* Hz, int nx, int ny, int nz){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int z = blockIdx.y * blockDim.y + threadIdx.y;
  if(y >= ny || z >= nz) return;
  int edgeIdx = z*nx*ny + y*nx;  //large stride, unavoidable
  int edgeIdx1 = edgeIdx + nx - 1;
  Hx[edgeIdx] = Hx[edgeIdx1];
  Hy[edgeIdx1] = Hy[edgeIdx];
  Hz[edgeIdx1] = Hz[edgeIdx];
}
__global__ void applyPeriodicx_E(Real* Ex, Real* Ey, Real* Ez, int nx, int ny, int nz){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int z = blockIdx.y * blockDim.y + threadIdx.y;
  if(y >= ny || z >= nz) return;
  int edgeIdx = z*nx*ny + y*nx;  //large stride, unavoidable
  int edgeIdx1 = edgeIdx + nx - 1;
  Ez[edgeIdx] = Ez[edgeIdx1];
  Ey[edgeIdx] = Ey[edgeIdx1];
  Ex[edgeIdx1] = Ex[edgeIdx];
}
__global__ void applyPeriodicy_H(Real* Hx, Real* Hy, Real* Hz, int nx, int ny, int nz){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int z = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= nx || z >= nz) return;
  int edgeIdx = z*nx*ny + x;  //large stride, unavoidable
  int edgeIdx1 = edgeIdx + (ny - 1)*nx;
  Hx[edgeIdx1] = Hx[edgeIdx];
  Hy[edgeIdx] = Hy[edgeIdx1];
  Hz[edgeIdx1] = Hz[edgeIdx];
}
__global__ void applyPeriodicy_E(Real* Ex, Real* Ey, Real* Ez, int nx, int ny, int nz){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int z = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= nx || z >= nz) return;
  int edgeIdx = z*nx*ny + x;  //large stride, unavoidable
  int edgeIdx1 = edgeIdx + (ny - 1)*nx;
  Ez[edgeIdx] = Ez[edgeIdx1];
  Ex[edgeIdx] = Ex[edgeIdx1];
  Ey[edgeIdx1] = Ey[edgeIdx];
}
__global__ void applyPeriodicz_H(Real* Hx, Real* Hy, Real* Hz, int nx, int ny, int nz){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= nx || y >= ny) return;
  int edgeIdx = y*nx + x;  //large stride, unavoidable
  int edgeIdx1 = edgeIdx + (nz - 1)*nx*ny;
  Hx[edgeIdx1] = Hx[edgeIdx];
  Hy[edgeIdx1] = Hy[edgeIdx];
  Hz[edgeIdx] = Hz[edgeIdx1];
}
__global__ void applyPeriodicz_E(Real* Ex, Real* Ey, Real* Ez, int nx, int ny, int nz){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= nx || y >= ny) return;
  int edgeIdx = y*nx + x;  //large stride, unavoidable
  int edgeIdx1 = edgeIdx + (nz - 1)*nx*ny;
  Ez[edgeIdx1] = Ez[edgeIdx];
  Ex[edgeIdx] = Ex[edgeIdx1];
  Ey[edgeIdx] = Ey[edgeIdx1];
}
__global__ void applySource(Real* Ez, size_t idx, Real val){
  Ez[idx] += val;
}
__global__ void applySourceV(Real* Ez, int nx, int ny, int nz, int pos, Real val){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int z = blockIdx.y * blockDim.y + threadIdx.y;
  if(y >= ny || z >= nz) return;
  int idx = z*nx*ny + y*nx + pos;
  Ez[idx] += val;
}
cuFunc(getXYSlice,(Real* slice, Real* data, int nx, int ny, int iz), (slice, data, nx, ny, iz), {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= nx*ny) return;
  slice[index] = data[index+iz*nx*ny];
})
cuFunc(getXZSlice,(Real* slice, Real* data, int nx, int ny, int nz, int iy), (slice, data, nx, ny, nz, iy), {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= nx*nz) return;
  int x = index%nx;
  int z = index/nx;
  slice[index] = data[x+nx*iy+nx*ny*z];
})
cuFunc(getYZSlice,(Real* slice, Real* data, Real* data2, Real* data3, int nx, int ny, int nz, int ix), (slice, data, data2, data3, nx, ny, nz, ix), {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= ny*nz) return;
  int y = index%ny;
  int z = index/ny;
  int idx = ix+nx*y+nx*ny*z;
  //slice[index] = sq(data[idx])+sq(data2[idx])+sq(data3[idx]);
  slice[index] = data2[idx];
  //slice[index] = data2[idx]*data2[idx];
})
int main(){
  int nsteps = 1000;
  const int nx = 200;
  const int ny = 200;
  const int nz = 200;
  dim3 nblk, nthd, nblkx,nblky,nblkz, nthd2d;
  dim3 nthdx_d;
  dim3 nblkx_d;
  //---------inner dimensions--------
  nthd.x = 256;
  nblk.x = (nx*ny*nz-1)/nthd.x+1;
  //-----boundary dimensions---------
  nthd2d.x = 256;
  nblkx.x = (ny*nz-1)/nthd2d.x+1;
  nblky.x = (nx*nz-1)/nthd2d.x+1;
  nblkz.x = (nx*ny-1)/nthd2d.x+1;
  //---------PML dimensions----------
  nthdx_d.x = 256;
  nblkx_d.x = (n_PML*nx*ny-1)/nthdx_d.x+1;

  size_t nnode = nx*ny*nz;
  size_t memsz = nnode*sizeof(Real);
  Real* Hz = (Real*)memMngr.borrowCleanCache(memsz);
  Real* Hx = (Real*)memMngr.borrowCleanCache(memsz);
  Real* Hy = (Real*)memMngr.borrowCleanCache(memsz);
  Real* Ex = (Real*)memMngr.borrowCleanCache(memsz);
  Real* Ey = (Real*)memMngr.borrowCleanCache(memsz);
  Real* Ez = (Real*)memMngr.borrowCleanCache(memsz);
  Real* Px = (Real*)memMngr.borrowCleanCache(memsz);  //polarization
  Real* Py = (Real*)memMngr.borrowCleanCache(memsz);
  Real* Pz = (Real*)memMngr.borrowCleanCache(memsz);
  Real* dPx = (Real*)memMngr.borrowCleanCache(memsz); //time derivative of polarization
  Real* dPy = (Real*)memMngr.borrowCleanCache(memsz);
  Real* dPz = (Real*)memMngr.borrowCleanCache(memsz);

  unsigned char* material_map = (unsigned char*)memMngr.borrowCleanCache(nnode);  //supports 255 kinds of materials
  //record boundaries for PML
  Real* EzBdx1 = (Real*)memMngr.borrowCleanCache(ny*nz*sizeof(Real));
  Real* EyBdx1 = (Real*)memMngr.borrowCleanCache(ny*nz*sizeof(Real));
  Real* EzBdy1 = (Real*)memMngr.borrowCleanCache(nz*nx*sizeof(Real));
  Real* ExBdy1 = (Real*)memMngr.borrowCleanCache(nz*nx*sizeof(Real));
  Real* ExBdz1 = (Real*)memMngr.borrowCleanCache(nx*ny*sizeof(Real));
  Real* EyBdz1 = (Real*)memMngr.borrowCleanCache(nx*ny*sizeof(Real));
  Real* HzBdx0 = (Real*)memMngr.borrowCleanCache(ny*nz*sizeof(Real));
  Real* HyBdx0 = (Real*)memMngr.borrowCleanCache(ny*nz*sizeof(Real));
  Real* HzBdy0 = (Real*)memMngr.borrowCleanCache(nz*nx*sizeof(Real));
  Real* HxBdy0 = (Real*)memMngr.borrowCleanCache(nz*nx*sizeof(Real));
  Real* HxBdz0 = (Real*)memMngr.borrowCleanCache(nx*ny*sizeof(Real));
  Real* HyBdz0 = (Real*)memMngr.borrowCleanCache(nx*ny*sizeof(Real));
  //select slice for visualization
  Real* slice = (Real*)memMngr.borrowCache(nx*ny*sizeof(Real));

  bool saveField = 0;
  int sourcePos = 100+nx*100 + nx*ny*100;
  resize_cuda_image(nx,ny);
  plt.init(nx,ny);
  init_cuda_image();
  
  int ezvid = plt.initVideo("Ez.mp4", 24);
  int hxvid = plt.initVideo("Hx.mp4", 24);
  int hyvid = plt.initVideo("Hy.mp4", 24);
  plt.showVid = -1;//ezvid;
  
  for(int i = 0; i < nsteps; i++){
    saveField = i%5==0;
    //applySource<<<1,1>>>(Ez, sourcePos, 20*sin(M_PI/70*i));//50*exp(-sq(double(i-100)/30))); 
    if(i < 280) {
      applySource<<<1,1>>>(Ez, sourcePos, 500*exp(-sq((i-140.)/70))*(sin(M_PI/35*i)));//50*exp(-sq(double(i-100)/30,2))); 
      applySource<<<1,1>>>(Ey, sourcePos, 500*exp(-sq((i-140.)/70))*(cos(M_PI/35*i)));//50*exp(-sq(double(i-100)/30,2))); 
    }//get circular polarized source!
    //applySourceV<<<nblkx,nthd2d>>>(Ez, nx, ny, nz, 100, 5*sin(M_PI/30*i));

    applyPMLx1<<<nblkx,nthd2d>>>(Hx, Hy, Hz, Ex, Ey, Ez, EyBdx1, EzBdx1, nx, ny, nz);
    applyPMLx1_d<<<nblkx_d,nthdx_d>>>(Hx, Hy, Hz, Ex, Ey, Ez, nx, ny, nz);  //do separately to speed up
    applyPMLy1<<<nblky,nthd2d>>>(Hx, Hy, Hz, Ex, Ey, Ez, ExBdy1, EzBdy1, nx, ny, nz);
    applyPMLz1<<<nblkz,nthd2d>>>(Hx, Hy, Hz, Ex, Ey, Ez, ExBdz1, EyBdz1, nx, ny, nz);

    //applyPeriodicx_E<<<nblkx, nthd2d>>>(Ex, Ey, Ez, nx, ny, nz);
    //applyPeriodicy_E<<<nblky, nthd2d>>>(Ex, Ey, Ez, nx, ny, nz);
    //applyPeriodicz_E<<<nblky, nthd2d>>>(Ex, Ey, Ez, nx, ny, nz);

    updateH<<<nblk,nthd>>>(Hx, Hy, Hz, Ex, Ey, Ez, nx, ny, nz);    //------------UPDATE H-----------

    //applyPeriodicx_H<<<nblkx, nthd2d>>>(Hx, Hy, Hz, nx, ny, nz);
    //applyPeriodicy_H<<<nblkx, nthd2d>>>(Hx, Hy, Hz, nx, ny, nz);
    //applyPeriodicz_H<<<nblkx, nthd2d>>>(Hx, Hy, Hz, nx, ny, nz);

    applyPMLx0<<<nblkx,nthd2d>>>(Hx, Hy, Hz, Ex, Ey, Ez, HyBdx0, HzBdx0, nx, ny, nz);
    applyPMLx0_d<<<nblkx_d,nthdx_d>>>(Hx, Hy, Hz, Ex, Ey, Ez, nx, ny, nz);
    applyPMLy0<<<nblky,nthd2d>>>(Hx, Hy, Hz, Ex, Ey, Ez, HxBdy0, HzBdy0, nx, ny, nz);
    applyPMLz0<<<nblkz,nthd2d>>>(Hx, Hy, Hz, Ex, Ey, Ez, HxBdz0, HyBdz0, nx, ny, nz);

    updateE<<<nblk,nthd>>>(Hx, Hy, Hz, Ex, Ey, Ez, nx, ny, nz);  //------------UPDATE E-----------
    
    if(saveField) {
      getYZSlice(slice, Ex, Ey, Ez , nx, ny, nz, 150);
      //getXZSlice(slice, Ey , nx, ny, nz, 100);
      plt.toVideo = ezvid;
      plt.plotFloat(slice, REAL, 0, 1, "",0,0,1, ("Ez,t="+to_string(i)).c_str());
      getXYSlice(slice, Hx , nx, ny, 100);
      plt.toVideo = hxvid;
      plt.plotFloat(slice, REAL, 0, 1, "",0,0,1,("Hx,t="+to_string(i)).c_str());
      getXYSlice(slice, Hy , nx, ny, 100);
      plt.toVideo = hyvid;
      plt.plotFloat(slice, REAL, 0, 1, "",0,0,1,("Hy,t="+to_string(i)).c_str());
    }
  }
  plt.saveVideo(ezvid);
  plt.saveVideo(hxvid);
  plt.saveVideo(hyvid);
}
