#include <fstream>
#include <cstring>
#include <math.h>
#include "cudaConfig.h"
#include "memManager.h"
#include "cuPlotter.h"
using namespace std;

__global__ void updateH(Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez, int nx, int ny, int nz)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if(x >= nx || y >= ny || z >= nz) return;
  int index = x + nx*y + nx*ny*z;
  Real mH = 0.5;
  if(z < nz-1 && x > 0 && y < ny-1){
    Hx[index] -= mH*(Ez[index+nx]-Ez[index]-Ey[index+nx*ny]+Ey[index]);
  }
  if(z < nz-1 && x < nx-1 && y > 0){
    Hy[index] -= mH*(Ex[index+nx*ny]-Ex[index]-Ez[index+1]+Ez[index]); //dEz/dy
  }
  if(z > 0 && x < nx-1 && y < ny-1){
    Hz[index] -= mH*(Ey[index+1]-Ey[index]-Ex[index+nx]+Ex[index]); //dEz/dx
  }
}
__global__ void updateE(Real* Hx, Real* Hy, Real* Hz, Real* Ex, Real* Ey, Real* Ez, int nx, int ny, int nz)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if(x >= nx || y >= ny || z >= nz) return;
  int index = x + nx*y + nx*ny*z;
  Real mE = 0.5;
  if(z > 0 && x < nx-1 && y > 0){
    Ex[index] += mE*(Hz[index]-Hz[index-nx]-Hy[index]+Hy[index-nx*ny]); //dEz/dy
  }
  if(z > 0 && x > 0 && y > 0){
    Ey[index] += mE*(Hx[index]-Hx[index-nx*ny]-Hz[index]+Hz[index-1]);
  }
  if(z < nz-1 && x > 0 && y > 0){
    Ez[index] += mE*(Hy[index]-Hy[index-1]-Hx[index]+Hx[index-nx]);
  }
}
__global__ void applyPMLx1(Real* Hx, Real* Hy, Real* Ez, Real* Eprevx1, int interval){
  int y =threadIdx.x;
  int edgeIdx = (y+1)*interval-1;
  Real a = Ez[edgeIdx];
  Hy[edgeIdx-1] += (Eprevx1[y] + a)/2;
  Eprevx1[y] = a;
  Ez[edgeIdx] = 0;
}
__global__ void applyPMLx0(Real* Hx, Real* Hy, Real* Ez, Real* Hprevx0, int interval){
  int y =threadIdx.x;
  int edgeIdx = y*interval;
  Real a = Hy[edgeIdx];
  Ez[edgeIdx+1] -= (a+Hprevx0[y])/2;
  Hprevx0[y] = a;
  Hy[edgeIdx] = 0;
}
__global__ void applyPMLy1(Real* Hx, Real* Hy, Real* Ez, Real* Eprevy1, int interval, int nx){
  int x =threadIdx.x;
  int edgeIdx = interval+x;
  Real a = Ez[edgeIdx];
  Hx[edgeIdx-nx] -= (Eprevy1[x] + a)/2;
  Eprevy1[x] = a;
  Ez[edgeIdx] = 0;
}
__global__ void applyPMLy0(Real* Hx, Real* Hy, Real* Ez, Real* Hprevy0, int nx){
  int x =threadIdx.x;
  Real a = Hx[x];
  Ez[x+nx] += (a+Hprevy0[x])/2;
  Hprevy0[x] = a;
  Hx[x] = 0;
}
__global__ void applyPeriodicx_H(Real* Hx, Real* Hy, int nx){
  int y =threadIdx.x;
  Hx[y*nx] = Hx[(y+1)*nx-1];
  y++;
  Hy[(y+1)*nx-1] = Hy[y*nx];
}
__global__ void applyPeriodicx_E(Real* Ez, int nx){
  int y =threadIdx.x+1;
  Ez[y*nx] = Ez[(y+1)*nx-1];
}
__global__ void applyPeriodicy_H(Real* Hx, Real* Hy, int displace){
  int x =threadIdx.x;
  Hy[x] = Hy[x+displace];
  x++;
  Hx[x+displace] = Hx[x];
}
__global__ void applyPeriodicy_E(Real* Ez, int displace){
  int x =threadIdx.x+1;
  Ez[x] = Ez[x+displace];
}
__global__ void applySource(Real* Ez, size_t idx, Real val){
  Ez[idx] += val;
}
__global__ void applySourceV(Real* Ez, Real* Hy, int nx, int pos, Real val, Real val1){
  int y = threadIdx.x;
  int idx = y*nx + pos;
  Ez[idx] += val;
  Hy[idx-1] += val1;
}
cuFunc(getXYSlice,(Real* slice, Real* data, int nx, int ny, int iz), (slice, data, nx, ny, iz), {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= nx || y >= ny) return;
  int index = x + nx*y;
  slice[index] = data[index+iz*nx*ny];
})
int main(){
  int nsteps = 2000;
  const int nx = 200;
  const int ny = 200;
  const int nz = 200;
  dim3 nblk, nthd, nblkx,nblky,nblkz, nthd2d;
  nthd.x = nthd.y = nthd.z = 8;
  nthd2d.x = nthd2d.y = 16;
  nblk.x = (nx-1)/nthd.x+1;
  nblk.y = (ny-1)/nthd.y+1;
  nblk.z = (nz-1)/nthd.z+1;
  nblkx.x = (ny-1)/nthd2d.x+1;
  nblkx.y = (nz-1)/nthd2d.y+1;
  nblky.x = (nx-1)/nthd2d.x+1;
  nblky.y = (nz-1)/nthd2d.y+1;
  nblkz.x = (nx-1)/nthd2d.x+1;
  nblkz.y = (ny-1)/nthd2d.y+1;

  size_t nnode = nx*ny*nz;
  size_t memsz = nnode*sizeof(Real);
  Real* Hz = (Real*)memMngr.borrowCleanCache(memsz);
  Real* Ex = (Real*)memMngr.borrowCleanCache(memsz);
  Real* Ey = (Real*)memMngr.borrowCleanCache(memsz);
  Real* Ez = (Real*)memMngr.borrowCleanCache(memsz);
  Real* Hx = (Real*)memMngr.borrowCleanCache(memsz);
  Real* Hy = (Real*)memMngr.borrowCleanCache(memsz);
  Real* Eprevx1 = (Real*)memMngr.borrowCleanCache(ny*sizeof(Real));
  Real* Eprevy1 = (Real*)memMngr.borrowCleanCache(ny*sizeof(Real));
  Real* Eprevz1 = (Real*)memMngr.borrowCleanCache(ny*sizeof(Real));
  Real* Hprevx0 = (Real*)memMngr.borrowCleanCache(ny*sizeof(Real));
  Real* Hprevy0 = (Real*)memMngr.borrowCleanCache(ny*sizeof(Real));
  Real* Hprevz0 = (Real*)memMngr.borrowCleanCache(ny*sizeof(Real));
  Real* slice = (Real*)memMngr.borrowCache(nx*ny*sizeof(Real));
  bool saveField = 0;
  int sourcePos = 50+nx*50 + nx*ny*50;
  resize_cuda_image(ny,nx);
  plt.init(ny,nx);
  init_cuda_image();
  
  int ezvid = plt.initVideo("Ez.mp4v");
  int hxvid = plt.initVideo("Hx.mp4v");
  int hyvid = plt.initVideo("Hy.mp4v");
  for(int i = 0; i < nsteps; i++){
    saveField = i%5==0;
    applySource<<<1,1>>>(Ez, sourcePos, 50*sin(M_PI/20*i));//50*exp(-pow(double(i-100)/30,2))); 
    //point source
    //applySourceV<<<1,ny>>>(Ez, Hy, nx, 50, exp(-pow(double(i-100)/30,2)), -exp(-pow(double(i-99.5)/30,2))); //plain wave source
    //applyPMLx1<<<1,ny>>>(Hx, Hy, Ez, Eprevx1, nx);
    //applyPMLy1<<<1,nx>>>(Hx, Hy, Ez, Eprevy1, nx*(ny-1), nx);
    //applyPeriodicy_E<<<1,nx-1>>>(Ez, nx*(ny-1));
    //applyPeriodicx_E<<<1,ny-1>>>(Ez, nx);
    updateH<<<nblk,nthd>>>(Hx, Hy, Hz, Ex, Ey, Ez, nx, ny, nz);
    //applyPMLx0<<<1,ny>>>(Hx, Hy, Ez, Hprevx0, nx);
    //applyPMLy0<<<1,nx>>>(Hx, Hy, Ez, Hprevy0, nx);
    //applyPeriodicx_H<<<1,ny-1>>>(Hx, Hy, nx);
    //applyPeriodicy_H<<<1,nx-1>>>(Hx, Hy, nx*(ny-1));
    updateE<<<nblk,nthd>>>(Hx, Hy, Hz, Ex, Ey, Ez, nx, ny, nz);
    if(saveField) {
      getXYSlice(slice, Ez , nx, ny, 53);
      plt.toVideo = ezvid;
      plt.plotFloat(slice, REAL, 0, 1, "",0,0,1);
      getXYSlice(slice, Hx , nx, ny, 50);
      plt.toVideo = hxvid;
      plt.plotFloat(slice, REAL, 0, 1, ("Hx"+to_string(i)).c_str(),0,0,1);
      getXYSlice(slice, Hx , nx, ny, 50);
      plt.toVideo = hyvid;
      plt.plotFloat(slice, REAL, 0, 1, ("Hy"+to_string(i)).c_str(),0,0,1);
    }
  }
}
