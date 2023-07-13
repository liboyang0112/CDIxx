#include <fstream>
#include <cstring>
#include <math.h>
#include "cudaConfig.h"
#include "memManager.h"
#include "cuPlotter.h"
using namespace std;

cuFunc(updateH, (Real* Hx, Real* Hy, Real* Ez), (Hx, Hy, Ez),
    {
    cudaIdx(); //x and y is flipped, cuda_row = ny, cuda_column = nx
    Real mH = 0.5;
    if(x >= cuda_row || y >= cuda_column) return;
    if(x >= 1 && y < cuda_column-1){
      Hy[index] -= mH*(Ez[index+1]-Ez[index]); //dEz/dx
    }
    if(x < cuda_row-1 && y >= 1){
      Hx[index] += mH*(Ez[index+cuda_column]-Ez[index]); //dEz/dy
    }
    }
    );
cuFunc(updateE, (Real* Hx, Real* Hy, Real* Ez), (Hx, Hy, Ez),{
    cudaIdx(); //x and y is flipped, cuda_row = ny, cuda_column = nx
    Real mE = 0.5;
    //if(x < 180) mE = 0.3;
    if(x != 0 && y!=0 && x < cuda_row && y < cuda_column){
      Ez[index] -= mE*(Hy[index]-Hy[index-1]);
      Ez[index] -= mE*(-Hx[index]+Hx[index-cuda_column]);
    }
    })
__global__ void applyPMLx1(Real* Hx, Real* Hy, Real* Ez, Real* Eprevx1, int interval){
  int y =threadIdx.x;
  int edgeIdx = (y+1)*interval-1;
  Real a = Ez[edgeIdx];
  Hy[edgeIdx-1] -= (Eprevx1[y] + a)/2;
  Eprevx1[y] = a;
  Ez[edgeIdx] = 0;
}
__global__ void applyPMLx0(Real* Hx, Real* Hy, Real* Ez, Real* Hprevx0, int interval){
  int y =threadIdx.x;
  int edgeIdx = y*interval;
  Real a = Hy[edgeIdx];
  Ez[edgeIdx+1] += (a+Hprevx0[y])/2;
  Hprevx0[y] = a;
  Hy[edgeIdx] = 0;
}
__global__ void applyPMLy1(Real* Hx, Real* Hy, Real* Ez, Real* Eprevy1, int interval, int nx){
  int x =threadIdx.x;
  int edgeIdx = interval+x;
  Real a = Ez[edgeIdx];
  Hx[edgeIdx-nx] += (Eprevy1[x] + a)/2;
  Eprevy1[x] = a;
  Ez[edgeIdx] = 0;
}
__global__ void applyPMLy0(Real* Hx, Real* Hy, Real* Ez, Real* Hprevy0, int nx){
  int x =threadIdx.x;
  Real a = Hx[x];
  Ez[x+nx] -= (a+Hprevy0[x])/2;
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
int main(){
  int nsteps = 2000;
  const int nx = 200;
  const int ny = 200;
  const int nz = 1;
  const bool isTE = 0;
  Real* Hx, *Hy, *Ez; // TE mode, index = ix + nx * iy + nx * ny * iz
  //Real* Hz, *Ex, *Ey; // TM mode
  size_t nnode = nx*ny*nz;
  size_t memsz = nnode*sizeof(Real);
  Real *Eprevx1,*Eprevy1;
  Real *Hprevx0,*Hprevy0;
  //Real *Eprevz1,*Hprevz0; 
  if(nz != 1 || isTE){
    //Hz = (Real*)memMngr.borrowCleanCache(memsz);
    //Ex = (Real*)memMngr.borrowCleanCache(memsz);
    //Ey = (Real*)memMngr.borrowCleanCache(memsz);
  }
  if(nz != 1 || !isTE){
    Ez = (Real*)memMngr.borrowCleanCache(memsz);
    Hx = (Real*)memMngr.borrowCleanCache(memsz);
    Hy = (Real*)memMngr.borrowCleanCache(memsz);
    Eprevx1 = (Real*)memMngr.borrowCleanCache(ny*sizeof(Real));
    Eprevy1 = (Real*)memMngr.borrowCleanCache(ny*sizeof(Real));
    Hprevx0 = (Real*)memMngr.borrowCleanCache(ny*sizeof(Real));
    Hprevy0 = (Real*)memMngr.borrowCleanCache(ny*sizeof(Real));
  }
  bool saveField = 0;
  int sourcePos = 50+nx*50;
  resize_cuda_image(ny,nx);
  plt.init(ny,nx);
  init_cuda_image();
  int ezvid = plt.initVideo("Ez.mp4v");
  int hxvid = plt.initVideo("Hx.mp4v");
  int hyvid = plt.initVideo("Hy.mp4v");
  for(int i = 0; i < nsteps; i++){
    saveField = i%5==0;
    applySource<<<1,1>>>(Ez, sourcePos, 50*exp(-pow(double(i-100)/30,2))); //point source
    //applySourceV<<<1,ny>>>(Ez, Hy, nx, 50, exp(-pow(double(i-100)/30,2)), -exp(-pow(double(i-99.5)/30,2))); //plain wave source
    applyPMLx1<<<1,ny>>>(Hx, Hy, Ez, Eprevx1, nx);
    applyPMLy1<<<1,nx>>>(Hx, Hy, Ez, Eprevy1, nx*(ny-1), nx);
    //applyPeriodicy_E<<<1,nx-1>>>(Ez, nx*(ny-1));
    //applyPeriodicx_E<<<1,ny-1>>>(Ez, nx);
    updateH(Hx, Hy, Ez);
    applyPMLx0<<<1,ny>>>(Hx, Hy, Ez, Hprevx0, nx);
    applyPMLy0<<<1,nx>>>(Hx, Hy, Ez, Hprevy0, nx);
    //applyPeriodicx_H<<<1,ny-1>>>(Hx, Hy, nx);
    //applyPeriodicy_H<<<1,nx-1>>>(Hx, Hy, nx*(ny-1));
    updateE(Hx, Hy, Ez);
    if(saveField) {
      plt.toVideo = ezvid;
      plt.plotFloat(Ez, REAL, 0, 1, ("Ez"+to_string(i)).c_str(),0,0,1);
      plt.toVideo = hxvid;
      plt.plotFloat(Hx, REAL, 0, 1, ("Hx"+to_string(i)).c_str(),0,0,1);
      plt.toVideo = hyvid;
      plt.plotFloat(Hy, REAL, 0, 1, ("Hy"+to_string(i)).c_str(),0,0,1);
    }
  }
}
