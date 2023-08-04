#include <fstream>
#include <cstring>
#include <math.h>
#include "cudaConfig.h"
#include "memManager.h"
#include "cuPlotter.h"
#define k_PML 1./20
#define n_PML 15
using namespace std;

#define mE 0.3
#define mH 0.3

cuFunc(updateH, (Real* Hx, Real* Hy, Real* Ez), (Hx, Hy, Ez),
    {
    cudaIdx(); //x and y is flipped, cuda_row = ny, cuda_column = nx
    if(x >= cuda_row || y >= cuda_column) return;
    if(x >= 1 && y < cuda_column-1){
      Hy[index] += mH*(Ez[index+1]-Ez[index]); //dEz/dx
    }
    if(x < cuda_row-1 && y >= 1){
      Hx[index] -= mH*(Ez[index+cuda_column]-Ez[index]); //dEz/dy
    }
    }
    );
cuFunc(updateE, (Real* Hx, Real* Hy, Real* Ez), (Hx, Hy, Ez),{
    cudaIdx(); 
    if(x != 0 && y!=0 && x < cuda_row && y < cuda_column){
      Ez[index] += mE*(Hy[index]-Hy[index-1]-Hx[index]+Hx[index-cuda_column]);
    }
    })
__global__ void applyPMLx1(Real* Hx, Real* Hy, Real* Ez, Real* Eprevx1, int nx){
  int y =threadIdx.x;
  int edgeIdx = (y+2)*nx-1;
  float dt = 0.5/sqrt(mE*mH)-0.5;
  Real a = Ez[edgeIdx];
  Hy[edgeIdx-1] += sqrt(mH/mE)*(dt*Eprevx1[y] + (1-dt)*a);
  Eprevx1[y] = a;
  Ez[edgeIdx] = 0;
  Hx[edgeIdx] = 0;
}
__global__ void applyprePMLx0(Real* Hx, Real* Hy, Real* dEz, Real* dHx, int nx){
  int y =threadIdx.x;
  int edgeIdx = (y+1)*nx+10;
  if(Hy[edgeIdx]*Hy[edgeIdx+1]>0) dHx[y] = -Hy[edgeIdx]/(Hy[edgeIdx]+Hy[edgeIdx+1])*(Hx[edgeIdx+1]+Hx[edgeIdx-nx+1]);
  else dHx[y] = 0;
  int sgn = 0;
  if(Hy[edgeIdx] > 0) sgn = 1;
  if(Hy[edgeIdx] < 0) sgn = -1;
  if(y == 160) printf("dEz= %f, %f\n", Hy[edgeIdx]+Hy[edgeIdx+1], Hx[edgeIdx+1]+Hx[edgeIdx-nx+1]);
  dEz[y] = -sgn*hypot(dHx[y],Hy[edgeIdx]);
}
__global__ void applyPMLx0(Real* Hx, Real* Hy, Real* Ez, Real* Hprevx0, Real* dEz, Real* dHx, int nx){
  int y =threadIdx.x;
  int edgeIdx = (y+1)*nx+10;
  Ez[edgeIdx+1] += dEz[y];
  Hx[edgeIdx+1] += (dHx[y]+dHx[y+1])/2;
  Hy[edgeIdx] = 0;
}
__global__ void applyPMLx0(Real* Hx, Real* Hy, Real* Ez, Real* Hprevx0, int nx){
  int y =threadIdx.x;
  int edgeIdx = y*nx;
  float dt = 0.5/sqrt(mE*mH)-0.5;
  Real a = Hy[edgeIdx];
  Ez[edgeIdx+1] -= sqrt(mE/mH)*((1-dt)*a+dt*Hprevx0[y]);
  Hprevx0[y] = a;
  Hy[edgeIdx] = 0;
  Hx[edgeIdx+1] = 0;
}
__global__ void applyPMLy1(Real* Hx, Real* Hy, Real* Ez, Real* Eprevy1, int ny, int nx){
  int x =threadIdx.x;
  int edgeIdx = (ny-1)*nx+x;
  Real a = Ez[edgeIdx];
  Hx[edgeIdx-nx] -= (Eprevy1[x] + a)/2;
  Eprevy1[x] = a;
  Ez[edgeIdx] = 0;
  for(float y = 0; y < n_PML; y++) {
    Ez[edgeIdx-nx*int(y)] *= y/n_PML*k_PML+1-k_PML;
    Hy[edgeIdx-nx*int(y)] *= y/n_PML*k_PML+1-k_PML;
    Hx[edgeIdx-nx*int(y)] *= y/n_PML*k_PML+1-k_PML;
  }
}
__global__ void applyPMLy0(Real* Hx, Real* Hy, Real* Ez, Real* Hprevy0, int nx){
  int x =threadIdx.x;
  Real a = Hx[x];
  Ez[x+nx] += (a+Hprevy0[x])/2;
  Hprevy0[x] = a;
  Hx[x] = 0;
  Hy[x+nx] = 0;
  for(float y = 0; y < n_PML; y++) {
    Ez[x+int(y)*nx] *= y/n_PML*k_PML+1-k_PML;
    Hy[x+int(y)*nx] *= y/n_PML*k_PML+1-k_PML;
    Hx[x+int(y)*nx] *= y/n_PML*k_PML+1-k_PML;
  }
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
__global__ void applySource(Real* Ez, Real* Hy, Real* Hx, size_t idx, Real val, Real val1, Real val2, int nx, int ny, int nz){
  Ez[idx] += val;
  //Hy[idx] += val1;
  Hy[idx] += val1/sqrtf(2);
  Hx[idx] += val1/sqrtf(2);
}
__global__ void applySourceV(Real* Ez, Real* Hy, int nx, int pos, Real val, Real val1){
  int y = threadIdx.x;
  int idx = y*nx + pos;
  Ez[idx] += val;
  Hy[idx-1] += val1;
}
int main(){
  int nsteps = 1000;
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
  //Real* dEz, *dHx;
  if(nz != 1 || !isTE){
    Ez = (Real*)memMngr.borrowCleanCache(memsz);
    Hx = (Real*)memMngr.borrowCleanCache(memsz);
    Hy = (Real*)memMngr.borrowCleanCache(memsz);
    Eprevx1 = (Real*)memMngr.borrowCleanCache(ny*sizeof(Real));
    Eprevy1 = (Real*)memMngr.borrowCleanCache(ny*sizeof(Real));
    Hprevx0 = (Real*)memMngr.borrowCleanCache(ny*sizeof(Real));
    Hprevy0 = (Real*)memMngr.borrowCleanCache(ny*sizeof(Real));
    //dEz = (Real*)memMngr.borrowCleanCache(ny*sizeof(Real));
    //dHx = (Real*)memMngr.borrowCleanCache(ny*sizeof(Real));
  }
  bool saveField = 0;
  int sourcePos = 100+nx*10;
  resize_cuda_image(ny,nx);
  plt.init(ny,nx);
  init_cuda_image();
  int ezvid = plt.initVideo("Ez.mp4","avc1",24);
  int hxvid = plt.initVideo("Hx.mp4","avc1",24);
  int hyvid = plt.initVideo("Hy.mp4","avc1",24);
  Real dt1 = 0.5-0.5/sqrt(mE*mH);
  Real dt2 = 0.5-sqrt(1.25/mE*mH);
  for(int i = 0; i < nsteps; i++){
    saveField = i%5==0;
    //applySource<<<1,1>>>(Ez, sourcePos, 0.3*sin(M_PI/20*i)*exp(-pow(double(i-100)/30,2))); //point source
    //applySourceV<<<1,ny>>>(Ez, Hy, nx, 50, exp(-pow(double(i-100)/30,2)), -exp(-pow(double(i-99.5)/30,2))); //plain wave source
    applyPMLx1<<<1,ny-1>>>(Hx, Hy, Ez, Eprevx1, nx);
    applyPMLy1<<<1,nx>>>(Hx, Hy, Ez, Eprevy1, ny, nx);
    //applyPeriodicy_E<<<1,nx-1>>>(Ez, nx*(ny-1));
    //applyPeriodicx_E<<<1,ny-1>>>(Ez, nx);
    //applySource<<<1,1>>>(Ez, Hy, Hx, sourcePos, exp(-pow(double(i-100)/30,2)), exp(-pow(double(i-100-dt1)/30,2)),exp(-pow(double(i-100-dt2)/30,2)),nx,ny,nz); //point source
    applySource<<<1,1>>>(Ez, sourcePos, 10*exp(-pow(double(i-100)/20,2))*sin(M_PI/50*i));//50*exp(-pow(double(i-100)/30,2))); 
    updateH(Hx, Hy, Ez);
    //applyprePMLx1<<<1,ny-1>>>(Hx, Hy, Ez, Eprevx1, nx, ny);
    //applyprePMLx0<<<1,ny-1>>>(Hx, Hy, Ez, Eprevx1, nx, ny);
    //applyprePMLx0<<<1,ny-2>>>(Hx, Hy, dEz, dHx, nx);
    //applyPMLx0<<<1,ny-1>>>(Hx, Hy, Ez, Hprevx0, dEz, dHx, nx);
    applyPMLx0<<<1,ny>>>(Hx, Hy, Ez, Hprevx0, nx);
    applyPMLy0<<<1,nx>>>(Hx, Hy, Ez, Hprevy0, nx);
    //applyPeriodicx_H<<<1,ny-1>>>(Hx, Hy, nx);
    //applyPeriodicy_H<<<1,nx-1>>>(Hx, Hy, nx*(ny-1));
    updateE(Hx, Hy, Ez);
    if(saveField) {
      plt.toVideo = ezvid;
      plt.plotFloat(Ez, REAL, 0, 10, ("Ez"+to_string(i)).c_str(),0,0,1);
      plt.toVideo = hxvid;
      plt.plotFloat(Hx, REAL, 0, 10, ("Hx"+to_string(i)).c_str(),0,0,1);
      plt.toVideo = hyvid;
      plt.plotFloat(Hy, REAL, 0, 10, ("Hy"+to_string(i)).c_str(),0,0,1);
    }
  }
  plt.saveVideo(ezvid);
  plt.saveVideo(hxvid);
  plt.saveVideo(hyvid);
}
