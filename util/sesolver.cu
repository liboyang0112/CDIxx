#include <fstream>
#include <cstring>
#include <math.h>
#include "cudaDefs_h.cu"
#include "cudaConfig.hpp"
#include "cuPlotter.hpp"
#include "cub_wrap.hpp"
using namespace std;

__global__ void Hpsifunc(Real* psi, Real* V, Real* Hpsi, int nx, int ny, int nz, Real Eshift)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= nx*ny*nz) return;
  int x = index%nx;
  int y = (index/nx)%ny;
  int z = index/(nx*ny);
  Hpsi[index] = (Eshift+6+V[index])*psi[index];
  if(z < nz-1){
    Hpsi[index] -= psi[index+nx*ny];
  }
  if(z > 0){
    Hpsi[index] -= psi[index-nx*ny];
  }
  if(x < nx-1){
    Hpsi[index] -= psi[index + 1];
  }
  if(x > 0){
    Hpsi[index] -= psi[index - 1];
  }
  if(y < ny-1){
    Hpsi[index] -= psi[index + nx];
  }
  if(y > 0){
    Hpsi[index] -= psi[index - nx];
  }
}
__global__ void initV(Real* V, int cuda_row, int cuda_column, int cuda_height, Real val)
{
  cuda3Idx()
  int &nx = cuda_row;
  int &ny = cuda_column;
  int &nz = cuda_height;
  //Real xmid = nx/2-0.5-2;
  //Real ymid = ny/2-0.5-1;
  //Real zmid = nz/2-0.5;
  //Real r2 = sq(x-xmid) + sq(y-ymid) + sq(z-zmid);
  //V[index] = -val/sqrt(r2);
  Real xmid1 = nx/3-0.5;
  Real ymid1 = ny/3-0.5;
  Real zmid1 = nz/2-0.5;
  Real xmid2 = nx*2/3-0.5;
  Real ymid2 = ny*2/3-0.5;
  Real zmid2 = nz/2-0.5;
  Real r21 = sq(x-xmid1) + sq(y-ymid1) + sq(z-zmid1);
  Real r22 = sq(x-xmid2) + sq(y-ymid2) + sq(z-zmid2);
  V[index] += val/sqrt(r21) + val/sqrt(r22);
  if(r21 < 0.5 || r22 < 0.5) printf("V=%f\n", V[index]);
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
cuFunc(getYZSlice,(Real* slice, Real* data, int nx, int ny, int nz, int ix), (slice, data, nx, ny, nz, ix), {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= ny*nz) return;
  int y = index%ny;
  int z = index/ny;
  int idx = ix+nx*y+nx*ny*z;
  slice[index] = data[idx];
})
int main(){
  init_cuda_image();
  int nsteps = 1000;
  const int nx = 200;
  const int ny = 200;
  const int nz = 200;
  dim3 nblk, nthd;
  //, nblkx,nblky,nblkz, nthd2d;
  //dim3 nthdx_d;
  //dim3 nblkx_d;
  //---------inner dimensions--------
  nthd.x = 256;
  nblk.x = (nx*ny*nz-1)/nthd.x+1;
  //-----boundary dimensions---------
  //nthd2d.x = 256;
  //nblkx.x = (ny*nz-1)/nthd2d.x+1;
  //nblky.x = (nx*nz-1)/nthd2d.x+1;
  //nblkz.x = (nx*ny-1)/nthd2d.x+1;
  //---------PML dimensions----------
  //nthdx_d.x = 256;
  //nblkx_d.x = (n_PML*nx*ny-1)/nthdx_d.x+1;

  size_t nnode = nx*ny*nz;
  size_t memsz = nnode*sizeof(Real);
  Real* psi = (Real*)memMngr.borrowCleanCache(memsz);
  Real* Hpsi = (Real*)memMngr.borrowCleanCache(memsz);
  Real* H2psi = (Real*)memMngr.borrowCleanCache(memsz);
  Real* mom = (Real*)memMngr.borrowCleanCache(memsz);
  Real* V = (Real*)memMngr.borrowCleanCache(memsz);
  //select slice for visualization
  Real* slice = (Real*)memMngr.borrowCache(nx*ny*sizeof(Real));
  Real beta = 0.1;
  initV<<<nblk,nthd>>>(V, nx, ny, nz, -0.2);
  initV<<<nblk,nthd>>>(H2psi, nx, ny, nz, 1);
  resize_cuda_image(nx*ny,nz);
  applyNorm(H2psi, 1./findRootSumSq(H2psi));

  resize_cuda_image(nx,ny);
  plt.init(nx,ny);
  int psivid = plt.initVideo("psi.mp4", 24);
  plt.showVid = -1;//ezvid;
  for(int i = 0; i < nsteps; i++){
    resize_cuda_image(nx*ny,nz);
    add(psi, H2psi, mom, 1-beta);
    if(i > 100) beta = 0.01;
    Hpsifunc<<<nblk,nthd>>>(psi, V, Hpsi, nx, ny, nz, -12);
    myMemcpyD2D(psi,H2psi,nx*ny*nz*sizeof(Real));
    Hpsifunc<<<nblk,nthd>>>(Hpsi, V, H2psi, nx, ny, nz, -12);
    applyNorm(H2psi, 1./findRootSumSq(H2psi));
    add(mom, H2psi, psi, -1);
    if(i%3==0) {
      resize_cuda_image(nx,ny);
      getXYSlice(slice, H2psi , nx, ny, nz/2);
      plt.toVideo = psivid;
      plt.plotFloat(slice, MOD2, 0, 5e4, "",1,0,1,("psi,iter="+to_string(i)).c_str());
    }
  }
  plt.saveVideo(psivid);
  plt.toVideo = -1;
  plt.plotFloat(slice, MOD2, 0, 5e4, "psi",1,0,1);
}
