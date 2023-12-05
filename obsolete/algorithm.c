#include <stdio.h>
#include <stdlib.h>
void clustering(double* datax, double* dataweight, int ndata, double* clusterx, double* clusterweight, int ncluster){
  double interval = (datax[ndata-1] - datax[0])/ncluster;
//  FILE *dataout = fopen("spectra_raw.txt", "w");
//  FILE *clustout = fopen("spectra.txt", "w");
//  for(int j = 0; j < ndata; j++){
//    fprintf(dataout, "%f %f\n", datax[j], dataweight[j]);
//  }
  clusterx[0] = datax[0] + interval/2;
  for(int i = 1; i < ncluster; i++){
    clusterx[i] = clusterx[i-1] + interval;
    //clusterx[i] = 0;
  }
  int niter = 100;
  double stepsize = 1e-3;
  for(int i = 0; i < niter; i++){
    int icluster = 0;
    double endpoint = (clusterx[0] + clusterx[1])/2;
    double dx = 0;
    for(int j = 0; j < ndata; j++){
      //dx -= 2*stepsize*dataweight[j]*((clusterx[icluster] - datax[j])>0?1:-1);
      dx -= 2*stepsize*dataweight[j]*(clusterx[icluster] - datax[j]);
      if(datax[j] > endpoint){
        if(icluster < ncluster-2 ) endpoint = (clusterx[icluster+1] + clusterx[icluster+2])/2;
        else endpoint = datax[ndata-2];
        clusterx[icluster++] += dx;
        dx = 0;
      }
    }
  }
  int icluster = 0;
  double endpoint = (clusterx[0] + clusterx[1])/2;
  double dx = 0;
  int npoints = 0;
  for(int j = 0; j < ndata; j++){
    dx += dataweight[j];
    npoints++;
    if(datax[j] > endpoint){
      if(icluster < ncluster-2 ) endpoint = (clusterx[icluster+1] + clusterx[icluster+2])/2;
      else endpoint = datax[ndata-2];
      clusterweight[icluster++] = dx / npoints;
      npoints = 0;
      dx = 0;
    }
  }
//  for(int j = 0; j < ncluster; j++){
//    fprintf(clustout, "%f %f\n", clusterx[j], clusterweight[j]);
//  }
}
/*
int main(){
  double datax[10] = {1,2,3,4,5,6,7,8,9,10};
  double weights[10] = {1,2,30,4,5,6,70,8,9,10};
  double clusterx[3] = {1,2,3};
  double clusterweights[3] = {1,2,3};
  clustering(datax, weights, 10, clusterx, clusterweights, 3);
  printf("clustering results: %f, %f, %f\n", clusterx[0], clusterx[1], clusterx[2]);
}
*/
